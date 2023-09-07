import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def country_region_mapping(df):
    # Country code to region mapping
    country_to_region = {}
    country_to_region.update({code: "East Asia & Pacific" for code in {"BD", "KH", "ID", "LA", "MM", "NP", "PH", "TL", "VN", "GU"}})
    country_to_region.update({code: "Central Asia" for code in {"TJ", "UZ", "KG", "KZ"}})
    country_to_region.update({code: "Europe & Central Asia" for code in {"AL", "AM", "AZ", "GE", "MD", "MK", "RS", "TR", "UA"}})
    country_to_region.update({code: "Sub-Saharan Africa" for code in {"AO", "BJ", "BF", "BU", "CM", "CF", "CI", "CD", "ET", "GA", "GH", "GN", "GY", "KE", "KM", "LS", "LB", "MD", "MW", "ML", "MZ", "NG", "NM", "RW", "SN", "SL", "SZ", "TD", "TG", "ZA", "TZ", "UG", "ZM", "ZW"}})
    country_to_region.update({code: "North Africa & Middle East" for code in {"EG", "JO", "YE", "MA", "LB", "MB"}})
    country_to_region.update({code: "South Asia" for code in {"AF", "BD", "IN", "PK", "NP", "IA"}})
    country_to_region.update({code: "Latin America & Caribbean" for code in {"BO", "CO", "DR", "HT", "HN", "MX", "PE", "NI"}})

    # Extract country codes from DHSID
    df['Country_Code'] = df['DHSID'].str.extract(r'([A-Za-z]+)')[0]
    
    # Correct any mislabeled country codes
    df.loc[df["Country_Code"] == "DHS", "Country_Code"] = "BD"

    # Map regions and countries
    df['target_region'] = df['Country_Code'].map(country_to_region)

    return df

def split_data_country_wise(df):
    X = df.drop(columns=labels)
    y = df[labels]
    # Unique country codes
    countries = X['Country_Code'].unique()

    # Initialize empty lists to store the split data
    X_train_list, X_dev_list, X_test_list = [], [], []
    y_train_list, y_dev_list, y_test_list = [], [], []

    # Split the data for each country
    for country in countries:
        X_country = X[df['Country_Code'] == country]
        y_country = y[df['Country_Code'] == country]
        
        X_train_country, X_temp, y_train_country, y_temp = train_test_split(X_country, y_country, test_size=0.2, random_state=1)
        X_dev_country, X_test_country, y_dev_country, y_test_country = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

        X_train_list.append(X_train_country)
        X_dev_list.append(X_dev_country)
        X_test_list.append(X_test_country)

        y_train_list.append(y_train_country)
        y_dev_list.append(y_dev_country)
        y_test_list.append(y_test_country)

    # Concatenate the splits
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_dev = pd.concat(X_dev_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)

    y_train = pd.concat(y_train_list, ignore_index=True)
    y_dev = pd.concat(y_dev_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)

    return X_train,X_dev,X_test,y_train,y_dev,y_test

def one_hot_encode_fit(X_train):
    # Specify the columns to be encoded
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    # Create transformer
    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
        remainder='passthrough')
    X_train_encoded = transformer.fit_transform(X_train)
    return X_train_encoded, transformer

def one_hot_encode_transform(X, transformer):
    X_encoded = transformer.transform(X)
    return X_encoded

def obj_columns(df):
    return(df.columns[df.dtypes == 'object'].tolist())

def mcrmse(y_true, y_pred):
    return np.mean(np.sqrt(np.mean(np.square(y_true - y_pred), axis=0)))

def make_predictions(input_df):
    # Ensure the input dataframe has been preprocessed
    # Load the trained models, scores, and weights
    chains = joblib.load('chains.joblib')
    weights = joblib.load('weights.joblib')
    
    # Make predictions using each model in the ensemble
    predictions = [chain.predict(input_df) for chain in chains]
    
    # Compute the ensemble predictions
    final_predictions = np.zeros_like(predictions[0])
    for weight, prediction in zip(weights, predictions):
        final_predictions += weight * prediction
        

    # Convert predictions to a dataframe with appropriate column names
    final_predictions_df = pd.DataFrame(final_predictions, columns=labels)
        
    return final_predictions_df


def evaluate_order(model, order, X_train, y_train, X_dev, y_dev):
    # Complete the ordering
    full_order = order + [i for i in range(y_train.shape[1]) if i not in order]

    chain = RegressorChain(model, order=full_order)
    chain.fit(X_train, y_train)
    y_pred = chain.predict(X_dev)
    score = mcrmse(y_dev, y_pred)

    return score

def full_greedy_search(model, X_train, y_train, X_dev, y_dev):
    all_targets = list(range(y_train.shape[1]))
    current_order = []
    history = []
    
    while all_targets:
        best_score = float('inf')
        best_target = None

        for t in all_targets:
            temp_order = current_order + [t]
            score = evaluate_order(model, temp_order, X_train, y_train, X_dev, y_dev)
            
            # Capture history for insights on progression
            history.append((temp_order, score))
            
            if score < best_score:
                best_score = score
                best_target = t

        print(f"Added target {best_target} to the order. Current best score: {best_score}")
        
        current_order.append(best_target)
        all_targets.remove(best_target)

    return current_order, history
