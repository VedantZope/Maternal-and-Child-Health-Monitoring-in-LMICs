import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.notebook import tqdm
import joblib
import optuna
from sklearn.model_selection import KFold
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor

def evaluate_order_kfold(models, order, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        fold_scores = []
        for model in models:
            chain = RegressorChain(model, order=order, random_state=1)
            chain.fit(X_train_fold, y_train_fold)
            y_pred_val = chain.predict(X_val_fold)
            score = mcrmse(y_val_fold, y_pred_val)
            fold_scores.append(score)

        scores.append(np.mean(fold_scores))

    return np.mean(scores)

def greedy_search_order(models, X, y, n_splits=5):
    all_targets = list(range(y.shape[1]))
    current_order = []
    while len(all_targets) > 0:
        best_score = float('inf')
        best_target = None

        for target in all_targets:
            temp_order = current_order + [target]
            temp_score = evaluate_order_kfold(models, temp_order, X, y, n_splits=n_splits)

            if temp_score < best_score:
                best_score = temp_score
                best_target = target

        current_order.append(best_target)
        all_targets.remove(best_target)
        
    return current_order


def make_predictions(input_df):
    """
    This function will preprocess the input dataframe, and use the ensemble of trained models to make predictions.
    Args:
    - input_df: The input dataframe. It should have the same features as your training data.
    
    Returns:
    - final_predictions_df: Predictions for the input dataframe in the form of a dataframe with appropriate column names.
    """
    # Ensure the input dataframe has been preprocessed
    # Example: if you used scaling before, ensure `input_df` has also been scaled
    
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
    final_predictions_df = pd.DataFrame(final_predictions, columns=labels_list())
        
    return final_predictions_df

# def apply_robust_scaling(df):
#     df = df.drop(columns=obj_columns(df))
#     scaler = RobustScaler()
#     df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
#     return df_scaled

def drop_high_nan(df, threshold=0.85):
    """
    Drops columns from the dataframe that have a proportion of NaN values greater than the specified threshold.
    
    Parameters:
    - df: Input DataFrame.
    - threshold: Proportion threshold for NaN values (default is 0.85).
    
    Returns:
    - DataFrame with columns having NaN proportion greater than the threshold dropped.
    """
    nan_proportion = df.isnull().mean()
    columns_to_drop = nan_proportion[nan_proportion > threshold].index.tolist()
    return df.drop(columns=columns_to_drop)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, df):
        numerical_data = df.select_dtypes(include=['float64', 'int64']).values
        self.scaler.fit(numerical_data)

    def transform(self, df):
        df_transformed = df.copy()
        numerical_features = df_transformed.select_dtypes(include=['float64', 'int64']).columns
        df_transformed[numerical_features] = self.scaler.transform(df_transformed[numerical_features].values)
        return df_transformed

    def inverse_transform(self, df):
        df_original = df.copy()
        numerical_features = df_original.select_dtypes(include=['float64', 'int64']).columns
        df_original[numerical_features] = self.scaler.inverse_transform(df_original[numerical_features].values)
        return df_original

class TargetPreprocessor:
    def __init__(self, column_names=None):
        self.scaler = StandardScaler()
        self.column_names = column_names

    def fit(self, y):
        self.scaler.fit(y)

    def transform(self, y):
        y_array = self.scaler.transform(y)
        if isinstance(y, pd.DataFrame):
            return pd.DataFrame(y_array, columns=y.columns, index=y.index)
        else:
            return y_array

    def inverse_transform(self, y_scaled):
        y_array = self.scaler.inverse_transform(y_scaled)
        if self.column_names:
            return pd.DataFrame(y_array, columns=self.column_names)
        elif isinstance(y_scaled, pd.DataFrame):
            return pd.DataFrame(y_array, columns=y_scaled.columns, index=y_scaled.index)
        else:
            return y_array


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import pandas as pd

def fit_transform_iterative_imputer(df, n_iter, random_state=None):
    """
    Fit the iterative imputer on the provided dataframe and then transform it.
    Returns the imputed dataframe and the fitted imputer.
    """
    # Configuration for XGBRegressor
    xgb_iconfig = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'gamma': 0.1,
        'random_state': random_state
    }
    
    # Initialize the iterative imputer with XGBRegressor
    imputer = IterativeImputer(estimator=XGBRegressor(**xgb_iconfig), max_iter=n_iter, random_state=random_state)
    
    # Fit and transform the dataframe
    imputed_data = imputer.fit_transform(df)
    
    # Convert the imputed data back to a dataframe
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    
    return imputed_df, imputer


def transform_iterative_imputer(df, fitted_imputer):
    """
    Transforms the given data using a provided fitted IterativeImputer.
    
    Parameters:
    - df: DataFrame to be transformed.
    - fitted_imputer: Fitted IterativeImputer.
    
    Returns:
    - DataFrame with imputed values.
    """
    
    imputed_df = df.copy()
    
    # Separate numerical and non-numerical columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    
    # Transform the data using the provided fitted imputer
    imputed_data = fitted_imputer.transform(imputed_df[numerical_cols])
    
    # Update the imputed values in the DataFrame
    imputed_df[numerical_cols] = imputed_data
    
    return imputed_df


# def apply_knn_iterative_imputation(df, knn_n_neighbors, max_iter, random_state=None):
#     warnings.warn("it is a good idea to apply scaling before imputation,ignore if already applied, if not apply_robust_scaling() and apply_standard_scaling() functions can be used")
#     """
#     Apply KNN imputation for columns with less than 30% missing data and Iterative imputation for columns with more.
    
#     Parameters:
#     - df: The input dataframe with missing values.
#     - knn_n_neighbors: Number of neighboring samples to use for KNN imputation.
#     - max_iter: Maximum number of imputation iterations for IterativeImputer.
#     - random_state: Seed used by the random number generator for IterativeImputer.
    
#     Returns:
#     - df: DataFrame with imputed values.
#     """
    
#     # Identify columns based on missing data threshold
#     missing_data = df.isna().mean()
#     cols_lt_30 = missing_data[missing_data < 0.3].index.tolist()
#     cols_gt_30 = missing_data[missing_data >= 0.3].index.tolist()

#     # KNN imputation for columns with < 30% missing data
#     if len(cols_lt_30) > 0:
#         print("KNN imputer for columns with < 30% missing data started")
#         knn_imputer = KNNImputer(n_neighbors=knn_n_neighbors)
#         df[cols_lt_30] = knn_imputer.fit_transform(df[cols_lt_30])
    
#     # Iterative imputation for columns with >= 30% missing data
#     if len(cols_gt_30) > 0:
#         print("Iterative imputer for columns with >= 30% missing data started")
#         iter_imputer = IterativeImputer(estimator=DecisionTreeRegressor(), max_iter=max_iter, random_state=random_state)
#         df[cols_gt_30] = iter_imputer.fit_transform(df[cols_gt_30])

#     return df

def load_csv_data(filepath):
    df = pd.read_csv(filepath)
    return df

def load_parquet_data(filepath):
    df = pd.read_parquet(filepath)
    return df

def labels_list():
    labels = ['Mean_BMI', 'Median_BMI',
           'Unmet_Need_Rate', 'Under5_Mortality_Rate',
         'Skilled_Birth_Attendant_Rate', 'Stunted_Rate']
    return labels

def heatmap(df):    
    plt.figure(figsize=(40, 30)) # Increase the size of the figure
    sns.heatmap(df.isnull(), cbar=False, cmap='binary')
    plt.show()

def one_hot_encode(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df

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
    df['target_country'] = df['Country_Code'].factorize()[0]

    return df

#df1 is main dataframe and df2 is secondary dataframe, merged on basis of DHSID
def merge_dataframes(df, df2):
    return df.merge(df2, on='DHSID', how='inner')

def obj_columns(df):
    return(df.columns[df.dtypes == 'object'].tolist())

def split_data_country_wise(df):
    if 'Country_Code' not in df.columns.to_list():
        raise Exception("dataframe doesnt have a column named Country_Code, use country_region_mapping(df) to get it")
    labels = labels_list()
    X = df.drop(columns=labels)
    y = df[labels]
    # Unique country codes
    countries = df['Country_Code'].unique()

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

def add_temporal_features(df):
    """
    Adds temporal features to the dataframe.
    
    For countries with data across multiple years:
    1. Computes the year-on-year difference for each feature.
    2. Generates aggregated temporal features like mean, median, 
       and standard deviation for each feature for each country over the years.
    
    Parameters:
    - df: DataFrame with original data.
    
    Returns:
    - DataFrame with additional temporal features.
    """
    
    # Excluding labels and other non-feature columns
    exclude_cols = obj_columns(df) + labels_list()
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Sort the dataframe by 'Country_Code' and 'DHSYEAR' for proper computation
    df = df.sort_values(by=['Country_Code', 'DHSYEAR'])
    
    new_cols = []
    
    # 1. Compute year-on-year difference for each feature
    for col in feature_cols:
        new_cols.append(pd.Series(df.groupby('Country_Code')[col].diff(), name=f"{col}_yearly_diff"))
    
    # 2. Generate aggregated temporal features for each feature for each country
    for col in feature_cols:
        new_cols.append(pd.Series(df.groupby('Country_Code')[col].transform('mean'), name=f"{col}_mean"))
        new_cols.append(pd.Series(df.groupby('Country_Code')[col].transform('median'), name=f"{col}_median"))
        new_cols.append(pd.Series(df.groupby('Country_Code')[col].transform('std'), name=f"{col}_std"))
    
    # Concatenate original DataFrame with new columns
    df = pd.concat([df] + new_cols, axis=1)
    
    return df

def nan_percentage(df):
    """
    Returns a DataFrame with columns and their corresponding % of NaN values in descending order.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with columns: 'Column Name' and 'NaN %'
    """
    # Calculate the percentage of NaNs for each column
    nan_percent = df.isnull().mean() * 100

    # Create a DataFrame with results
    result_df = pd.DataFrame({
        'Column Name': nan_percent.index,
        'NaN %': nan_percent.values
    })

    # Sort the DataFrame in descending order
    result_df = result_df.sort_values(by='NaN %', ascending=False)

    return result_df

# Define the function to calculate MCRMSE
def mcrmse(y_true, y_pred):
    return np.mean(np.sqrt(np.mean(np.square(y_true - y_pred), axis=0)))