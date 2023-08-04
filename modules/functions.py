import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


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

def print_obj_columns(df):
    print(df.columns[df.dtypes == 'object'].tolist())

def split_data_country_wise(df):
    if 'Country_Code' not in df.columns.to_list():
        raise Exception("dataframe doesnt have a column named Country_Code, use country_region_mapping(df) to get it")

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

# Define the function to calculate MCRMSE
def mcrmse(y_true, y_pred):
    return np.mean(np.sqrt(np.mean(np.square(y_true - y_pred), axis=0)))