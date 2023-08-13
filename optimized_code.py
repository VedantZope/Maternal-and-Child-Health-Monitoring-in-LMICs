# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from tqdm import tqdm
import joblib
import numpy as np
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import os
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
from sklearn.model_selection import KFold


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
    imputer = IterativeImputer(estimator=XGBRegressor(n_jobs=40,**xgb_iconfig), max_iter=n_iter, random_state=random_state)
    
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

df = load_csv_data("top120_geetl.csv")
wbdata = load_csv_data("features_wbdata.csv")
whodata = load_parquet_data("WHO_data_feature.parquet")
df.drop(columns=['key3.1'],inplace = True)


merged_df_1 = merge_dataframes(df,wbdata)
merged_df = merge_dataframes(merged_df_1,whodata)


merged_df.dropna(subset=labels_list(),inplace = True)
merged_df.reset_index(drop = True, inplace = True)



#adding temporal features to the merged df where multiple year data is present
merged_df = add_temporal_features(merged_df)

labels = labels_list()
# dropping columns with nan count more than 85% and then country region mapping 
X_train,X_dev,X_test,y_train,y_dev,y_test = split_data_country_wise(country_region_mapping(drop_high_nan((merged_df))))

X_train.drop(columns =['DHSID','WB_Country_Code','WHO_Country_code'],inplace = True)
X_train = one_hot_encode(X_train,['URBAN_RURA','Country_Code', 'target_region'])

X_dev.drop(columns =['DHSID','WB_Country_Code','WHO_Country_code'],inplace = True)
X_dev = one_hot_encode(X_dev,['URBAN_RURA','Country_Code', 'target_region'])

X_test.drop(columns =['DHSID','WB_Country_Code','WHO_Country_code'],inplace = True)
X_test = one_hot_encode(X_test,['URBAN_RURA','Country_Code', 'target_region'])




# Create an instance of the preprocessor for features
features_preprocessor = DataPreprocessor()

# Fit the preprocessor on the training data
features_preprocessor.fit(X_train)

# Transform the training, validation, and test data
X_train_scaled = features_preprocessor.transform(X_train)
X_dev_scaled = features_preprocessor.transform(X_dev)
X_test_scaled = features_preprocessor.transform(X_test)

# Create an instance of the preprocessor for targets
target_preprocessor = TargetPreprocessor()

# Fit the preprocessor on the training target data
target_preprocessor.fit(y_train)

# Transform the training and validation target data
y_train_scaled = target_preprocessor.transform(y_train)
y_dev_scaled = target_preprocessor.transform(y_dev)
y_test_scaled = target_preprocessor.transform(y_test)


# Fit and transform the training data
print("Fitting and Transforming the Training data")
X_train_scaled_imputed, fitted_imputer = fit_transform_iterative_imputer(X_train_scaled, n_iter=10)

print("Transforming the Test and Validation data")
# Transform the validation and test data using the fitted imputer
X_dev_scaled_imputed = transform_iterative_imputer(X_dev_scaled, fitted_imputer)
X_test_scaled_imputed = transform_iterative_imputer(X_test_scaled, fitted_imputer)

X_train_scaled_imputed.to_csv("X_train_scaled_imputed.csv",index = False)
X_dev_scaled_imputed.to_csv("X_dev_scaled_imputed.csv",index = False)
X_test_scaled_imputed.to_csv("X_test_scaled_imputed.csv",index = False)

y_train_scaled.to_csv("y_train_scaled.csv",index = False)
y_dev_scaled.to_csv("y_dev_scaled.csv",index = False)
y_test_scaled.to_csv("y_test_scaled.csv",index = False)



optuna.logging.set_verbosity(optuna.logging.INFO)

# Define order for the RegressorChain

def objective(trial):
    # 1. Model selection
    model_type = trial.suggest_categorical("model_type", ["xgb", "lgb", "cat"])
    
    # Define order for the RegressorChain
    order = [0, 1, 5, 4, 3, 2]
    
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, valid_idx in kf.split(X_train_scaled_imputed):
        X_train_fold = X_train_scaled_imputed[train_idx]
        y_train_fold = y_train_scaled[train_idx]
        X_valid_fold = X_train_scaled_imputed[valid_idx]
        y_valid_fold = y_train_scaled[valid_idx]
        
        # 2. Hyperparameter definitions based on the model
        if model_type == "xgb":
            model = XGBRegressor(n_jobs=40,
                random_state=1,
                learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("xgb_n_estimators", 50, 300),
                max_depth=trial.suggest_int("xgb_max_depth", 2, 10),
                min_child_weight=trial.suggest_int("xgb_min_child_weight", 1, 10),
                subsample=trial.suggest_float("xgb_subsample", 0.5, 1),
                colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.5, 1),
                reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True)
            )
            model.fit(X_train_fold, y_train_fold, early_stopping_rounds=10, eval_metric="rmse", eval_set=[(X_valid_fold, y_valid_fold)])
            
        elif model_type == "lgb":
            model = LGBMRegressor(n_jobs=40,
                random_state=1,
                learning_rate=trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("lgb_n_estimators", 50, 300),
                max_depth=trial.suggest_int("lgb_max_depth", 2, 10),
                num_leaves=trial.suggest_int("lgb_num_leaves", 2, 2**trial.suggest_int("lgb_max_depth", 2, 10)),
                min_child_samples=trial.suggest_int("lgb_min_child_samples", 5, 100),
                subsample=trial.suggest_float("lgb_subsample", 0.5, 1),
                colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.5, 1),
                reg_alpha=trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True)
            )
            model.fit(X_train_fold, y_train_fold, early_stopping_rounds=10, eval_metric="rmse", eval_set=[(X_valid_fold, y_valid_fold)])
            
        else:
            model = CatBoostRegressor(
                random_seed=1,
                learning_rate=trial.suggest_float("cat_learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("cat_n_estimators", 50, 300),
                depth=trial.suggest_int("cat_depth", 2, 10),
                l2_leaf_reg=trial.suggest_float("cat_l2_leaf_reg", 1e-3, 10.0, log=True),
                border_count=trial.suggest_int("cat_border_count", 5, 200),
                subsample=trial.suggest_float("cat_subsample", 0.5, 1)
            )
            model.fit(X_train_fold, y_train_fold, early_stopping_rounds=10, eval_metric="RMSE", eval_set=[(X_valid_fold, y_valid_fold)], verbose=0)
        
        # Model evaluation for this fold
        chain = RegressorChain(model, order=order)
        chain.fit(X_train_fold, y_train_fold)

        y_pred_fold = chain.predict(X_valid_fold)
        y_pred_original = target_preprocessor.inverse_transform(y_pred_fold)
        scores.append(mcrmse(y_valid_fold, y_pred_original))
    
    # Return the average score over all folds
    return np.mean(scores)

# Hyperparameter optimization
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=120, n_jobs=-1)  # 40 trials for each model

# Results
best_trials = {}
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        model_type = trial.params['model_type']
        if model_type not in best_trials or trial.value < best_trials[model_type].value:
            best_trials[model_type] = trial

best_params_dict = {}
for model_type, trial in best_trials.items():
    print(f"Best parameters for {model_type}: {trial.params}")
    print(f"Score: {trial.value}")
    print('-' * 50)
    best_params_dict[model_type] = trial.params

np.save('best_hyperparams.npy', best_params_dict)

#loading the hyperparams from optuna
hyperparams = np.load('best_hyperparams.npy', allow_pickle=True).item()

xgb_params = hyperparams['xgb']
lgb_params = hyperparams['lgb']
cat_params = hyperparams['cat']

models = [
    XGBRegressor(n_jobs=40,**xgb_params),
    LGBMRegressor(n_jobs=40,**lgb_params),
    CatBoostRegressor(**cat_params)
]

#finding the best regression chain order using greedyt search and k fold validation
best_order = greedy_search_order(models, X_train_scaled_imputed, y_train_scaled)
print(f"Best order found: {best_order}")



# Initialize the models


chains = []
scores = []

# Train a RegressorChain with each model and calculate validation scores
for i, model in tqdm(enumerate(models), total=len(models), desc="Training models"):
    
    print(f"\n\nTraining model {i+1}/{len(models)}: {model.__class__.__name__}\n")
    
    chain = RegressorChain(model, order=best_order, random_state=1)
    chain.fit(X_train_scaled_imputed, y_train_scaled)
    chains.append(chain)
    
    y_pred_dev_scaled = chain.predict(X_dev_scaled_imputed)
    
    # Inverse transform both predictions and targets to original scale
    y_pred_dev_original = target_preprocessor.inverse_transform(y_pred_dev_scaled)
    y_dev_original = target_preprocessor.inverse_transform(y_dev_scaled)
    
    score = mcrmse(y_dev_original, y_pred_dev_original)
    scores.append(score)
    
    print(f"\nValidation MCRMSE for model {i+1}: {score:.4f}\n")
    print("="*50)

# Calculate BMA weights based on the inverse of the validation scores
weights = [1/score for score in scores]
weights = [weight/sum(weights) for weight in weights]

# Save models, scores, and weights for later use
joblib.dump(chains, 'chains.joblib')
joblib.dump(scores, 'scores.joblib')
joblib.dump(weights, 'weights.joblib')

print("\nComputing the ensemble predictions on the dev set...\n")
predictions_dev_scaled = [chain.predict(X_dev_scaled_imputed) for chain in chains]
final_predictions_dev_scaled = np.zeros_like(predictions_dev_scaled[0])
for weight, prediction in zip(weights, predictions_dev_scaled):
    final_predictions_dev_scaled += weight * prediction

# Inverse transform for ensemble predictions
final_predictions_dev_original = target_preprocessor.inverse_transform(final_predictions_dev_scaled)
final_score_dev = mcrmse(y_dev_original, final_predictions_dev_original)
print(f"Final ensemble validation MCRMSE: {final_score_dev:.4f}")

print("\nEvaluating on the test set...\n")
predictions_test_scaled = [chain.predict(X_test_scaled_imputed) for chain in chains]
final_predictions_test_scaled = np.zeros_like(predictions_test_scaled[0])
for weight, prediction in zip(weights, predictions_test_scaled):
    final_predictions_test_scaled += weight * prediction

# Inverse transform for test ensemble predictions
final_predictions_test_original = target_preprocessor.inverse_transform(final_predictions_test_scaled)
y_test_original = target_preprocessor.inverse_transform(y_test_scaled)
final_score_test = mcrmse(y_test_original, final_predictions_test_original)
print(f"Final ensemble test MCRMSE: {final_score_test:.4f}")

df_sample = load_parquet_data("sample with features.parquet")
wbdata_sample = load_csv_data("sample_wbdata.csv")
whodata_Sample = load_csv_data("WHO_data_sample.csv")

merged_df_sample1 = merge_dataframes(df_sample,wbdata_sample)
merged_df_sample = merge_dataframes(merged_df_sample1,whodata_Sample)

#adding temporal features to the sample
merged_df_sample = add_temporal_features(merged_df_sample)
#applying country region mapping
X_sample = country_region_mapping(merged_df_sample)
#dropping unncecesarry columns
X_sample.drop(columns =['DHSID','WB_Country_Code','WHO_Country_code'],inplace = True)
X_sample = one_hot_encode(X_sample,['URBAN_RURA','Country_Code', 'target_region'])
X_sample['DHSYEAR'] = X_sample['key3']
X_sample = X_sample[X_train.columns]

# Convert float32 columns to float64
float32_cols = X_sample.select_dtypes(include=['float32']).columns
X_sample[float32_cols] = X_sample[float32_cols].astype('float64')

# Convert non-NaN values to int64
X_sample.loc[X_sample['DHSYEAR'].notna(), 'DHSYEAR'] = X_sample.loc[X_sample['DHSYEAR'].notna(), 'DHSYEAR'].astype('int64')


X_sample_scaled = features_preprocessor.transform(X_sample)

X_sample_scaled_imputed = transform_iterative_imputer(X_sample_scaled, fitted_imputer)

prediction_scaled = make_predictions(X_sample_scaled_imputed)

df_submission = target_preprocessor.inverse_transform(prediction_scaled)

df_sample.reset_index(drop=True,inplace=True)
df_submission['DHSID'] = df_sample['DHSID']

df_submission = df_submission[["DHSID"]+labels]
df_submission.to_csv("save.csv",index = False)

df_submission



