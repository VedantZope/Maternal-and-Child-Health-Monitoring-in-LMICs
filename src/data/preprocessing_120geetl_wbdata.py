import sys
sys.path.append("/Users/vedantzope/Desktop/Maternal-and-Child-Health-Monitoring-in-LMICs")

from modules.functions import *
import pandas as pd
import os
df = load_csv_data("data/interim/top120_geetl.csv")
wbdata = load_csv_data("data/external/wb_data_features/features_wbdata.csv")

merged_df = merge_dataframes(df,wbdata)
labels =labels_list()

merged_df.dropna(subset=labels,inplace = True)
merged_df.reset_index(drop = True, inplace = True)
print("df_knn_iter")
df_knn_iter = apply_knn_iterative_imputation(apply_standard_scaling(merged_df),15,25)
print("df_iter")
df_iter = apply_iterative_imputation(apply_standard_scaling(merged_df),15)

print("saving data")
df_knn_iter.to_csv("data/interim/120_gee_wbdata+knn_iter_imputaion.csv",index=False)
df_iter.to_csv("data/interim/120_gee_wbdata+iter_imputaion.csv",index=False)