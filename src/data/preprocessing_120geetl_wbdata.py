import sys
sys.path.append("/Users/vedantzope/Desktop/Maternal-and-Child-Health-Monitoring-in-LMICs")


import pandas as pd
import os
from modules.functions import *

df = load_data("data/interim/top120_geetl.csv")
labels =labels_list()

wbdata = load_data("data/interim/120geetl_wbdata.csv")

merged_df = merge_dataframes(df,wbdata)

