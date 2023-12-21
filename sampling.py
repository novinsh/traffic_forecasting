#%%
# import numpy as np
# import os
# import geopandas as gpd
# from shapely import wkt
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random
random.seed(6531)

count_pattern = r"data/*_count.csv"

def load_count_files(list_files, ratio=1):
	count_traffic_list = []
	for file_name in list_files:
		print(file_name)
		df = pd.read_csv(file_name, engine='pyarrow', sep=",")
		df_grouped = df.groupby("vehicle_id")
		df_sampled = df_grouped.sample(frac=ratio, random_state=1).reset_index()

		agg_count = df_sampled.groupby(["hour","edge_id"]).apply(len).reset_index()
		agg_count = agg_count.rename(columns={0:"count"})
		agg_count["date"] = file_name.split("/")[-1].split("_")[0]
		agg_count["edge_id"] = agg_count["edge_id"].astype('string')
		merge_hour_and_date = lambda df: pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

		df_traffic = agg_count.copy(deep=True)
		df_traffic['timestamp'] = merge_hour_and_date(df_traffic)
		df_traffic.drop(columns=['hour', 'date'], inplace=True)
		df_traffic.set_index('timestamp', inplace=True)

		count_traffic_list.append(df_traffic)
	return count_traffic_list

#%%
def sample_data(list_files, ratio=1):
 
    alldata_list = load_count_files(list_files, ratio=ratio)
    df_traffic = pd.concat(alldata_list)
    merge_hour_and_date = lambda df: pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

    datelist = []
    for file_name in list_files:
        datelist.append(file_name.split("/")[-1].split("_")[0])
    
    hourlist = list(range(24))
    edgelist = list(set(df_traffic["edge_id"]))

    df1 = pd.DataFrame({'date':datelist})
    df2 = pd.DataFrame({'hour':hourlist})  
    df3 = pd.DataFrame({'edge_id':edgelist})  

    temp_df1 = df1.merge(df2, how='cross')
    temp_df2 = temp_df1.merge(df3, how="cross")

    temp_df2["timestamp"] = merge_hour_and_date(temp_df2)
    temp_df2 = temp_df2[["timestamp","edge_id"]]

    df_traffic = df_traffic.reset_index()
    df_traffic = temp_df2.merge(df_traffic, on=["timestamp","edge_id"], how="left")
    df_traffic = df_traffic.fillna(0)
    return df_traffic

#%%
list_files = glob.glob(count_pattern)
df_traffic_ratio_50 = sample_data(list_files=list_files, ratio=.5)

#%%
df_traffic_ratio_50