import pandas as pd
import dataset_creation as dc
import outlier_removal as outrem

df = dc.dataset_creation()
df_cleaned = outrem(df,'duration')

categorical = ['PULocationID','DOLocationID']
numerical = ['trip_distance']
target = 'duration'

df_train = df_train_clean[categorical + numerical]