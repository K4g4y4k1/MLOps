import pandas as pd
import dataset_creation as dc
import outlier_removal as outrem
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

training_dataset = dc.dataset_creation()
evaluation_dataset = dc.dataset_creation()

training_dataset_cleaned = outrem.remove_outliers_iqr(training_dataset, 'duration')
evaluation_dataset_cleaned = outrem.remove_outliers_iqr(evaluation_dataset, 'duration')

categorical = ['PULocationID','DOLocationID']
numerical = ['trip_distance']
target = 'duration'


df_train = df_train_clean[categorical + numerical]