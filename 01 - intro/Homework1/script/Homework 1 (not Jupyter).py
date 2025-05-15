import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


### Data analysis
### Q1. Downloading the data
# We'll use the same NYC taxi dataset, but instead of "Green Taxi Trip Records", we'll use "Yellow Taxi Trip Records".
# Download the data for January and February 2023.
# Read the data for January. How many columns are there?

df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')

print("Number of columns ", len(df.columns))


### Q2. Computing duration
# Now let's compute the duration variable. It should contain the duration of a ride in minutes.
# What's the standard deviation of the trips duration in January?

df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration'] = df['duration'].dt.total_seconds()/60

std_dev = np.std(df['duration'], ddof=1)
print("the standard deviation of the trips duration in January is " ,std_dev)


### Q3. Dropping outliers
# Next, we need to check the distribution of the duration variable. There are some outliers.
# Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
# What fraction of the records left after you dropped the outliers?

df_cleaned = df[df['duration'] >= 1][df['duration'] <= 60]
pct_left = (len(df_cleaned['duration'])*100)/len(df['duration'])
print("Percentage left after cleaning: ",pct_left)

### Q4. One-hot encoding
# Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model.
# Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will label encode them)
# Fit a dictionary vectorizer
# Get a feature matrix from it
# What's the dimensionality of this matrix (number of columns)?

categorical = ['PULocationID', 'DOLocationID']
df_categorical = df_cleaned[categorical].astype(str)
print(df_categorical.info())
dv = DictVectorizer()
train_dict = df_categorical.to_dict(orient='records')


### Q5. Training a model
# Now let's use the feature matrix from the previous step to train a model.
# Train a plain linear regression model with default parameters, where duration is the response variable
# Calculate the RMSE of the model on the training data
# What's the RMSE on train?

target = 'duration'
X_train = dv.fit_transform(train_dict)
y_train = df_cleaned[target].values
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f"Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

### Q6. Evaluating the model
# Now let's apply this model to the validation dataset (February 2023).
# What's the RMSE on validation?

df_val = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')
df_val['duration'] = df_val['tpep_dropoff_datetime'] - df_val['tpep_pickup_datetime']
df_val['duration'] = df_val['duration'].dt.total_seconds()/60
df_val_cleaned = df_val[df_val['duration'] >= 1][df_val['duration'] <= 60]

df_val_categorical = df_val_cleaned[categorical].astype(str)

val_dict = df_val_categorical.to_dict(orient='records')

X_val = dv.transform(val_dict)
y_val = df_val_cleaned[target].values

y_pred_val = model.predict(X_val)

mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)
print(f"Validation Mean Squared Error: {rmse_val:.2f}")
print(f"Validation R² Score: {r2_val:.2f}")
