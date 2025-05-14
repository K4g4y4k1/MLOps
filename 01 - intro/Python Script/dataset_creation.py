import pandas as pd

def dataset_creation():
    color = input('Color')
    year = input('Year')
    month = input('Month')
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/'+color+'_tripdata_'+year+'-'+month+'.parquet'
    df = pd.read_parquet(url)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df['duration'] = None
    df['duration']= df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds()/60
    categorical = ['PULocationID','DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df