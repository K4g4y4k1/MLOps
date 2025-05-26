import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("nyc-taxi-experiment")  


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    params = {
        'bootstrap':True,
        'ccp_alpha':0.0,
        'criterion': 'squared_error',
        'max_depth':10,
        'max_features':1.0,
        'max_leaf_nodes':None,
        'max_samples':None,
        'min_impurity_decrease':0.0,
        'min_samples_leaf':1,
        'min_samples_split':2,
        'min_weight_fraction_leaf':0.0,
        'monotonic_cst':None,
        'n_estimators':100,
        'n_jobs':None,
        'oob_score':False,
        'random_state':0,
        'verbose':0,
        'warm_start':False
    }

    # Enable autologging
    mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    with mlflow.start_run():
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"RMSE: {rmse}")

        # Log the model
        mlflow.sklearn.log_model(rf, "model")
        # Log the RMSE metric
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()