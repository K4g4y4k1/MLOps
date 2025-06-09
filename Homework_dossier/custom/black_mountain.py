import xgboost as xgb
import mlflow
import pickle

mlflow.set_tracking_uri("http://localhost:5001")  # si tu utilises un MLflow local
mlflow.set_experiment("nyc_taxi_experiment")

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@custom
def train_model(X_train, y_train, dv, **kwargs):
    with mlflow.start_run():
        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'objective': 'reg:squarederror',
            'seed': 42
        }

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100
        )

        # Log artefact
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")