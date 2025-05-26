import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        # Debug: display all available parameters
        print(f"🔍 Available parameters in run: {list(params.keys())}")
        
        new_params = {}
        for param in RF_PARAMS:
            if param in params:
                try:
                    # All parameters come as strings from MLflow, convert appropriately
                    if param == 'random_state':
                        new_params[param] = int(params[param])
                    elif param in ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']:
                        new_params[param] = int(params[param])
                    else:
                        new_params[param] = params[param]
                except (ValueError, TypeError) as e:
                    print(f"   ⚠️ Error converting parameter '{param}': {e}")
                    continue
                print(f"✅ Parameter '{param}': {new_params[param]}")
            else:
                print(f"⚠️ Parameter '{param}' missing in run.")
        
        print(f"🛠️ Parameters used for model: {new_params}")
        
        if new_params:  # Ensure we have at least some parameters
            rf = RandomForestRegressor(**new_params)
        else:
            print("⚠️ No valid parameters found, using default parameters")
            rf = RandomForestRegressor(random_state=42)
        
        rf.fit(X_train, y_train)

        # Évaluer et logger les métriques
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)
        
        print(f"📊 Validation RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    try:
        # Retrieve the top_n model runs and log the models
        experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
        if not experiment:
            raise ValueError(f"Experiment '{HPO_EXPERIMENT_NAME}' does not exist!")
        
        print(f"🔍 Searching for top {top_n} runs in experiment: {HPO_EXPERIMENT_NAME}")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n * 3,  # Get more runs to account for failed ones
            order_by=["metrics.val_rmse ASC"],  # Use val_rmse which exists
            filter_string="status = 'FINISHED'"  # Only get successful runs
        )
        
        # Filter out failed runs and take only top_n successful ones
        successful_runs = [run for run in runs if run.info.status == 'FINISHED' and run.data.params]
        runs = successful_runs[:top_n]
        
        if not runs:
            print("❌ No successful runs found in HPO experiment!")
            return
        
        print(f"✅ {len(runs)} successful runs found for retraining")
        
        # Debug: examine first run to see available parameters
        if runs:
            first_run = runs[0]
            print(f"🔍 First run - Parameters: {list(first_run.data.params.keys())}")
            print(f"🔍 First run - Metrics: {list(first_run.data.metrics.keys())}")
        
        for i, run in enumerate(runs):
            print(f"\n🏃 Processing run {i+1}/{len(runs)}: {run.info.run_id} (Status: {run.info.status})")
            if run.info.status == 'FINISHED' and run.data.params:
                train_and_log_model(data_path=data_path, params=run.data.params)
            else:
                print(f"   ⚠️ Skipping run - Status: {run.info.status}, Has params: {bool(run.data.params)}")

        # Select the model with the lowest test RMSE
        print(f"\n🏆 Selecting best model in experiment: {EXPERIMENT_NAME}")
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        best_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_rmse ASC"],
            max_results=1
        )
        
        if not best_runs:
            print("❌ No runs found in best models experiment!")
            return
            
        best_run = best_runs[0]
        print(f"🥇 Best run: {best_run.info.run_id} with test_rmse: {best_run.data.metrics.get('test_rmse', 'N/A')}")

        # Register the best model
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model_name = "random-forest-best-model"
        
        print(f"📝 Registering model: {model_name}")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"✅ Model registered successfully: {registered_model.name} v{registered_model.version}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == '__main__':
    run_register_model()