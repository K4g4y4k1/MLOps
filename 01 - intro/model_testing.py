from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def model_evaluator(y_train, y_pred):
    # 4. Evaluate the model
    mse = mean_squared_error(y_train, y_pred, squared=False)
    r2 = r2_score(y_train, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    sns.distplot(y_pred, label='prediction')
    sns.distplot(y_train, label='actual')
    
    plt.legend()