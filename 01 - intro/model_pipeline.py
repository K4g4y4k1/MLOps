from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd


def train_preset(df_train):
    dv = DictVectorizer()
    train_dict = df_train.to_dict(orient='records')    
    X_train = dv.fit_transform(train_dict)
    
    return X_train, dv



def val_preset(df_test, dv):
    test_dict = df_test.to_dict(orient='records')
    X_test = dv.transform(test_dict)
    
    return X_test


def train_model(X_train, df, target):
    y_train = df[target].values
    
    # 1. Initialize the model
    model = LinearRegression()
    
    # 2. Train the model
    model.fit(X_train, y_train)

    return model




def predict(model, X_test):
        # 3. Make predictions
    y_pred = model.predict(X_test)

    return y_pred