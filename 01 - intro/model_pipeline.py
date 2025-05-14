from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd


def train_preset(df_train, categorical, numerical, target):
    dv = DictVectorizer()
    train_dict = df_train[categorical + numerical].to_dict(orient='records')    
    X_train = dv.fit_transform(train_dict)
    y_train = df_train[target].values
    
    return X_train , y_train


def test_preset(df_test, categorical, numerical, target):
    dv = DictVectorizer()
    test_dict = df_test[categorical + numerical].to_dict(orient='records')    
    X_test = dv.transform(test_dict)
    y_test = df_test[target].values
    return X_test,y_test


def train_model(X_train, y_train):
    
    # 1. Initialize the model
    model = LinearRegression()
    
    # 2. Train the model
    model.fit(X_train, y_train)

    return model



def predict(model, X_test):
        # 3. Make predictions
    y_pred = model.predict(X_test)

    return y_pred