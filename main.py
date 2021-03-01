import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

boston = load_boston()

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

X, y = data.iloc[:, :-1], data.iloc[:, -1]

data_matrix = xgb.DMatrix(data=X, label=y)
