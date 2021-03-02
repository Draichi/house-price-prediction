import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

X, y = data.iloc[:, :-1], data.iloc[:, -1]

data_matrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3,
                          learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)
print('> X_test:', X_test)
print('> Preds:', preds)
print('> Len:', len(preds))
print('> Y Test:', y_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print('> RMSE:', rmse)
