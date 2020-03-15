import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("used_car_train_20200313_revised.csv")
data_test = pd.read_csv("used_car_testA_20200313_revised.csv")

X = data.dropna(axis=0)
y = X['price']
X = X.drop('price',axis=1) #dropped about 15000 rows with missing values

def preprocess(X_train):# transform 'notRepairedDamage column from 0.0, -, 1.0 to 0,1,2 as int'
    new_col = X_train.notRepairedDamage.map(lambda x: 1 if x == '-' else int(float(x))*2)
    X_train = X_train.drop('notRepairedDamage',axis=1)
    X_train = X_train.join(new_col)
    return X_train

(X_train, X_valid, y_train, y_valid) = train_test_split(X, y)
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)

# num_estimators = [5, 10, 20]
# for n in num_estimators:       
rf_model = RandomForestRegressor(n_estimators=10 ,n_jobs=-1, criterion='mae',verbose=True)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_valid)
mae = mean_absolute_error(preds, y_valid)
print("n_estimators {}, MAE is {}".format(10, mae))