import pandas as pd
import numpy as np
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
# rf_model = RandomForestRegressor(n_estimators=10 ,n_jobs=-1, criterion='mae',verbose=True)
# rf_model.fit(X_train, y_train)
# preds = rf_model.predict(X_valid)
# mae = mean_absolute_error(preds, y_valid)
# print("n_estimators {}, MAE is {}".format(10, mae))

## split X_train into 3 sets, train 3 rf models then take y as y_avg of 3 models
def train_rf_model(X, y):
	rf_model = RandomForestRegressor(n_estimators=10, n_jobs=-1, criterion='mae', verbose=1)
	rf_model.fit(X,y)
	return rf_model

def validate(model, X_valid, y_valid):
	preds = model.predict(X_valid)
	mae = mean_absolute_error(preds, y_valid)
	return (preds, mae)
X_train_len = X_train.shape[0]
model_count = 3 #how many models to train and how many subsets to split
X_subtrains = [X_train.iloc[0:X_train_len//model_count,:], X_train.iloc[X_train_len//model_count:X_train_len*2//model_count,:], X_train.iloc[X_train_len*2//model_count:,:]]
y_subtrains = [y_train[subtrain.index] for subtrain in X_subtrains]
# print(X_train.head())
# print(y_train.head())
# print('\n')
# for i in range(3):
# 	print(X_subtrains[i].head())
# 	print(y_subtrains[i].head())
# 	print('\n')

# print(X_train.shape)
# print(y_train.shape)
# print('\n')
# for i in range(3):
# 	print(X_subtrains[i].shape)
# 	print(y_subtrains[i].shape)
# 	print('\n')
rf_models = []
for i in range(3):
	rf_models.append(train_rf_model(X_subtrains[i],y_subtrains[i]))
	
import pickle
from joblib import dump
for i in range(3):
	dump(rf_models[i], 'bundled_rf_model{}.joblib'.format(i))


(preds1, mae1) = validate(rf_models[0], X_valid, y_valid)
(preds2, mae2) = validate(rf_models[1], X_valid, y_valid)
(preds3, mae3) = validate(rf_models[2], X_valid, y_valid)
preds_avg = (preds1+preds2+preds3)/3
mae = mean_absolute_error(preds_avg, y_valid)
print(mae1)
print(mae2)
print(mae3)
print(mae)


