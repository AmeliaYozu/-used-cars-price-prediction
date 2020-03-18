import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from joblib import dump

import logging
logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

data = pd.read_csv("./data/used_car_train_20200313_revised.csv", index_col='SaleID')

def preprocess(X_train):
    '''
    transform 'notRepairedDamage column from 0, -, 1 to 0,1,2
    return transformed dataset
    '''
    new_col = X_train.notRepairedDamage.map(lambda x: 1 if x == '-' else int(float(x))*2)
    X_train = X_train.drop('notRepairedDamage',axis=1)
    X_train = X_train.join(new_col)   
    return X_train

def validate(model, X_train, X_valid, y_train, y_valid):
    '''
    return validation scores of training data & validation data
    '''
    preds_valid = model.predict(X_valid)
    preds_train = model.predict(X_train)
    mae_valid = mean_absolute_error(preds_valid, y_valid)
    mae_train = mean_absolute_error(preds_train, y_train)
    print("Validation result:")
    print("train set mae on training set is {}".format(mae_train))
    print("valid set mae on validation set is {}".format(mae_valid))


# Preprocess data:
logging.info('Started...')
y = data['price']
X = data.drop('price',axis=1) #dropped about 15000 rows with missing values

full_cols = X.columns
selected_cols=['v_12','v_10','regDate','kilometer','v_0','v_14',
               'power','v_8','v_1','v_5','v_3','v_11',
               'v_9','v_6','v_4','notRepairedDamage','model',
               'v_2','v_13','name','brand','v_7','fuelType']

#split dataset & handling cat features
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.1)
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
cols = X_train.columns

#Imputation
imputer = SimpleImputer()
X_train = pd.DataFrame(imputer.fit_transform(X_train))
X_valid = pd.DataFrame(imputer.transform(X_valid))
X_train.columns = cols
X_valid.columns = cols

# (X_train, y_train) = X_train[:10], y_train[:10]

# Tune HyperParameters
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10,110,num=11)]
max_depth.append(None)
bootstrap = [True, False]
max_leaf_nodes = [int(x) for x in np.linspace(start=100,stop=1000,num=10)]
max_leaf_nodes.append(None)
min_impurity_decrease = [float(x/100) for x in np.linspace(start=0, stop=50, num=11)]
warm_start=[True, False]

random_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'max_depth':max_depth,
              'bootstrap':bootstrap,
              'max_leaf_nodes':max_leaf_nodes,
              'min_impurity_decrease':min_impurity_decrease,
              'warm_start':warm_start}
logging.info('Data preprocessing finished.\n')
for cols in [selected_cols, full_cols]:
    XX_train = X_train[cols]
    XX_valid = X_valid[cols]
    # Fit and validate model to get the best model settings (hyperparameters)
    rf = RandomForestRegressor(criterion='mae')
    print("====================================================================================================")
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, 
                                verbose=1, random_state=42, n_jobs=-1)
    rf_random.fit(XX_train, y_train)
    dump(rf_random, 'rf_random_grid_search_buffer_{}_cols.joblib'.format(len(cols)))
    print("----------------------------------------------------------------------------------------------------")
    logging.info("Best score of RandomizedSearchCV's best estimator: {}".format(rf_random.best_score_))
    print("----------------------------------------------------------------------------------------------------")
    logging.info("[Done] Validation score - RandomizedSearchCV on {} features:".format(len(cols)))
    validate(rf_random.best_estimator_, XX_train, XX_valid, y_train, y_valid)
    print("----------------------------------------------------------------------------------------------------")
    # use best params to train rf_model
    rf_model = RandomForestRegressor(criterion='mae', **rf_random.best_params_)
    rf_model.fit(XX_train, y_train)
    dump(rf_model, 'rf_model_buffer_{}_cols.joblib'.format(len(cols)))
    logging.info("[Done] Validation score - RandomForest(best_params_) on {} features:".format(len(cols)))
    validate(rf_model, XX_train, XX_valid, y_train, y_valid)







