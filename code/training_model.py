import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from joblib import dump, load


df = pd.read_csv("data.csv")

random_state = 500

# extracting features

Feature = df[['Sum', 'Max', 'Y_mass', 'More_than', "weighted_sum", '0', '1', '2', '3', '4', '5', '6', '7']]
X = Feature.values
y = df['Label'].values
oversample = SMOTE()

X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# scaling data


scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# training model with xgboost

learning_rate_list = [ 0.1, 0.01]
max_depth_list = [3, 4, 5, 6]
n_estimators_list = [500, 1000, 2000]

params_dict = {"learning_rate": learning_rate_list,
               "max_depth": max_depth_list,
               "n_estimators": n_estimators_list}

model_xgboost = xgb.XGBClassifier(learning_rate=0.01,
                                      max_depth=5,
                                      n_estimators=3000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      eval_metric="auc"
                                      )

# evaluation

eval_set = [(X_test, y_test)]
model_xgboost.fit(X_train, y_train)
y_train_pred = model_xgboost.predict(X_train)
y_test_pred = model_xgboost.predict(X_test)
print(f1_score(y_train, y_train_pred, average='micro'))
print(f1_score(y_test, y_test_pred, average='micro'))

plot_confusion_matrix(model_xgboost, X_test, y_test)
plot.show()

#dump(model_xgboost, 'model_new.joblib')
#dump(scaler, 'scaler_new.joblib')


# model_xgboost_hp = GridSearchCV(estimator=xgb.XGBClassifier(subsample=0.5,
#                                                                 colsample_bytree=0.25,
#                                                                 eval_metric='auc',
#                                                                 use_label_encoder=False),
#                                 param_grid=params_dict,
#                                 cv=2,
#                                 scoring='f1_micro',
#                                 return_train_score=True,
#                                 verbose=4)
#
# model_xgboost_hp.fit(X, y)
#
# df_cv_results = pd.DataFrame(model_xgboost_hp.cv_results_)
# df_cv_results = df_cv_results[['rank_test_score','mean_test_score','mean_train_score',
#                                'param_learning_rate', 'param_max_depth', 'param_n_estimators']]
# df_cv_results.sort_values(by='rank_test_score', inplace=True)
# print(df_cv_results)