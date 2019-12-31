# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:48:49 2019

@author: Kevin
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import gc
import math
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import warnings
# stacking融合
import xgboost as xgb

start = time.time()

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('second_round_training_data.csv')
test = pd.read_csv('second_round_testing_data.csv')
submit = pd.read_csv('submit_example2.csv')
data = train.append(test).reset_index(drop=True)
dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
data['label'] = data['Quality_label'].map(dit)

feature_name = ['Parameter{0}'.format(i) for i in range(5, 11)]
tr_index = ~data['label'].isnull()
X_train = data[tr_index][feature_name].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][feature_name].reset_index(drop=True)

X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')
# y = pd.read_csv('labels.csv')
# y=y['label']
# X_train['Parameter9'] = np.log(X_train['Parameter9'])
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test= ss.fit_transform(X_test)

print(X_train.shape, X_test.shape)
oof = np.zeros((X_train.shape[0], 4))
prediction = np.zeros((X_test.shape[0], 4))

ac = np.zeros((420, 1))
mae = np.zeros((420, 1))
s00 = np.zeros((420,1))
# 特征工程
#for k in range(1,400):
#
#    cbt_model = cbt.CatBoostClassifier(iterations=1080, random_seed =k,learning_rate= 0.05, depth=6,verbose=1500,early_stopping_rounds=1000,task_type='GPU',
#                                       loss_function='MultiClass')
##cbt_model = lgb.LGBMClassifier(n_estimators = 1200, learning_rate=0.05, early_stopping_rounds = 500,verbose = 600)
##cbt_model = cbt.CatBoostClassifier(iterations=5000,depth = 7,learning_rate=0.01,random_state =2200,verbose=1000,early_stopping_rounds=1000,task_type='GPU',
##                                   loss_function='MultiClass')
#    cbt_model.fit(X_train, y ,eval_set=(X_train,y))
#    oof = cbt_model.predict_proba(X_train)
#    prediction = cbt_model.predict_proba(X_test)
#    gc.collect()     #清理内存
#
#    print('logloss',log_loss(pd.get_dummies(y).values, oof))
#    print('ac',accuracy_score(y, np.argmax(oof,axis=1)))
#    ac[k] = accuracy_score(y, np.argmax(oof,axis=1))
#    print('mae',1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof))/480))
#    mae[k]= 1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof))/480)
    

#    sub = test[['Group']]
#    prob_cols = [i for i in submit.columns if i not in ['Group']]
#    
#    for i, f in enumerate(prob_cols):
#        sub[f] = prediction[:, i]
#    for i in prob_cols:
#        sub[i] = sub.groupby('Group')[i].transform('mean')
#    
#    sub = sub.drop_duplicates()
#    
#    s00[k] = sub.iloc[0,1]

train_x,val_x,train_y,val_y = train_test_split(X_train,y,test_size = 0.2,random_state=5)

clf1 = cbt.CatBoostClassifier(iterations=1080, random_seed=205, learning_rate=0.05, depth=6, verbose=1500,
                                   early_stopping_rounds=1000, task_type='GPU',
                                   loss_function='MultiClass')

#clf1.fit(X_train,y)

clf2 = xgb.XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators = 900, silent=True,
                                objective='muliti:softmax',n_jobs = 8, subsample=0.5)
#clf2.fit(X_train,y)

#clf3 = lgb.LGBMClassifier(n_estimators = 1200, learning_rate=0.05, verbose = 600)
##clf3.fit(X_train,y)
#
#clf4 = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.015,subsample = 0.7)
## cbt_model = cbt.CatBoostClassifier(iterations=5000,depth = 7,learning_rate=0.01,random_state =2200,verbose=1000,early_stopping_rounds=1000,task_type='GPU',
##                                   loss_function='MultiClass')
##clf4.fit(X_train,y)
#
#clf5 = AdaBoostClassifier()
##clf5.fit(X_train,y)


cbt_model = VotingClassifier( estimators=[('CBT', clf1), ('XGB', clf2), ('LGB', clf3),
                 ('GBDT', clf4), ('Ada', clf5)], voting = 'soft')

ssr = cbt_model.fit(X_train, y)


oof = cbt_model.predict_proba(X_train)
prediction = cbt_model.predict_proba(X_test)
gc.collect()  # 清理内存

print('logloss', log_loss(pd.get_dummies(y).values, oof))
print('ac', accuracy_score(y, np.argmax(oof, axis=1)))
# ac[i] = accuracy_score(y, np.argmax(oof,axis=1))
print('mae', 1 / (1 + np.sum(np.absolute(np.eye(4)[y] - oof)) / 480))



#feature_importance = cbt_model.feature_importances_
#feature_importance = 100.0 * (feature_importance / feature_importance.max())

sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]

for i, f in enumerate(prob_cols):
    sub[f] = prediction[:, i]
for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')

sub = sub.drop_duplicates()



sub.to_csv("submission0.csv", index=False)

sub1 = pd.read_csv('0.6399.csv')

end = time.time()

print("it last", end - start, "seconds")
