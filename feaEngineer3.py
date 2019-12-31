
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.cluster import KMeans
import xgboost as xgb

start = time.time()

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('second_round_training_data.csv')
test = pd.read_csv('second_round_testing_data.csv')
submit = pd.read_csv('submit_example2.csv')


data = train.append(test).reset_index(drop=True)
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
data['label'] = data['Quality_label'].map(dit)

feature_name = ['Parameter{0}'.format(i) for i in range(5,11)]
tr_index = ~data['label'].isnull()
X_train = data[tr_index][feature_name].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][feature_name].reset_index(drop=True)

#找出空缺值得行数
#没有空缺的行数
tes = pd.DataFrame(columns = ['Parameter5','Parameter6','Parameter7','Parameter8','Parameter9','Parameter10'])
tr = pd.DataFrame(columns = ['Parameter5','Parameter6','Parameter7','Parameter8','Parameter9','Parameter10'])
for i in range(len(X_test)):
    if math.isnan(X_test.iloc[i,4]):
        tes = tes.append(X_test.iloc[i,:])
    else:
        tr = tr.append(X_test.iloc[i,:])
        

tr = tr.append(X_train).reset_index(drop=True)


clf = xgb.XGBClassifier(n_estimators = 1300, n_jobs =4, sub_sample=0.7,learning_rate=0.01)
#clf = cbt.CatBoostClassifier(iterations=1070, random_seed =154,learning_rate= 0.05, depth=6,verbose=1500,early_stopping_rounds=1000,task_type='GPU',
#                                   loss_function='MultiClass')
train_x = tr[['Parameter5','Parameter6','Parameter7','Parameter8','Parameter10']]
train_y = tr[['Parameter9']]

test_x = tes[['Parameter5','Parameter6','Parameter7','Parameter8','Parameter10']]
clf.fit(train_x, train_y)
predy = clf.predict(test_x)
#X_test['Parameter9'].fillna(0.5930812, inplace=True)     #取众数填充
#特征工程
#取对数， int,每个参数进行计数
#scale=MinMaxScaler().fit(Xt)
#Xt = scale.transform(Xt)
predy = pd.DataFrame(predy)
predy.columns=['Parameter9']
#tes =pd.concat(predy)
j=0

for i in range(len(X_test)):
    if math.isnan(X_test.iloc[i,4]):
        X_test.iloc[i,4] = predy.iloc[j,:].values
        j += 1

test_xx = X_test

Xt = X_train.append(X_test).reset_index(drop=True)
Xt = pd.DataFrame(Xt)
#Xt = pd.concat([data['Parameter1'], Xt], axis=1)
X_train = Xt[:12934]
X_test = Xt[12934:].reset_index(drop=True)
X_train.columns = feature_name
X_test.columns = feature_name

trains = pd.concat([X_train, y], axis=1)

#X_train['Parameter2'] = np.log10(X_train['Parameter2'])
#X_test['Parameter2'] = np.log10(X_test['Parameter2'])
X_train['Parameter5'] = np.log10(X_train['Parameter5'])
X_test['Parameter5'] = np.log10(X_test['Parameter5'])

X_train['Parameter6'] = np.log10(X_train['Parameter6'])
X_test['Parameter6'] = np.log10(X_test['Parameter6'])
X_train['Parameter7'] = np.log10(X_train['Parameter7'])
X_test['Parameter7'] = np.log10(X_test['Parameter7'])
X_train['Parameter8'] = np.log10(X_train['Parameter8'])
X_test['Parameter8'] = np.log10(X_test['Parameter8'])
X_train['Parameter9'] = np.log10(X_train['Parameter9'])
X_test['Parameter9'] = np.log10(X_test['Parameter9'])
X_train['Parameter10'] = np.log10(X_train['Parameter10'])
X_test['Parameter10'] = np.log10(X_test['Parameter10'])


#X_train['Parameter2'] = np.floor_divide(X_train['Parameter2'], 0.2)
#X_train['Parameter6'] = np.floor(np.log10(X_train['Parameter6']))
#X_test['Parameter6'] = np.floor(np.log10(X_test['Parameter6']))

#trains = pd.concat([X_train,y], axis =1)
#for i in range(len(X_train)):
#    w = X_train.loc[i,:].values
#
#    for j in range(i+1, len(X_train)):
#    
#        v = X_train.loc[j,:].values
#    
#        if (v == w).all():
#            count[i,0] += 1

# 每个特征计数
def count_cat(par, newfea, x):
    dic = {}

    for i in range(len(par)):
        par[i] = str(par[i])
        if par[i] not in dic:
        
            dic[par[i]] = 1
        else:
            dic[par[i]] += 1


    x[newfea] = 0
    for i in range(len(par)):
  
        for key, value in dic.items():
            if par[i] == key:
                x[newfea].loc[i] = dic.get(key)
#    
#X_train[feature_name] = np.floor_divide(X_train[feature_name],0.2)
#X_test[feature_name] = np.floor_divide(X_test[feature_name],0.2)
#
#count_cat(X_train['Parameter5'], 'Parameter5c', X_train)
#count_cat(X_train['Parameter2'], 'P2c',X_train)           
count_cat(X_train['Parameter6'], 'P6c',X_train )
count_cat(X_train['Parameter7'], 'P7c',X_train)
count_cat(X_train['Parameter8'], 'P8c',X_train)
#count_cat(X_train['Parameter9'], 'P9c',X_train)
#count_cat(X_train['Parameter10'], 'Parameter10c',X_train)
 
#count_cat(X_test['Parameter5'], 'Parameter5c',X_test)
#count_cat(X_test['Parameter2'], 'P2c',X_test)       
count_cat(X_test['Parameter6'], 'P6c',X_test)
count_cat(X_test['Parameter7'], 'P7c',X_test)
count_cat(X_test['Parameter8'], 'P8c',X_test)
#count_cat(X_test['Parameter9'], 'P9c',X_test)
#count_cat(X_test['Parameter10'], 'Parameter10c',X_test)
     

#X_train['P6int'] = 0
#X_test['P6int'] = 0
#X_train['P7int'] = 0
#X_test['P7int'] = 0
#X_train['P9int'] = 0
#X_test['P9int'] = 0

#for i in range(len(X_train)):
#    X_train['P5int'][i] = int(X_train['Parameter5'][i])
#    X_test['P5int'][i] = int(X_test['Parameter5'][i])
#    X_train['P6int'][i] = int(X_train['Parameter6'][i])
#    X_test['P6int'][i] = int(X_test['Parameter6'][i])
#    X_train['P7int'][i] = int(X_train['Parameter7'][i])
#    X_test['P7int'][i] = int(X_test['Parameter7'][i])
#    X_train['P9int'][i] = int(X_train['Parameter9'][i])
#    X_test['P9int'][i] = int(X_test['Parameter9'][i])
#    X_train['Parameter7'][i] = int(X_train['Parameter7'][i])
#    X_test['Parameter7'][i] = int(X_test['Parameter7'][i])
#    Xt['Parameter8'][i] = int(Xt['Parameter8'][i])
#    Xt['Parameter9'][i] = int(Xt['Parameter9'][i])
#    Xt['Parameter10'][i] = int(Xt['Parameter10'][i])




Xt = X_train.append(X_test).reset_index(drop=True)
Xt = pd.DataFrame(Xt)



#Xt['P78c'] = Xt['P7c'] * Xt['P8c']
#Xt['P78c'] = np.log(Xt['P78c'])
#def minmax(x, fea):
#    mins = min(x[fea])
#    maxs = max(x[fea])
#    x[fea] = (x[fea]- mins)/(maxs-mins)
#    
#    
#minmax(Xt, 'P6c')
#minmax(Xt,'P7c')
#minmax(Xt, 'P8c')

feature_name = ['P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P6c', 'P7c', 'P8c']
X_train = Xt[:12934]
X_test = Xt[12934:].reset_index(drop=True)
X_train.columns = feature_name
X_test.columns = feature_name


#data = train.append(test).reset_index(drop=True)
#dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
#data['label'] = data['Quality_label'].map(dit)

#feature_name = ['Parameter{0}'.format(i) for i in range(5, 11)]
#tr_index = ~data['label'].isnull()
#X_train2 = data[tr_index][feature_name].reset_index(drop=True)
#y = data[tr_index]['label'].reset_index(drop=True).astype(int)
#X_test2 = data[~tr_index][feature_name].reset_index(drop=True)

#X_train['Parameter5'] = np.log(X_train2['Parameter5'])
#X_test['Parameter5'] = np.log(X_test2['Parameter5'])
#X_train['Parameter6'] = np.log(X_train2['Parameter6'])
#X_test['Parameter6'] = np.log(X_test2['Parameter6'])
#X_train['Parameter7'] = np.log(X_train2['Parameter7'])
#X_test['Parameter7'] = np.log(X_test2['Parameter7'])

trains = pd.concat([X_train,y],axis=1)


X_train.to_csv("xtrain.csv", index=False)
X_test.to_csv("xtest.csv", index=False)


#if __name__=="__main__":
#    os.system("python test21.py")    #运行p1文件

end =time.time()

print(end-start,"seconds")
            