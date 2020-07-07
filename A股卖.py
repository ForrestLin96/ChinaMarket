import pandas as pd
import datetime
import numpy as np
import tushare as ts
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn import svm
from xgboost import XGBClassifier as Xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from Afunction import selljudge,stochastic_oscillator,plot_precision_recall_vs_threshold,plot_sell
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier   
matplotlib.style.use('ggplot')

#数据获取模块
ts.set_token('fb60c870e18798f256a5d5dcd9faac6cb458ac9640144da99d998d90')
pro = ts.pro_api()

stock='002839.SZ '#农行 601288.SH   #张家港 002839.SZ 
df = pro.daily(ts_code=stock, start_date='20050101',end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df=df.set_index(['trade_date'])

df_HS300 = pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False).iloc[-len(df):,:]
df_HS300['trade_date'] = pd.to_datetime(df_HS300['trade_date'])
df_HS300=df_HS300.set_index(['trade_date'])
# df_VIX = pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False).iloc[-len(df):,:]
# df_VIX['trade_date'] = pd.to_datetime(df_VIX['trade_date'])
# df_VIX=df_HS300.set_index(['trade_date'])
testduration=-90
rawdata=df.iloc[testduration:]['close']

method_name = [{
                'Random Forrest':RandomForestClassifier(),
                'Random Forrest30':RandomForestClassifier(oob_score=True, random_state=30),
                'Bayes(smo=1e-01)':GaussianNB(var_smoothing=1e-01),
                'Bayes(smo=0.5)':GaussianNB(var_smoothing=0.5),
                'Bayes(smo=1)':GaussianNB(var_smoothing=1),
                'SVC(C=1)':svm.SVC(probability=True),
                'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                'SVC(poly, C=1)':svm.SVC(kernel='poly',probability=True),
                'XGBT(λ=1)':Xgb(reg_lambda=1),#Result of parameter tunning in XGBPara.py
                'XGBT(λ=0.8)':Xgb(reg_lambda=0.8)
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])

#Add features in
df['close_roc'] = 100*df['close'].diff()/df['close'].shift()
df['close/open'] = 100*df['close']/df['open']
df['close/high'] = 100*df['close']/df['high']
df['close/low'] = 100*df['close']/df['low']
df['rlt_close'],df['rlt_vol'],df['rlt_amount'] = df['close_roc']/df_HS300['pct_chg'],df['vol']/df_HS300['vol'],df['amount']/df_HS300['amount']
df['MAVOL5'],df['MAVOL10'] = df['vol']/df['vol'].rolling(5).mean(),df['vol']/df['vol'].rolling(10).mean()
df['MAVOL20'],df['MAVOL150'] = df['vol']/df['vol'].rolling(20).mean(),df['vol']/df['vol'].rolling(150).mean()
df['amount5'],df['amount10'] = df['amount']/df['amount'].rolling(5).mean(),df['amount']/df['amount'].rolling(10).mean()
df['amount20'],df['amount100'] = df['amount']/df['amount'].rolling(20).mean(),df['amount']/df['amount'].rolling(100).mean()
df=stochastic_oscillator(df)

df['UpInter'] = 0
df.loc[(df['K']>df['D']) & (df['K'].shift()<df['D'].shift()) & (df['D']<=80) & (df['D_diff']>0),'UpInter']=1# Inters: K go exceeds D   
df['UpInter10'] = df['UpInter'].rolling(10).sum()
df['DnInter'] = 0
df.loc[(df['K']<df['D']) & (df['K'].shift()>df['D'].shift()) & (df['D']>=20) & (df['D_diff']<0),'DnInter']=1
df['DnInter10'] = df['DnInter'].rolling(10).sum()
      
df['close/MA10']= df['close']/df['close'].rolling(10).mean()
df['close/MA20']= df['close']/df['close'].rolling(20).mean()
df['close/MA50']= df['close']/df['close'].rolling(50).mean()
df['close/MA100']= df['close']/df['close'].rolling(100).mean()
df['close/MA150']= df['close']/df['close'].rolling(150).mean()
df['VAR5']= df['close_roc'].rolling(5).std()
df['VAR10']= df['close_roc'].rolling(10).std()
df=selljudge(df,duration=testduration)
featurelist=['close_roc','close/open','close/high','close/low',
             'amount100','amount20','amount10','amount5',
             'UpInter10','UpInter','DnInter10','DnInter',
             'rlt_close','rlt_vol','rlt_amount',
             'MAVOL20','MAVOL10','MAVOL5','MAVOL150',
             'close/MA10','close/MA20','close/MA50','close/MA100', 'close/MA150',
             'rsv','K','D','J',
             'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff',
             'VAR5','VAR10']
  
df.dropna(axis=0, how='any', inplace=True)
xshow=df.iloc[testduration:,:].loc[:,featurelist]
xshow = preprocessing.MinMaxScaler().fit_transform(xshow)
if None in xshow[-1,:]:
    xshow=np.delete(xshow,-1,0)
X=df.loc[:,featurelist]
X = preprocessing.MinMaxScaler().fit_transform(X)
#X = preprocessing.StandardScaler().fit_transform(X)

y=df.loc[:,'Good sell Point?']
# Split train set and test set
xtrain,ytrain=X[:testduration],y[:testduration]
xtest,ytest=X[testduration:],y[testduration:]

Market_GoodRatio=sum(df['Good sell Point?'].iloc[:testduration,]==1)/len(df['Good sell Point?'].iloc[:testduration,])#Good sell Point Ratio in market is manully set to nearly 0.5 
ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good sell Ratio','AvgScores':Market_GoodRatio,'StdScores':0},ignore_index=True)

#Compare and Plot the precision rate of each algorithm        
index=0
for method in method_list.loc[0,:]:
    clf = method
    cv=TimeSeriesSplit(n_splits=4) #Time series test
    scores = cross_val_score(clf,xtrain, ytrain, cv=4,scoring='precision')
    print(scores[scores>0])
    series={'Stock':stock,'Method':method_list.columns[index],'AvgScores':scores[scores>0].mean(),'StdScores':scores[scores>0].std()}
    index=index+1
    ResultTable=ResultTable.append(series,ignore_index=True)

name_list= ['Market Good sell Ratio']
name_list=np.append(name_list,method_list.columns)
num_list= ResultTable.loc[ResultTable['Stock']==stock]['AvgScores']
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.title(stock+'\nPrecision Rate')
plt.show()
    
#Plot precision rate of each method 
index=0
for method in method_list.loc[0,:]:
     clf = method
     clf.fit(xtrain, ytrain)
     sellpredicted = clf.predict_proba(xtest)
     precision, recall, threshold = precision_recall_curve(ytest, sellpredicted[:,1])
     plot_precision_recall_vs_threshold(index,stock,method_list,precision, recall, threshold)
     plt.show()
     index=index+1
#%%       Naive Bayes       
clfsell =GaussianNB(var_smoothing=1) 
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodsellProb']=sellpredicted[:,1]
plot_sell('Naive Bayes',dfplot,stock,0.9,1,0.03)
#%%  SVM             
clfsell = svm.SVC(C=1,kernel='linear',probability=True)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodsellProb']=sellpredicted[:,1]
plot_sell('SVM Linear',dfplot,stock,0.9,0.99,0.02)
#%%  SVM             
clfsell = svm.SVC(C=1,probability=True)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodsellProb']=sellpredicted[:,1]
plot_sell('SVM ',dfplot,stock,0.9,0.99,0.02)
#%%  SVM Poly         
clfsell = svm.SVC(C=1,kernel='poly',probability=True)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodsellProb']=sellpredicted[:,1]
plot_sell('SVM Poly',dfplot,stock,0.93,0.99,0.015)

#%%  Random Forrest       
clfsell =RandomForestClassifier(oob_score=True, random_state=30)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodsellProb']=sellpredicted[:,1]
plot_sell('Random Forrest',dfplot,stock,0.93,0.99,0.015)
#%%       XGboost       
clfsell =Xgb(reg_lambda=1)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodsellProb']=sellpredicted[:,1]
plot_sell('XGBoost',dfplot,stock,0.9,1,0.03)

