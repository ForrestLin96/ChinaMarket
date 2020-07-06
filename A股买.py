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
from Afunction import buyjudge,stochastic_oscillator,plot_precision_recall_vs_threshold,plot_buy
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier   
matplotlib.style.use('ggplot')
#%%数据获取模块
ts.set_token('fb60c870e18798f256a5d5dcd9faac6cb458ac9640144da99d998d90')
pro = ts.pro_api()

stock='601288.SH'#农行 601288.SH   
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
#%%
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

rawdata=df.iloc[testduration:]['close']
#Add features in
df['MAVOL150'] = df['vol']/df['vol'].rolling(150).mean()
df['MAVOL20'] = df['vol']/df['vol'].rolling(20).mean()
df['MAVOL10'] = df['vol']/df['vol'].rolling(10).mean()
df['MAVOL5'] = df['vol']/df['vol'].rolling(5).mean()
df['amount100'] = df['amount']/df['amount'].rolling(100).mean()
df['amount20'] = df['amount']/df['amount'].rolling(20).mean()
df['amount10'] = df['amount']/df['amount'].rolling(10).mean()
df['amount5'] = df['amount']/df['amount'].rolling(5).mean()
df['HS300'] = df_HS300['close']
df['HS300_ROC'] = 100*df['HS300'].diff(1)/df['HS300'].shift(1)
df['close_ROC'] = 100*df['close'].diff(1)/df['close'].shift(1)
df['close/HS300'] = df['close_ROC']/df['close_ROC']  
stochastic_oscillator(df)
df['UpInter'] = 0
df.loc[(df['K']>df['D']) & (df['K_prev']<df['D_prev']) & (df['D']<=80) & (df['D_diff']>0),'UpInter']=1# Inters: K go exceeds D   
df['UpInter10'] = df['UpInter'].rolling(10).sum()# number of Inters during past 10 days
df['DnInter'] = 0
df.loc[(df['K']<df['D']) & (df['K_prev']>df['D_prev']) & (df['D']>=20) & (df['D_diff']<0),'DnInter']=1
df['DnInter10'] = df['DnInter'].rolling(10).sum()      
df['close/MA10']= df['close']/df['close'].rolling(10).mean()
df['close/MA20']= df['close']/df['close'].rolling(20).mean()
df['close/MA50']= df['close']/df['close'].rolling(50).mean()
df['close/MA100']= df['close']/df['close'].rolling(100).mean()
df['close/MA150']= df['close']/df['close'].rolling(150).mean()
df['VAR5']= df['close_ROC'].rolling(5).std()
df['VAR10']= df['close_ROC'].rolling(10).std()
buyjudge(df,duration=testduration)
featurelist=['amount100','amount20','amount10','amount5',
             'UpInter10','UpInter','DnInter10','DnInter',
             'close/HS300',
             'MAVOL20','MAVOL10','MAVOL5','MAVOL150',
            'close_ROC','rsv','K','D','J',
            'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','close/MA10',
            'close/MA20','close/MA50','close/MA100',
            'close/MA150',
            'VAR5','VAR10']
  
df=df.drop(columns=['10天最高价'])
df.dropna(axis=0, how='any', inplace=True)
xshow=df.iloc[testduration:,:].loc[:,featurelist]
xshow = preprocessing.MinMaxScaler().fit_transform(xshow)
if None in xshow[-1,:]:
    xshow=np.delete(xshow,-1,0)
X=df.loc[:,featurelist]
X = preprocessing.MinMaxScaler().fit_transform(X)

y=df.loc[:,'Good Buy Point?']
# Split train set and test set
xtrain,ytrain=X[:testduration],y[:testduration]
xtest,ytest=X[testduration:],y[testduration:]

Market_GoodRatio=sum(df['Good Buy Point?'].iloc[:testduration,]==1)/len(df['Good Buy Point?'].iloc[:testduration,])#Good Buy Point Ratio in market is manully set to nearly 0.5 
ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good Buy Ratio','AvgScores':Market_GoodRatio,'StdScores':0},ignore_index=True)

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

name_list= ['Market Good Buy Ratio']
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
     buypredicted = clf.predict_proba(xtest)
     precision, recall, threshold = precision_recall_curve(ytest, buypredicted[:,1])
     plot_precision_recall_vs_threshold(index,stock,method_list,precision, recall, threshold)
     plt.show()
     index=index+1
#%%       Naive Bayes       
clfbuy =GaussianNB(var_smoothing=1) 
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('Naive Bayes',dfplot,stock,0.9,1,0.03)
#%%  SVM             
clfbuy = svm.SVC(C=1,kernel='linear',probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('SVM Linear',dfplot,stock,0.9,0.99,0.02)
#%%  SVM             
clfbuy = svm.SVC(C=1,probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('SVM ',dfplot,stock,0.9,0.99,0.02)
#%%  SVM Poly         
clfbuy = svm.SVC(C=1,kernel='poly',probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('SVM Poly',dfplot,stock,0.93,0.99,0.015)

#%%  Random Forrest       
clfbuy =RandomForestClassifier(oob_score=True, random_state=30)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('Random Forrest',dfplot,stock,0.93,0.99,0.015)
#%%       XGboost       
clfbuy =Xgb(reg_lambda=1)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('XGBoost',dfplot,stock,0.9,1,0.03)
