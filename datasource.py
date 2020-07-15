import pandas as pd
import datetime
import numpy as np
import tushare as ts
import talib as ta
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
# from sklearn.model_selection import cross_val_score,cross_val_predict
# from sklearn.naive_bayes import GaussianNB
# from sklearn import linear_model, datasets
# from sklearn import svm
# from xgboost import XGBClassifier as Xgb
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve
# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier 
# from sklearn.model_selection import GridSearchCV
# matplotlib.style.use('ggplot')

#数据获取模块
ts.set_token('fb60c870e18798f256a5d5dcd9faac6cb458ac9640144da99d998d90')
pro = ts.pro_api()

stock='601288.SH'#农行 601288.SH   #张家港 002839.SZ 
df = pro.daily(ts_code=stock, start_date='20050101',end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df=df.set_index(['trade_date'])

df1 = pro.moneyflow(ts_code=stock, start_date='20050101',end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False)
df1['trade_date'] = pd.to_datetime(df1['trade_date'])
df1=df1.set_index(['trade_date'])

df_HS300 = pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False).iloc[-len(df):,:]
df_HS300['trade_date'] = pd.to_datetime(df_HS300['trade_date'])
df_HS300=df_HS300.set_index(['trade_date'])
# df_VIX = pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date=datetime.date.today().strftime('%Y%m%d')).sort_index(axis=0, ascending=False).iloc[-len(df):,:]
# df_VIX['trade_date'] = pd.to_datetime(df_VIX['trade_date'])
# df_VIX=df_HS300.set_index(['trade_date'])

testduration=-150
rawdata=df.iloc[testduration:]['close']

#Add features in
df['close_roc'] = 100*df['close'].diff()/df['close'].shift()
df['close/open'] = 100*df['close']/df['open']
df['close/high'] = 100*df['close']/df['high']
df['close/low'] = 100*df['close']/df['low']
df['rlt_close'],df['rlt_vol'],df['rlt_amount'] = df['close_roc']/df_HS300['pct_chg'],df['vol']/df_HS300['vol'],df['amount']/df_HS300['amount']
df['MAVOL5'],df['MAVOL10'] = df['vol']/df['vol'].rolling(5).mean(),df['vol']/df['vol'].rolling(10).mean()
df['MAVOL20'],df['MAVOL150'] = df['vol']/df['vol'].rolling(20).mean(),df['vol']/df['vol'].rolling(150).mean()
df['K'], df['D'] = ta.STOCH(df['high'], df['low'],df['close'],fastk_period=12, slowk_period=4, slowk_matype=0, slowd_period=3, slowd_matype=0)
df['J']=3*df['K']-2*df['D']
df['K_diff'],df['K_ROC'],df['D_diff'],df['D_ROC'],df['J_diff'],df['J_ROC']=df['K'].diff(),df['K']/df['K'].shift(),df['D'].diff(),df['D']/df['D'].shift(),df['J'].diff(),df['J']/df['J'].shift()

df['MACD'],df['MACDsignal'],df['MACDhist'] = ta.MACD(df['close'],fastperiod=6, slowperiod=12, signalperiod=9) 

df['UpInter'] = 0
df.loc[(df['K']>df['D']) & (df['K'].shift()<df['D'].shift()) & (df['D']<=80) & (df['D_diff']>0),'UpInter']=1# Inters: K go exceeds D   
df['UpInter10'] = df['UpInter'].rolling(10).sum()
df['DnInter'] = 0
df.loc[(df['K']<df['D']) & (df['K'].shift()>df['D'].shift()) & (df['D']>=20) & (df['D_diff']<0),'DnInter']=1
df['DnInter10'] = df['DnInter'].rolling(10).sum()

df['MA5'],df['MA10']= df['close']/df['close'].rolling(5).mean(),df['close']/df['close'].rolling(10).mean()      
df['MA51']=df['close']/ ta.MA(df['close'], timeperiod=5, matype=0)      
df['MA20'],df['MA50']= df['close']/df['close'].rolling(20).mean(),df['close']/df['close'].rolling(50).mean()
df['MA100'],df['MA150']= df['close']/df['close'].rolling(100).mean(),df['close']/df['close'].rolling(150).mean()
df['VAR5']= df['close_roc'].rolling(5).std()
df['VAR10']= df['close_roc'].rolling(10).std()
df['NATR']= ta.NATR(df['high'], df['low'], df['close'],timeperiod=14)

df['MFI']= ta.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=14)
df['AD']= ta.AD(df['high'], df['low'], df['close'], df['vol'])
df['ADOSC']= ta.ADOSC(df['high'], df['low'], df['close'], df['vol'], fastperiod=3, slowperiod=10)
df['net_mf5']=(df1['net_mf_vol']-df1['net_mf_vol'].rolling(10).mean())/df1['net_mf_vol'].rolling(10).std()
df['net_mf50']=(df1['net_mf_vol']-df1['net_mf_vol'].rolling(50).mean())/df1['net_mf_vol'].rolling(50).std()
df['net_mf100']=(df1['net_mf_vol']-df1['net_mf_vol'].rolling(100).mean())/df1['net_mf_vol'].rolling(100).std()

df['buy_vol']=df1['buy_sm_vol']+df1['buy_md_vol']+df1['buy_lg_vol']+df1['buy_elg_vol']
df['sell_vol']=df1['sell_sm_vol']+df1['sell_md_vol']+df1['sell_lg_vol']+df1['sell_elg_vol']
df['buy/sell']=df['buy_vol']/df['sell_vol']

df['buy_sm']=df1['buy_sm_vol']/df['buy_vol']
df['sell_sm']=df1['sell_sm_vol']/df['sell_vol']
df['buy_md']=df1['buy_md_vol']/df['buy_vol']
df['sell_md']=df1['sell_md_vol']/df['sell_vol']
df['buy_lg']=df1['buy_lg_vol']/df['buy_vol']
df['sell_lg']=df1['sell_lg_vol']/df['sell_vol']
df['buy_elg']=df1['buy_elg_vol']/df['buy_vol']
df['sell_elg']=df1['sell_elg_vol']/df['sell_vol']


featurelist=['close_roc','close/open','close/high','close/low',
             'UpInter10','UpInter','DnInter10','DnInter',
             'rlt_close','rlt_vol','rlt_amount',
             'MAVOL20','MAVOL10','MAVOL5','MAVOL150',
             'MA5','MA10','MA20','MA50','MA100', 'MA150',
             'K','D','J','K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff',
             'VAR5','VAR10','NATR',
             'MFI','AD','ADOSC',
             'net_mf5','net_mf50','net_mf100',
             'MACD','MACDsignal','MACDhist',
             'buy/sell','buy_sm','sell_sm','buy_md','sell_md','buy_lg','sell_lg','buy_elg','sell_elg'
             ]

