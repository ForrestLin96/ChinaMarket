import pandas as pd
import datetime
import numpy as np
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_recall_curve
matplotlib.style.use('ggplot')

def plot_precision_recall_vs_threshold(index,stock,method_list,precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    # plt.xlim([0.5, thresholds.max()])
    plt.xlim([thresholds.min(), thresholds.max()])
    plt.ylim([0, 1])
    plt.title(stock+'\n'+method_list.columns[index])
#Define Stochastic Osciliator
def calculate_k(df,cycle, M1 ):
    close = df['close']
    highest_hi = df['high'].rolling(window = cycle).max()
    lowest_lo = df['low'].rolling(window=10).min()
    df['rsv'] = (close - lowest_lo)/(highest_hi - lowest_lo)*100

#Evenly separeate all days into good selling points and poor selling points
def selljudge(df,cycle=10,duration=-180):
    df['Good sell Point?'] =0
    df['10天最低价'] =df['close'].rolling(window =cycle).min().shift(-cycle)/df['close']
    df.loc[(df['10天最低价']<df['10天最低价'][:duration,].quantile()),'Good sell Point?'] = 1 #6/18新改动在train group里分好坏
    del df['10天最低价']
    return df
    
def buyjudge(df,cycle=10,duration=-180):
    df['Good buy Point?'] =0
    df['10天最高价'] =df['close'].rolling(window =cycle).max().shift(-cycle)/df['close']
    df.loc[(df['10天最高价']>df['10天最高价'].iloc[:duration,].quantile()),'Good buy Point?'] = 1
    df=df.drop('10天最高价', axis=1)
    return df

def plot_buy(name,dfplot,stock,a=0.93,b=0.99,c=0.015):
    for ratio in np.arange(a,b,c):
        dfplot['buy']=0
        dfplot['buyPrice']=0
        dfplot.loc[(dfplot['GoodbuyProb']>dfplot['GoodbuyProb'].quantile(ratio)),'buy'] = 1
        dfplot.loc[(dfplot['buy']==1),'buyPrice'] = dfplot['close']
        buyratio=round(100*dfplot['buy'].sum()/len(dfplot['buy']),2)
        x=dfplot.index
        y1=dfplot['close']
        y2=dfplot['buyPrice']
        plt.plot(x, y1,'c',label='Price')
        plt.plot(x, y2, 'o', ms=4.5, label='buy Point')
        plt.ylim([min(y1)*0.98, max(y1)*1.02])
        plt.title(stock+'\n'+name+'\nbuy Ratio='+str(buyratio)+'%, '+'Threshold='+str(round(dfplot['GoodbuyProb'].quantile(ratio),3)))
        plt.figtext(0.7,0.5,dfplot.iloc[-1,].name.strftime('%b.%d')+':'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
        plt.figtext(0.7,0.45,'1 day:'+str(round(dfplot.iloc[-2,1],3)) , fontsize=13)
        plt.figtext(0.7,0.4,'2 day:'+str(round(dfplot.iloc[-3,1],3)) , fontsize=13)
        #plt.legend(loc='upper left')
        plt.show()

def plot_sell(name,dfplot,stock,a=0.93,b=0.99,c=0.015):
    for ratio in np.arange(a,b,c):
        dfplot['sell']=0
        dfplot['sellPrice']=0
        dfplot.loc[(dfplot['GoodsellProb']>dfplot['GoodsellProb'].quantile(ratio)),'sell'] = 1
        dfplot.loc[(dfplot['sell']==1),'sellPrice'] = dfplot['close']
        sellratio=round(100*dfplot['sell'].sum()/len(dfplot['sell']),2)
        x=dfplot.index
        y1=dfplot['close']
        y2=dfplot['sellPrice']
        plt.plot(x, y1,'c',label='Price')
        plt.plot(x, y2, 'o', ms=4.5, label='sell Point',color='blue')
        plt.ylim([min(y1)*0.98, max(y1)*1.02])
        plt.title(stock+'\n'+name+'\nsell Ratio='+str(sellratio)+'%, '+'Threshold='+str(round(dfplot['GoodsellProb'].quantile(ratio),3)))
        plt.figtext(0.7,0.5,dfplot.iloc[-1,].name.strftime('%b.%d')+':'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
        plt.figtext(0.7,0.45,'1 day:'+str(round(dfplot.iloc[-2,1],3)), fontsize=13)
        plt.figtext(0.7,0.4,'2 day:'+str(round(dfplot.iloc[-3,1],3)) , fontsize=13)
        #plt.legend(loc='upper left')
        plt.show()


