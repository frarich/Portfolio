# Disclaimer
"""
This package and all code is property of Fiji Water Partners.

Unless otherwise noted, all materials, including but not limited to images, illustrations, designs, icons, photographs, video clips, software, and written and other materials that are part of this project
 or any other FIJI WATER material are protected under copyright laws and are the trademarks, trade dress and/or other intellectual properties owned, controlled or licensed by FIJI WATER.
No part of these materials may otherwise be copied, reproduced, stored, republished, uploaded, posted, transmitted, or distributed in any form or by any means, electronic or mechanical, now known or hereafter invented, without the prior written permission from FIJI WATER.

"""

# importing the required packages
from os.path import exists
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.stats import norm, kurtosis
from scipy.optimize import minimize
from matplotlib.pyplot import rcParams
from numpy.linalg import multi_dot
from statsmodels.regression.rolling import RollingOLS
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from binance_data.client import DataClient
from arch import arch_model
from arch.univariate import GARCH, EWMAVariance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
import statsmodels.api as sm
import seaborn as sns
import statsmodels.tsa.stattools as ts
# import flair
import itertools
import urllib.request
import time
import json
import requests
import locale
import warnings
import re
import quandl
# import pickle5 as pickle
sns.set_style("white")

### API Keys
rcParams['figure.figsize'] = 16, 8
quandl.ApiConfig.api_key = "hJrpPb7hMqbAvC9tkQZy"


gn_api_key = '25RRFJd5tzkrpmOoeH0Pev8RwXM'

warnings.filterwarnings('ignore')
# sentiment_model = flair.models.TextClassifier.load('sentiment-fast')

plt.rcParams['figure.figsize'] = 16, 8

# to perform rolling calculation

# at the end before submitting, suppress all the warnings


#=======================================================================================================================

                # Getting the data and placing it in dataframes and merging with the futures data

#=======================================================================================================================

# We get data for the S&P 500 and the CBOE VIX index
tickers = ['BTC-USD', 'ADA-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD', 'ATOM-USD',
           'LINK-USD', 'DOGE-USD', 'SHIB-USD']

crypto_list = ['BTC-USD', 'ADA-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD', 'ATOM-USD',
               'LINK-USD', 'DOGE-USD', 'SHIB-USD']
ticker_list = ['SPX', 'VIX', 'SVXY', 'BTC-USD', 'ADA-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD',
               'LTC-USD', 'ATOM-USD', 'LINK-USD', 'DOGE-USD', 'SHIB-USD','BTCFUT']
start = '2015-01-01'
end = '2022-02-20'

### Data Initialization for UTXO/SOPR

res_btc = requests.get('https://api.glassnode.com/v1/metrics/indicators/sopr', params={'a': 'BTC', 'api_key': gn_api_key})
df_btc  = pd.read_json(res_btc.text, convert_dates=['t']).set_index('t')
res_eth = requests.get('https://api.glassnode.com/v1/metrics/indicators/sopr', params={'a': 'ETH', 'api_key': gn_api_key})
df_eth  = pd.read_json(res_eth.text, convert_dates=['t']).set_index('t')


def getTabulatedPriceVolume(crypto_list, start, end):
    """
    This function gets the both the price and volume data for the set of cryptos and indicies from yahoo
    finance using unique columns



    :returns a data frame containing the data
    """

    all_info = pd.DataFrame()
    for c in tqdm(crypto_list):
        if exists("{}_Tab_PriceVolume.pkl".format(c)):
            store = pd.concat([all_info, pd.read_pickle('{}_Tab_PriceVolume.pkl'.format(c))], 1)
        else:
            store_close = yf.download(c, start="2015-01-01", progress=False)[['Close']]
            store_close = store_close.rename(columns={"Close": "{}_Close".format(c)})
            store_volume = yf.download(c, start="2015-01-01", progress=False)[['Volume']]
            store_volume = store_volume.rename(columns={"Volume": "{}_Volume".format(c)})
            store = pd.concat([store_close, store_volume], 1)
            store['{}_Market_Cap'.format(c)] = store['{}_Close'.format(c)] * store['{}_Volume'.format(c)]
            store.to_pickle('{}_Tab_PriceVolume.pkl'.format(c))
        all_info = store
    spx = yf.download(['^GSPC'], start="2015-01-01", progress=False)[['Close', 'Volume']]
    spx = spx.assign(SPX_Market_Cap=spx['Close'] * spx['Volume'])
    spx = spx.rename(columns={'Close': 'SPX_Close', 'Volume': 'SPX_Volume'})
    vix = yf.download(['^VIX'], start="2015-01-01", progress=False)[['Close']]
    vix = vix.rename(columns={'Close': 'VIX_Close'})
    svxy = yf.download(['SVXY'], start="2015-01-01", progress=False)[['Close', 'Volume']]
    svxy = svxy.assign(SVXY_Market_Cap=svxy['Close'] * svxy['Volume'])
    svxy = svxy.rename(columns={'Close': 'SVXY_Close', 'Volume': 'SVXY_Volume'})
    all_info = pd.concat([spx, vix, svxy, all_info], 1).ffill().fillna(0)
    return all_info

### Function to download price and volume for cryptos/indices/ETFs from yahoo finance, using truncated columns

def getPriceVolume(crypto_list, start, end):
    all_info = pd.DataFrame()
    for c in tqdm(crypto_list):
        if exists("{}_PriceVolume.pkl".format(c)):
            all_info = pd.concat([all_info, pd.read_pickle('{}_PriceVolume.pkl'.format(c))], 0)
        else:
            store_close = yf.download(c, start="2015-01-01", progress=False)[['Close']]
            store_volume = yf.download(c, start="2015-01-01", progress=False)[['Volume']]
            store = pd.concat([store_close, store_volume], 1)
            store = store.assign(Ticker=c)
            store = store.set_index(['Ticker', store.index])
            store = store.assign(Market_Cap=store['Close'] * store['Volume'])
            store.to_pickle('{}_PriceVolume.pkl'.format(c))
            all_info = pd.concat([all_info, store], 0)
    dates = pd.DataFrame(all_info.reset_index()['Date'].unique(), columns=['Date'])
    spx = yf.download(['^GSPC'], start="2015-01-01", progress=False)[['Close', 'Volume']]
    spx = spx.assign(Market_Cap=spx['Close'] * spx['Volume'])
    spx = spx.assign(Ticker='SPX')
    spx = spx.merge(dates, how='right', on='Date').ffill().dropna()
    spx = spx.set_index(['Ticker', 'Date'])
    vix = yf.download(['^VIX'], start="2015-01-01", progress=False)[['Close']]
    vix = vix.assign(Ticker='VIX')
    vix = vix.merge(dates, how='right', on='Date').ffill().dropna()
    vix = vix.set_index(['Ticker', 'Date'])
    svxy = yf.download(['SVXY'], start="2015-01-01", progress=False)[['Close', 'Volume']]
    svxy = svxy.assign(Market_Cap=svxy['Close'] * svxy['Volume'])
    svxy = svxy.assign(Ticker='SVXY')
    svxy = svxy.merge(dates, how='right', on='Date').ffill().dropna()
    svxy = svxy.set_index(['Ticker', 'Date'])
    all_info = pd.concat([spx, vix, svxy, all_info], 0).ffill().fillna(0)
    return all_info

def get_TabulatedPriceVolumeData():
    crypto_data = getTabulatedPriceVolume(crypto_list, start, end)
    return crypto_data

def get_PriceVolume():
    crypto_data_strat = getPriceVolume(crypto_list, start, end)
    return crypto_data_strat

def final_data_frame():
    crypto_data_strat = get_PriceVolume()
    Binance_Data = pd.read_pickle("Binance_Perp_Fut.pkl")  ### Downloaded from Binance API
    CME_Data = pd.read_csv("CME_BTC_Futures.csv")  ### Downloaded from barchart.com
    # CME_Data = pd.read_csv("CME_BTC_Futures.csv")
    Binance_Data = Binance_Data[['close', 'volume']].rename(columns={'close': 'Last', 'volume': 'Volume'})
    CME_Data = CME_Data.set_index('Time').loc['2017-12-18':'2019-09-08',
               :]  ### Filter for the period where Volume > 0 before Binance product launch
    Futures_Data = pd.concat([CME_Data, Binance_Data], 0).ffill()
    Futures_Data = Futures_Data.rename(columns={'Last': 'Close'})
    index = Futures_Data.index
    index.name = 'Date'
    Futures_Data = Futures_Data.assign(Ticker='BTCFUT').reset_index()
    Futures_Data['Date'] = pd.to_datetime(Futures_Data['Date'])
    placeholder = crypto_data_strat[crypto_data_strat.index.get_level_values(0) == 'BTC-USD'].reset_index()[['Date']]
    Futures_Data = Futures_Data.set_index(['Ticker', 'Date'])
    crypto_data_strat = pd.concat([crypto_data_strat, Futures_Data], 0)
    tedrate = quandl.get('FRED/TEDRATE', start_date=start, end_date=end)
    tres_3m = quandl.get('FRED/DTB3', start_date=start, end_date=end)
    fundingrate = (tedrate + tres_3m - 1).squeeze().dropna()
    crypto_data_strat['Funding'] = pd.Series(crypto_data_strat.index.get_level_values(1)).map(fundingrate).values
    crypto_data_strat[['Funding']] = crypto_data_strat[['Funding']].ffill(limit=2)
    return crypto_data_strat


crypto_list = ['BTC-USD','ADA-USD','ETH-USD','SOL-USD','AVAX-USD','DOT-USD','MATIC-USD','LTC-USD','ATOM-USD','LINK-USD','DOGE-USD','SHIB-USD']
crypto_data_strat = final_data_frame()


def calc_vol(df):
    #df = df.pct_change().rolling(30).std() * 100   ## ~ 70%
    #df = df.pct_change().expanding(30).std() * 100 ## ~ 40%
    df = df.pct_change().ewm(30).std() * 100        ## ~ 68%
    return df

#=======================================================================================================================

                                                #Building the Signals

#=======================================================================================================================

#GARCH/VOl Signals

#-----------------------------------------------------------------------------------------------------------------------

#Place code here

### Returns list of GARCH dfs and lists of GARCH parameters per ticker
def get_garch_val(df = crypto_data_strat):
    ### Calculate the vol of the returns of the df.
    def calc_vol(df):
        # df = df.pct_change().rolling(30).std() * 100   ## ~ 70%
        # df = df.pct_change().expanding(30).std() * 100 ## ~ 40%
        df = df.pct_change().ewm(30).std() * 100  ## ~ 68%
        return df

    historicalvols = df.groupby("Ticker")[['Close']].apply(calc_vol)

    fgarlist, garlist, omegalist, alphalist, betalist = [], [], [], [], []
    for i in tqdm(np.unique(df.index.get_level_values('Ticker'))):
        df_pct = df[df.index.isin([i], level=0)].pct_change().replace([np.inf, -np.inf], np.nan).dropna()['Close'] * 100

        # DOGE, MATIC,SHIB
        if (i == 'DOGE-USD'):
            df_pct = df_pct * 100

        def arch_model_1(df):
            if df.shape[0] < 30:
                return np.nan
            g1_e = arch_model(df, vol='GARCH', mean='AR', p=1, o=0, q=1, dist='Normal')
            model_e = g1_e.fit(disp='off')
            model_forecast = model_e.forecast(horizon=30)
            forecast_df = pd.DataFrame(np.sqrt(model_forecast.variance.dropna().T))
            return forecast_df.loc['h.30'][0]

        df_gar = pd.DataFrame(df_pct[i].expanding().apply(arch_model_1))
        df_gar = df_gar.assign(Ticker=i)
        df_gar = df_gar.rename(columns={'Close': 'FGARCH'})
        df_gar = df_gar.reset_index().set_index(['Ticker', 'Date']).shift()

        if (i == 'DOGE-USD'):
            df_gar = df_gar / 100

        fgarlist.append(df_gar)

    pd_fgarch = pd.concat(fgarlist, 0)
    pd_fgarch = pd.concat([pd_fgarch, historicalvols], 1)

    return pd_fgarch


### Function to compare GARCH vs historical vol and to generate a signal
### This isn't vectorized.
def build_garch_signal(compare_vol):
    ### set an array of bounds on the magnitude of diff in vols
    magnitude = np.array([-2, -1, -0.75, -0.5, -0.1, 0, 0.1, 0.5, 0.75, 1, 2])

    ### set corresponding weights on the signal
    signalsize = np.array([-1, -0.75, -0.65, -0.5, -0.25, 0, 0.25, 0.5, 0.65, 0.75, 1])

    def find_nearest(mag, sigsize, value):
        if pd.isnull(value):
            return 0
        value = value
        idx = np.abs((mag - value)).argmin()
        return sigsize[idx]

    compare_vol = compare_vol.assign(Diff=(compare_vol['FGARCH'] - compare_vol['Close']))
    compare_vol = compare_vol.assign(Signal_1=0)
    for i in tqdm(range(compare_vol.shape[0])):
        compare_vol.iloc[i, 3] = find_nearest(magnitude, signalsize, compare_vol.iloc[i, 2])

    return compare_vol




#-----------------------------------------------------------------------------------------------------------------------


#Reddit/NLP Signals


#-----------------------------------------------------------------------------------------------------------------------

# Place code here
signal2 = pd.read_csv('reddit.csv')

signal2['Date'] = pd.to_datetime(signal2['Date'])
signal2 = signal2.set_index(["Ticker","Date"])


def build_nlp_signal(compare_nlp):
    ### set an array of bounds on the magnitude of diff in vols
    magnitude = np.array([-1, -0.75, -0.5, -0.1, 0, 0.1, 0.5, 0.75, 1])

    ### set corresponding weights on the signal
    signalsize = np.array([-1, -0.75, -0.65, -0.5, -0.25, 0, 0.25, 0.5, 0.65, 0.75, 1])

    def find_nearest(mag, sigsize, value):
        if pd.isnull(value):
            return 0
        value = value
        idx = np.abs((mag - value)).argmin()
        return sigsize[idx]

    compare_nlp = compare_nlp.assign(Signal_2=0)
    for i in tqdm(range(compare_nlp.shape[0])):
        compare_nlp.iloc[i, 1] = find_nearest(magnitude, signalsize, compare_nlp.iloc[i, 0])

    return compare_nlp

# #-----------------------------------------------------------------------------------------------------------------------
#
#
# #=======================================================================================================================
#
#                                                 #Building the signals
#
# #-----------------------------------------------------------------------------------------------------------------------
#
# ### Function to build signals specifically. Pass the Multi-Index (Ticker, Date) dataframe into this function.
#
# ### Arguments: df_data = main crypto_data df. For other dfs, make sure to pass the df with index format ['Ticker','Date']
#
def BuildSignals(df_data = crypto_data_strat):
    test_copy = df_data.copy()

    ### Enter Vol signal logic here
    df_data   = pd.concat([df_data,build_garch_signal(get_garch_val(df_data))['Signal_1']],1)
    ### Enter NLP signal logic here
    df_data   = pd.concat([df_data,build_nlp_signal(signal2)['Signal_2']],1)
    df_btc.index.names = ['Date']
    btc_price_sopr = pd.merge(df_data.loc['BTC-USD'], df_btc, left_index=True, right_index=True)
    btc_price_sopr = btc_price_sopr.loc['2017-12-18':'2022-01-21']
    btc_price_sopr.rename(columns={"Close": "Close_BTC", "Volume": "Volume_BTC", "Market_Cap": "Market_Cap_BTC", "v": "v_BTC"}, inplace=True)

    df_eth.index.names = ['Date']
    eth_price_sopr = pd.merge(df_data.loc['ETH-USD'], df_eth, left_index=True, right_index=True)
    eth_price_sopr = eth_price_sopr.loc['2017-12-18':'2022-01-21']
    eth_price_sopr.rename(columns={"Close": "Close_ETH", "Volume": "Volume_ETH", "Market_Cap": "Market_Cap_ETH", "v":"v_ETH"}, inplace=True)

    ### Enter Crypto signal logic here
    test_copy.index = test_copy.index.set_levels([test_copy.index.levels[0], pd.to_datetime(test_copy.index.levels[1])])
    test_merge = test_copy.merge(btc_price_sopr, left_on="Date", right_index=True, how="left")
    test_merge2 = test_merge.merge(eth_price_sopr, left_on="Date", right_index=True, how="left")
    test_merge2['coin_20dma'] = test_merge2.groupby(['Ticker'])['Close'].rolling(20).mean().reset_index(level=0, drop=True)
    test_merge2['coin_5dma'] = test_merge2.groupby(['Ticker'])['Close'].rolling(5).mean().reset_index(level=0, drop=True)
    test_merge2['trend'] = np.where(test_merge2['coin_5dma'] > test_merge2['coin_20dma'], 'up', 'down')
    test_merge2['BTC_score'] = np.where(test_merge2['coin_5dma'] > test_merge2['coin_20dma'],
                                       np.maximum(-1,np.minimum(0,-(test_merge2['v_BTC'] - 1.01)/0.07)),
                                       np.minimum(1,np.maximum(0,(0.99 - test_merge2['v_BTC'])/0.04)))
    test_merge2['ETH_score'] = np.where(test_merge2['coin_5dma'] > test_merge2['coin_20dma'],
                                       np.maximum(-1,np.minimum(0,-(test_merge2['v_ETH'] - 1.05)/0.1)),
                                       np.minimum(1,np.maximum(0,(0.95 - test_merge2['v_ETH'])/0.1)))
    test_merge2['Signal_SOPR'] = np.where(test_merge2.index.get_level_values(0)=="BTC-USD", test_merge2['BTC_score'],
                                          np.where(test_merge2.index.get_level_values(0)=="ETH-USD", test_merge2['ETH_score'], (test_merge2['BTC_score'] + test_merge2['ETH_score']) / 2))
    final_sopr = test_merge2.drop(columns=['Close_BTC', 'Volume_BTC','Market_Cap_BTC','v_BTC','Close_ETH','Volume_ETH','Market_Cap_ETH','v_ETH','coin_20dma','coin_5dma','trend','BTC_score','ETH_score'])
    df_data   = pd.concat([df_data,final_sopr['Signal_SOPR']],1)

    ### Enter aggregated signal logic here
    df_data   = df_data.assign(Signal_Score = (df_data['Signal_1'].squeeze() + df_data['Signal_2'].squeeze()+ df_data['Signal_SOPR'].squeeze())/3)

    return df_data

#=======================================================================================================================

                                                #Building the strategy

#=======================================================================================================================

#-----------------------------------------------------------------------------------------------------------------------

#Place code here


def rolling_hedge(crypto_data_strat, coins,hedge_instrument):
    """
    The function below get the hedge ratio for a coin
    and the hedging instrument
    Arguments are the coins and the hedge instrument ticker
    """
    all_instruments = [coins] + [hedge_instrument]
    df_list = []
    for instr in all_instruments:
        a = crypto_data_strat.loc[instr][['Close']]
        a.columns = [c+'_'+instr for c in list(a.columns)]
        df_list.append(a)
    final_df = pd.concat(df_list,axis= 1)
    z = final_df.pct_change()
    y = z[z.columns[z.columns.str.contains(coins)]]
    x = sm.add_constant(z[z.columns[z.columns.str.contains(hedge_instrument)]])
    # print(x)
    rols = RollingOLS(y, x, window=30)
    rres = rols.fit()
    params = rres.params.copy()
    params.index = np.arange(1, params.shape[0] + 1)
    p = pd.DataFrame(params['Close_BTCFUT']).set_index(z.index).rename(columns={'Close_BTCFUT':coins})
    return p

def total_hedge(df,crypto_list,hedge_instrument):
    """
    This function will calculate the hedge for 2 coins and give out the total hedge ratio
    Arguments: Pass the first two coins and then the hedging instrument
    """
    df_list = []
    for c in crypto_list:
        c1 = rolling_hedge(df,c,hedge_instrument)
        df_list.append(c1)
    final_df = pd.concat(df_list,axis=1)
    return final_df

def get_total_hedge():
    crypto_data_strat = final_data_frame()
    return total_hedge(crypto_data_strat, crypto_list, 'BTCFUT')


### Arguments: df_data = data with close prices and ONE ratio per day per stock, symbol_list = list of symbols, colname = the name of the column you want to analyze, freq = rebalance frequency, q_l/q_s = quantile of stocks to include, h_l/h_s = hold limit for open positions,
### higher = y/n: If you want to buy at a higher ratio, enter "y", otherwise "n", q_l2/q_s2 = second quantile bound
def QuantileTrade(df_data, trade_amt, T0, T1, symbol_list, crypto_list, colname, freq, q_l, q_s, h_l, h_s, higher="y",
                  q_l2=0, q_s2=0):
    notional_long = trade_amt // 2
    notional_short = trade_amt // 2
    capital = trade_amt
    counter = T1

    df_data = df_data.rename_axis(['ticker', 'trade dates'])
    df_fullset = df_data.copy()
    idx = pd.IndexSlice
    df_data = df_data.loc[idx[:, '2017-12':], :]

    df_data[['Funding']] = df_data[['Funding']].ffill()

    df_data = df_data.unstack('ticker').iloc[::freq, :]
    df_data = df_data.stack('ticker', dropna=False).reset_index().set_index(['ticker', 'trade dates']).sort_index()
    df_DE = df_data[['Close', colname, 'Funding']]
    df_DE[[colname]] = df_DE[[colname]].fillna(0)
    df_DE[['Close']] = df_DE[['Close']].ffill()
    df_DE = df_DE.assign(
        Rank=df_DE[df_DE.index.isin(crypto_list, level=0)][colname].groupby('trade dates').rank(method='first'))

    # Define Buy/Sell depending on whether you want to buy at a higher ratio "y" or lower ratio "n"
    if higher == "n":
        df_DE = df_DE.assign(Signal=np.where(df_DE['Rank'] <= q_l, "Buy",
                                             np.where(df_DE['Rank'] > len(crypto_list) - q_s, "Sell",
                                                      np.where((df_DE['Rank'] <= h_l) | (
                                                                  df_DE['Rank'] > len(crypto_list) - h_s), "Hold",
                                                               "No Action"))))
    elif higher == "y":
        df_DE = df_DE.assign(Signal=np.where(df_DE['Rank'] <= q_l, "Sell",
                                             np.where(df_DE['Rank'] > len(crypto_list) - q_s, "Buy",
                                                      np.where((df_DE['Rank'] <= h_l) | (
                                                                  df_DE['Rank'] > len(crypto_list) - h_s), "Hold",
                                                               "No Action"))))

    df_DE = df_DE.reset_index().set_index('ticker').sort_index()
    df_DE['Signal_LBD'] = df_DE['Signal'].shift()
    df_DE = df_DE.reset_index().set_index(['ticker', 'trade dates']).sort_index()
    df_DE = df_DE.reorder_levels(['trade dates', 'ticker']).sort_index()
    df_DE.loc[(counter, slice(None)), 'Signal_LBD'] = "No Signal"
    df_DE = df_DE.assign(Priority=0, Hold=0, Hold_Count_L=0, Hold_Count_S=0, Position=0, Ntnl_Entry=0, Quantity=0,
                         MV_LBD=0, MV_PBD=0, PL_DLY=0, PTF_Val=0)

    # define different types of positions: 1 = new, 2 = old position in quantile, 3 = old position in hold range
    df_DE['Priority'] = np.where((((df_DE['Signal'] == 'Buy') & (df_DE['Signal_LBD'] == 'Buy')) | (
                (df_DE['Signal'] == 'Sell') & (df_DE['Signal_LBD'] == 'Sell'))), 3,
                                 np.where((((df_DE['Signal'] == 'Hold') & (df_DE['Signal_LBD'] == 'Buy')) | (
                                             (df_DE['Signal'] == 'Hold') & (df_DE['Signal_LBD'] == 'Sell'))), 2,
                                          np.where(((df_DE['Signal'] == 'Buy') & (df_DE['Signal_LBD'] == 'Sell')) | (
                                                      (df_DE['Signal'] == 'Sell') & (df_DE['Signal_LBD'] == 'Buy')), 1,
                                                   np.where(((df_DE['Signal'] == 'Buy') & (
                                                               df_DE['Signal_LBD'] == 'Hold')) | (
                                                                        (df_DE['Signal'] == 'Sell') & (
                                                                            df_DE['Signal_LBD'] == 'Hold')), 1,
                                                            np.where(((df_DE['Signal'] == 'Buy') & (
                                                                        df_DE['Signal_LBD'] == 'No Action') & (
                                                                                  (df_DE['Rank'] <= h_l) | (
                                                                                      df_DE['Rank'] > len(
                                                                                  crypto_list) - h_s))) | (
                                                                                 (df_DE['Signal'] == 'Sell') & (df_DE[
                                                                                                                    'Signal_LBD'] == 'No Action') & (
                                                                                             (df_DE['Rank'] <= h_l) | (
                                                                                                 df_DE['Rank'] > len(
                                                                                             crypto_list) - h_s))), 1,
                                                                     0)))))

    # Start logic to define hold positions, and to count them per time stamp so you know how many new rank positions to exclude
    df_DE.loc[(counter, slice(None)), 'Priority'] = 1
    df_DE['Hold'] = np.where(df_DE['Priority'] == 2, 2, 0)
    holds_long = np.array([])
    holds_short = np.array([])
    store = np.array([])

    buy = df_DE[(df_DE['Priority'] == 1) & ((df_DE['Signal_LBD'] == 'Buy'))].groupby(
        ['trade dates', 'Signal', 'Signal_LBD']).agg('count').reset_index().set_index('trade dates')[['Priority']]
    sell = df_DE[(df_DE['Priority'] == 1) & ((df_DE['Signal_LBD'] == 'Sell'))].groupby(
        ['trade dates', 'Signal', 'Signal_LBD']).agg('count').reset_index().set_index('trade dates')[['Priority']]

    df_DE['Hold_Count_L'] = pd.Series(df_DE.index.get_level_values(0)).map(buy.squeeze()).values
    df_DE['Hold_Count_L'] = df_DE['Hold_Count_L'].fillna(0)
    df_DE['Hold_Count_S'] = pd.Series(df_DE.index.get_level_values(0)).map(sell.squeeze()).values
    df_DE['Hold_Count_S'] = df_DE['Hold_Count_S'].fillna(0)

    # Mark positions that will be held per time stamp. Should equal q_l + q_s at all times, or at least be less than the total for edge cases
    df_DE['Position'] = np.where((df_DE['Priority'] == 2) | ((((df_DE['Rank'] <= q_l - df_DE['Hold_Count_L']) | (
    (df_DE['Rank'] > len(crypto_list) - q_s + df_DE['Hold_Count_S']))) & ((df_DE['Priority'] == 1) | (
                df_DE['Priority'] == 3)))),
                                 np.where(df_DE['Signal'] == 'Buy', 1,
                                          np.where(df_DE['Signal'] == 'Sell', -1,
                                                   np.where((df_DE['Signal'] != 'No Action') & (
                                                               df_DE['Signal_LBD'] == 'Buy'), 1,
                                                            np.where((df_DE['Signal'] != 'No Action') & (
                                                                        df_DE['Signal_LBD'] == 'Sell'), -1, 0)))), 0)

    df_DE['Pos_Prev_MS'] = df_DE['Position'].shift(len(symbol_list))
    df_DE['Px_Prev_MS'] = df_DE['Close'].shift(len(symbol_list))

    # Determine the size of positions with different logic for single quantile and two-tier quantile strategies
    if (q_s2 == 0) and (q_l2 == 0):
        df_DE['Ntnl_Entry'] = np.where(
            ((df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Pos_Prev_MS'] == 1),
            df_DE['Pos_Prev_MS'] * notional_long / q_l,
            np.where(((df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Pos_Prev_MS'] == -1),
                     df_DE['Pos_Prev_MS'] * notional_short / q_s,
                     np.where((df_DE['Priority'] == 1) & (df_DE['Position'] == 1),
                              df_DE['Position'] * notional_long / q_l,
                              np.where((df_DE['Priority'] == 1) & (df_DE['Position'] == -1),
                                       df_DE['Position'] * notional_short / q_s,
                                       np.where((df_DE['Priority'] == 2) & (df_DE['Signal'] == 'Hold') & (
                                                   df_DE['Position'] == -1), df_DE['Position'] * notional_short / q_s,
                                                np.where((df_DE['Priority'] == 2) & (df_DE['Signal'] == 'Hold') & (
                                                            df_DE['Position'] == 1),
                                                         df_DE['Position'] * notional_long / q_l,
                                                         np.where((df_DE['Priority'] == 3) & (
                                                                     (df_DE['Rank'] <= q_l - df_DE['Hold_Count_L']) | ((
                                                                         df_DE['Rank'] > len(crypto_list) - q_s + df_DE[
                                                                     'Hold_Count_S']))),
                                                                  df_DE['Position'] * notional_long / q_l,
                                                                  np.where((df_DE['Priority'] == 0) & ((df_DE[
                                                                                                            'Rank'] <= q_l -
                                                                                                        df_DE[
                                                                                                            'Hold_Count_L']) | (
                                                                                                       (df_DE[
                                                                                                            'Rank'] > len(
                                                                                                           crypto_list) - q_s +
                                                                                                        df_DE[
                                                                                                            'Hold_Count_S']))),
                                                                           df_DE['Position'] * notional_long / q_l,
                                                                           0))))))))

        df_DE['Quantity'] = np.where(
            ((df_DE['Priority'] == 1) | (df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Position'] == 1),
            df_DE['Position'] * notional_long / q_l / df_DE['Close'],
            np.where(((df_DE['Priority'] == 1) | (df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (
                        df_DE['Position'] == -1), df_DE['Position'] * notional_short / q_s / df_DE['Close'],
                     np.where(((df_DE['Priority'] == 2)) & ((df_DE['Pos_Prev_MS'] == 0) & (df_DE['Position'] == -1)),
                              df_DE['Position'] * notional_short / q_s / df_DE['Close'],
                              np.where(
                                  ((df_DE['Priority'] == 2)) & ((df_DE['Pos_Prev_MS'] == 0) & (df_DE['Position'] == 1)),
                                  df_DE['Position'] * notional_long / q_l / df_DE['Close'], 0))))

        df_DE['Quantity_PBD'] = df_DE[['Quantity']].shift(len(symbol_list))
        df_DE['Quantity'] = np.where(
            ((df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Quantity_PBD'] != 0), df_DE['Quantity_PBD'],
            df_DE['Quantity'])
        df_DE['Quantity_PBD'] = df_DE[['Quantity']].shift(len(symbol_list))
        df_DE['Ntnl_Entry_PBD'] = df_DE['Ntnl_Entry'].shift(len(symbol_list))

    else:
        df_DE['Ntnl_Entry'] = np.where(
            ((df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Pos_Prev_MS'] == 1),
            df_DE['Pos_Prev_MS'] * notional_long / q_l,
            np.where(((df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Pos_Prev_MS'] == -1),
                     df_DE['Pos_Prev_MS'] * notional_short / q_s,
                     np.where((df_DE['Priority'] == 1) & (df_DE['Position'] == 1),
                              df_DE['Position'] * notional_long / q_l,
                              np.where((df_DE['Priority'] == 1) & (df_DE['Position'] == -1),
                                       df_DE['Position'] * notional_short / q_s,
                                       np.where((df_DE['Priority'] == 2) & (df_DE['Signal'] == 'Hold') & (
                                                   df_DE['Position'] == -1), df_DE['Position'] * notional_short / q_s,
                                                np.where((df_DE['Priority'] == 2) & (df_DE['Signal'] == 'Hold') & (
                                                            df_DE['Position'] == 1),
                                                         df_DE['Position'] * notional_long / q_l,
                                                         np.where((df_DE['Priority'] == 3) & (
                                                                     (df_DE['Rank'] <= q_l - df_DE['Hold_Count_L']) | ((
                                                                         df_DE['Rank'] > len(crypto_list) - q_s + df_DE[
                                                                     'Hold_Count_S']))),
                                                                  df_DE['Position'] * notional_long / q_l,
                                                                  np.where((df_DE['Priority'] == 0) & ((df_DE[
                                                                                                            'Rank'] <= q_l -
                                                                                                        df_DE[
                                                                                                            'Hold_Count_L']) | (
                                                                                                       (df_DE[
                                                                                                            'Rank'] > len(
                                                                                                           crypto_list) - q_s +
                                                                                                        df_DE[
                                                                                                            'Hold_Count_S']))),
                                                                           df_DE['Position'] * notional_long / q_l,
                                                                           0))))))))

        df_DE['Ntnl_Entry'] = np.where((df_DE['Rank'] <= q_l2) | (df_DE['Rank'] > len(crypto_list) - q_s2),
                                       df_DE['Ntnl_Entry'] * 3, df_DE['Ntnl_Entry'] / 3)

        df_DE['Quantity'] = np.where(
            ((df_DE['Priority'] == 1) | (df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (df_DE['Position'] == 1),
            df_DE['Position'] * notional_long / q_l / df_DE['Close'],
            np.where(((df_DE['Priority'] == 1) | (df_DE['Priority'] == 2) | (df_DE['Priority'] == 3)) & (
                        df_DE['Position'] == -1), df_DE['Position'] * notional_short / q_s / df_DE['Close'],
                     np.where(((df_DE['Priority'] == 2)) & ((df_DE['Pos_Prev_MS'] == 0) & (df_DE['Position'] == -1)),
                              df_DE['Position'] * notional_short / q_s / df_DE['Close'],
                              np.where(
                                  ((df_DE['Priority'] == 2)) & ((df_DE['Pos_Prev_MS'] == 0) & (df_DE['Position'] == 1)),
                                  df_DE['Position'] * notional_long / q_l / df_DE['Close'], 0))))

        df_DE['Quantity'] = np.where((df_DE['Rank'] <= q_l2) | (df_DE['Rank'] > len(crypto_list) - q_s2),
                                     df_DE['Quantity'] * 3, df_DE['Quantity'] / 3)

        df_DE['Quantity_PBD'] = df_DE['Quantity'].shift(len(symbol_list))
        df_DE['Ntnl_Entry_PBD'] = df_DE['Ntnl_Entry'].shift(len(symbol_list))

    x = total_hedge(crypto_data_strat, crypto_list, 'BTCFUT')
    x = x.stack().reset_index().set_index(['Date', 'level_1']).rename_axis(['trade dates', 'ticker']).rename(
        columns={0: 'Hedge Ratio'})
    df_DE = pd.concat([df_DE, x], axis=1)
    df_DE['Hedge Ratio'] = np.where(np.isnan(df_DE['Hedge Ratio']), 1, df_DE['Hedge Ratio'])
    df_DE['Hedge Ratio'] = np.where(df_DE['Hedge Ratio'] > 10, 10, df_DE['Hedge Ratio'])
    df_DE = df_DE.assign(Future_Flag=0)
    df_DE.loc[df_DE['Signal'] == 'Sell', 'Future_Flag'] = -1
    df_DE = df_DE.assign(BTCFUT_Q_PER=df_DE['Future_Flag'].squeeze() * df_DE['Hedge Ratio'].squeeze())

    BTCFUT_Q = df_DE.groupby('trade dates').agg('sum')[['BTCFUT_Q_PER']]
    BTCFUT_Q = BTCFUT_Q.assign(ticker='BTCFUT')
    BTCFUT_Q = BTCFUT_Q.reset_index().set_index(['trade dates', 'ticker']).rename(columns={'BTCFUT_Q_PER': 'BTCFUT_Q'})
    df_DE = pd.concat([df_DE, BTCFUT_Q], 1)

    df_DE = df_DE.assign(BTCFUT_Q_PBD=0)
    df_DE['BTCFUT_Q_PBD'] = df_DE['BTCFUT_Q'].shift(len(symbol_list))
    df_DE['Quantity'] = np.where(np.isnan(df_DE['BTCFUT_Q']) == False,
                                 notional_short / q_s * df_DE['BTCFUT_Q'] / df_DE['Close'],
                                 np.where(df_DE['Signal'] == 'Sell', 0, df_DE['Quantity']))
    df_DE['Quantity_PBD'] = np.where(np.isnan(df_DE['BTCFUT_Q_PBD']) == False,
                                     df_DE['Quantity'].shift(len(symbol_list)),
                                     np.where(df_DE['Signal'] == 'Sell', 0, df_DE['Quantity_PBD']))
    df_DE['Quantity'] = np.where((df_DE['Quantity'] == -1), 0, df_DE['Quantity'])
    df_DE['Quantity_PBD'] = np.where((df_DE['Pos_Prev_MS'] == -1), 0, df_DE['Quantity_PBD'])

    LBD_P = np.array([])
    PBD_P = np.array([])
    LBD_Q = np.array([])
    PBD_Q = np.array([])

    LBD_P = np.append(LBD_P, df_DE['Close'])
    PBD_P = np.append(PBD_P, df_DE['Px_Prev_MS'])
    LBD_Q = np.append(LBD_Q, df_DE['Quantity'])
    PBD_Q = np.append(PBD_Q, df_DE['Quantity_PBD'])
    LBD_MV = np.multiply(LBD_P, PBD_Q)
    PBD_MV = np.multiply(PBD_P, PBD_Q)

    DLY_PL = np.subtract(LBD_MV, PBD_MV)

    df_DE['MV_LBD'] = LBD_MV.tolist()
    df_DE['MV_PBD'] = PBD_MV.tolist()
    df_DE['PL_DLY'] = DLY_PL.tolist()

    ### Include TED Spread on leveraged position of $1m, AND include Funding rate on BTCFUT position, where the max rate observed in market was 3bps DLY funding on LONG positions.
    DLY_PL_Net = np.add(DLY_PL,
                        np.where(df_DE['MV_LBD'] > 0, np.multiply(df_DE['Funding'] / 100 * freq / 365, trade_amt / 2),
                                 0))
    df_DE['PL_DLY_Net'] = DLY_PL_Net.tolist()

    ### Include Fixed Binance Funding costs of 3bps daily in the Net Returns (Assume negative for conservative estimate)
    df_DE['PL_DLY_Net'] = np.where((df_DE.index.get_level_values(1) == 'BTCFUT'),
                                   df_DE['PL_DLY_Net'] + (df_DE['MV_LBD'] * 0.0003), df_DE['PL_DLY_Net'])

    ### Include B/O cost on Binance for spot trades on all trades (10bps - 1bp from friend referral discount)
    df_DE['PL_DLY_Net'] = np.where(
        ((df_DE['MV_LBD'] > 0) & (df_DE['MV_PBD'] == 0) & (df_DE['Quantity'] != df_DE['Quantity_PBD'])) | (
                    (df_DE['MV_LBD'] == 0) & (df_DE['MV_PBD'] > 0) & (df_DE['Quantity'] != df_DE['Quantity_PBD'])),
        df_DE['PL_DLY_Net'] + (abs(df_DE['Quantity'] - df_DE['Quantity_PBD']) * df_DE['Close'] * -0.0009),
        df_DE['PL_DLY_Net'])

    ### Include B/O cost on Binance for futures trades on all trades (4bps - 0.4bp from friend referral discount)
    df_DE['PL_DLY_Net'] = np.where(df_DE['BTCFUT_Q'] < 0, df_DE['PL_DLY_Net'] + (
                abs(df_DE['BTCFUT_Q'] - df_DE['BTCFUT_Q_PBD']) * df_DE['Close'] * -0.00036), df_DE['PL_DLY_Net'])

    df_DE['PL_LTD'] = df_DE['PL_DLY'].cumsum()
    df_DE['PL_LTD_Net'] = df_DE['PL_DLY_Net'].cumsum()
    df_DE['PTF_Val'] = df_DE['PL_LTD'] + capital

    ### Include the Macro Hedge

    df_Macro_Hedge = df_DE[['Close', 'PTF_Val']]
    df_Macro_Hedge = df_Macro_Hedge[df_Macro_Hedge.index.get_level_values(1) == 'VIX']
    df_Macro_Hedge = df_Macro_Hedge.assign(PTF_Return=df_Macro_Hedge['PTF_Val'].pct_change(),
                                           VIX_Return=df_Macro_Hedge['Close'].pct_change())
    y = df_Macro_Hedge['PTF_Return']
    x = sm.add_constant(df_Macro_Hedge['VIX_Return'])
    rols = RollingOLS(y, x, window=30)
    rres = rols.fit()
    params = rres.params.copy()
    params.index = np.arange(1, params.shape[0] + 1)
    p = pd.DataFrame(params['VIX_Return']).set_index(df_Macro_Hedge.index)  # .rename(columns={'Close_BTCFUT':coins})
    p = p.iloc[::30]  ### re-hedge every 30 days

    df_DE = df_DE.assign(Macro_Hedge_MV=(p))
    df_DE['Macro_Hedge_MV'] = df_DE['Macro_Hedge_MV'] * df_DE['PTF_Val'].squeeze()  # ).shift(-1).ffill())
    df_DE['Macro_Hedge_MV'] = df_DE['Macro_Hedge_MV'].shift(-1)
    df_DE = df_DE.assign(Macro_Hedge_Px=np.where(
        (df_DE.index.get_level_values(1) == 'SVXY') & (pd.isnull(df_DE['Macro_Hedge_MV']) == False), df_DE['Close'],
        np.nan))
    df_DE['Macro_Hedge_Px'] = df_DE['Macro_Hedge_Px'].bfill()
    df_DE = df_DE.assign(Macro_Hedge_Px_Nxt=df_DE['Macro_Hedge_Px'].shift(-1))
    df_DE = df_DE.assign(Macro_Hedge_PL=df_DE['Macro_Hedge_MV'].squeeze() * (
                (df_DE['Macro_Hedge_Px_Nxt'].squeeze() / df_DE['Macro_Hedge_Px'].squeeze()) - 1))
    df_DE['Macro_Hedge_PL'] = np.where(pd.isnull(df_DE['Macro_Hedge_PL']), 0, df_DE['Macro_Hedge_PL'])
    df_DE = df_DE.assign(PL_LTD_wo_Macro = df_DE['PL_LTD'])
    df_DE['PL_DLY'] += df_DE['Macro_Hedge_PL'].squeeze()
    df_DE['PL_DLY_Net'] += df_DE['Macro_Hedge_PL'].squeeze()
    df_DE['PL_LTD'] = df_DE['PL_DLY'].cumsum()
    df_DE['PL_LTD_Net'] = df_DE['PL_DLY_Net'].cumsum()

    ### control check to see if backtest bankrupts
    if df_DE[df_DE['PTF_Val'] < 0]['PTF_Val'].count() > 0:
        return "WARNING!: This strategy depletes capital!"

    print('LTD PL (Gross) for this strategy is {}'.format(df_DE.iloc[-1, -7]))
    print('LTD PL (Net) for this strategy is {}'.format(df_DE.iloc[-1, -6]))

    return df_DE

#-----------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================

                                                #Data Plots

#=======================================================================================================================

#Place code here
crypto_data = get_TabulatedPriceVolumeData()
close_data = crypto_data.filter(regex='Close')
def close_data_plot(crypto_data = crypto_data):
    close_data = crypto_data.filter(regex='Close')
    close_data.plot(kind='line', subplots=True, grid=True, title="Closing Prices for SPX and various cryptocurrencies",
        layout=(5, 3), sharey=False, legend=False,
        style=['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'r', 'r', 'r'],
        )
    plt.tight_layout()
    for ax in plt.gcf().axes:
        ax.legend(loc=0)

def volume_data_plot(crypto_data =crypto_data):
    close_data = crypto_data.filter(regex='Volume')
    close_data.plot(kind='line', subplots=True, grid=True, title="Closing Prices for SPX and various cryptocurrencies",
                    layout=(5, 3), sharey=False, legend=False,
                    style=['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'r', 'r', 'r'],
                    )
    plt.tight_layout()
    for ax in plt.gcf().axes:
        ax.legend(loc=0)

def close_excl_vix(close_data =close_data):
    close_excl_vix = close_data[close_data.columns.difference(['VIX_Close'])]
    close_excl_vix.pct_change()
    # volume_data = crypto_data.filter(regex='Volume')
    close_excl_vix.pct_change().plot(kind='line', subplots=True, grid=True,
                                     title="Daily Returns for SPX and various cryptocurrencies",
                                     layout=(5, 3), legend=False,
                                     style=['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'r', 'r', 'r'],
                                     )
    plt.tight_layout()
    for ax in plt.gcf().axes:
        ax.legend(loc=0)

def corr_heatmap(close_data = close_data):
    # close_data
    cor_mat = close_data.pct_change().corr()
    sns.set(font_scale=1)
    plt.figure(figsize=(20, 10))
    hm = sns.heatmap(cor_mat,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.4f',
                     annot_kws={'size': 9},
                     yticklabels=cor_mat.columns,
                     xticklabels=cor_mat.columns, cmap="RdBu")
    plt.title('Correlation matrix heatmap')

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------


#=======================================================================================================================

                                            #Portfolio Strategy Analysis

#=======================================================================================================================


def returns(df =crypto_data_strat ,ticker_list = crypto_list):
    df_data= df.unstack('Ticker')
    df_data= df_data.stack('Ticker',dropna=False).reset_index().set_index(['Ticker','Date']).sort_index()
    d = df_data.loc[pd.IndexSlice[ticker_list,:],['Close']]
    d = d.reset_index()
    a = pd.pivot_table(d,index= 'Date', columns = 'Ticker')
    a.ffill() #double check with richmond
    ret = a.pct_change().replace([np.inf,-np.inf], np.nan)
    return ret


def benchmark_portfolios(TradingPL) :

    def risk_contribution(w, cov):
        """
        Compute the contributions to risk of the constituents of a portfolio,
        given a set of portfolio weights and a covariance matrix
        """
        total_portfolio_var = portfolio_volatility(w, cov) ** 2
        # Marginal contribution of each constituent
        marginal_contrib = cov @ w
        risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
        return risk_contrib


    def portfolio_volatility(weights, covariance_matrix):
        """
        Inputs: Weights and Expected returns
        Calculates the portfolio return
        Parameters : Weights, covariance matrix
        Outputs: Portfolio Volatility
        """
        return np.sqrt(multi_dot([weights.T, covariance_matrix, weights]))


    def target_risk_contributions(target_risk, cov, total_weight=1):
        """
        Returns the weights of the portfolio that gives you the weights such
        that the contributions to portfolio risk are as close as possible to
        the target_risk, given the covariance matrix, total portfolio weight can be changed
        """
        n = cov.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - total_weight
                            }

        def msd_risk(weights, target_risk, cov):
            """
            Returns the Mean Squared Difference in risk contributions
            between weights and target_risk
            """
            w_contribs = risk_contribution(weights, cov)
            return ((w_contribs - target_risk) ** 2).sum()

        weights = minimize(msd_risk, init_guess,
                           args=(target_risk, cov), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    def equal_risk_contributions(cov, total_weights=1):
        """
        Returns the weights of the portfolio that equalizes the contributions
        of the constituents based on the given covariance matrix
        """
        n = cov.shape[0]
        return target_risk_contributions(target_risk=np.repeat(1 / n, n), cov=cov, total_weight=total_weights)


    def weights_df(stocks, er, cov, riskfree_rate=0, total_weights=1):
        """
        Returns the weights in a dataframe for Max Sharpe Ratio, Equally weighted,
        Global Minimum Variance and min VaR portfolios
        """
        n = er.shape[1]
        w_ew = np.repeat(1 / n, n)

        w_equal_risk = equal_risk_contributions(cov, total_weights=total_weights)
        weights_df = pd.DataFrame({'Ticker': stocks,
                                   'weights_ew': w_ew,
                                   'equal_risk': w_equal_risk

                                   })

        return weights_df

    def risk_contributions(covariance,weights,stock_list):
        """
        weights must be a column vector
        """
        weights = np.matrix(weights).T
        covariance = np.matrix(covariance)
        portfolio_variance = weights.T@covariance@weights
        marginal_vol = ((1/portfolio_variance[0,0]**0.5)*covariance*weights)
        MCTR = np.multiply(weights,marginal_vol)
        arr = np.hstack((weights, marginal_vol,MCTR))
        risk_decomp = pd.DataFrame(arr,columns = ['weights','MR','RC'],index=stock_list)
        risk_decomp['% contribution'] = risk_decomp['RC']/risk_decomp['RC'].sum()
       # print(np.where(np.sum(MCTR) == portfolio_variance[0,0]**0.5,'True','False'))
        return risk_decomp

    ret = returns(crypto_data_strat,crypto_list)

    weight_df = weights_df(crypto_list, ret, ret.cov())

    n = len(crypto_list)
    w_ew = np.repeat(1 / n, n)

    ret_w = pd.DataFrame(ret @ np.array(weight_df['weights_ew'])).rename(columns={0: 'Equally_Weighted_Return'})

    ret_rc = pd.DataFrame(ret @ np.array(weight_df['equal_risk'])).rename(columns={0: 'Equal_Risk_Contribution'})

    bitw_data = yf.download(['BITW', 'BTC-USD'])['Close']

    bitw_ret = bitw_data.pct_change().rename(columns={'BITW': 'BITW_return', 'BTC-USD': 'BTC_return'})

    idx = pd.IndexSlice
    port_ret = TradingPL.loc[idx[:, 'SVXY'], 'PTF_Val'].pct_change()
    port_ret = pd.DataFrame(port_ret).reset_index().set_index('trade dates').drop(columns='ticker')
    benchmark_data = pd.concat([bitw_ret, ret_rc, ret_w, port_ret], axis=1)

    return benchmark_data


def performanceMetrics(returns, annualization=1):
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Mean'] = returns.mean() * annualization
    metrics['Vol'] = returns.std() * np.sqrt(annualization)
    metrics['Sharpe'] = (returns.mean() / returns.std()) * np.sqrt(annualization)
    metrics['VaR (0.05)'] = returns.quantile(0.05)
    metrics['Min'] = returns.min()
    metrics['Max'] = returns.max()
    metrics['Expected_Shortfall (5%)'] = returns[returns < returns.quantile(0.05)].mean()
    metrics['Skewness'] = returns.skew()
    metrics['Excess Kurtosis'] = returns.kurt()

    return pd.concat([metrics], axis=1)


def plot_drawdown(TradingPL,benchmark):
    """
    :param: TradingPL
    Returns: Plot of the drawdowns
    """
    idx = pd.IndexSlice
    cumulative_PnL_LTD_Gross = pd.DataFrame(TradingPL.loc[idx[:, 'SVXY'], 'PL_LTD_wo_Macro']).reset_index().set_index('trade dates').drop(columns='ticker')
    cumulative_PnL_LTD = pd.DataFrame(TradingPL.loc[idx[:, 'SVXY'], 'PL_LTD']).reset_index().set_index('trade dates').drop(columns='ticker')
    cumulative_PnL_Macro = pd.DataFrame(TradingPL.loc[idx[:, 'SVXY'], 'Macro_Hedge_PL']).reset_index().set_index('trade dates').drop(columns='ticker').cumsum()
    # cumulative_PnL = pd.DataFrame(cumulative_PnL).reset_index().set_index('trade dates').drop(columns='ticker')
    #benchmark_return = (benchmark[['BITW_return','BTC_return']] + 1).cumprod()
    cumulative_PnL = pd.concat([cumulative_PnL_LTD,cumulative_PnL_Macro,cumulative_PnL_LTD_Gross],axis =1)
    peaks = cumulative_PnL.cummax()
    # calculate the drawdown in percentage
    drawdown = (cumulative_PnL - peaks) / 10000000
    drawdown.plot.line()
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Percentage drawdown', fontsize=10)
    plt.title('Drawdown Levels', fontsize=10)
    plt.gcf().set_size_inches(10, 5)
    plt.show()

def FF_factors(TradingPL):

    """
    :param TradingPL:
    :return: Regression summary for using th Fama-French factors
    """
    idx = pd.IndexSlice
    port_ret = TradingPL.loc[idx[:, 'SVXY'], 'PTF_Val'].pct_change()
    port_ret = pd.DataFrame(port_ret).reset_index().set_index('trade dates').drop(columns='ticker')
    FF_factors = pd.read_csv(
        r"D:\Documents\Education\University of Chicago\Q2\Regression Analysis\HW\HW2_Fama_French_Factors.csv")
    FF_factors['Date'] = pd.to_datetime(FF_factors['Date'], format='%Y%m%d', errors='ignore')
    FF_factors.set_index('Date', inplace=True)
    ff_return_data = pd.concat([pd.DataFrame(port_ret).reset_index().set_index('trade dates'), FF_factors], axis=1,
                               join='inner')
    # ff_return_data.head()
    X = sm.add_constant(ff_return_data[['Mkt-RF', 'SMB', 'HML']])
    reg = sm.OLS(ff_return_data['PTF_Val'], X, missing='drop').fit()
    return reg.summary()

def information_ratio(benchmark, TradingPL):
    """
    Returns the dataframe containing the information ratio versus
    a particular benchmark
    :param: benchmark

    """
    returns_col = ['Equal_Risk_Contribution', 'Equally_Weighted_Return', 'PTF_Val']
    benchmark_data = benchmark_portfolios(TradingPL)
    coef_IR = pd.DataFrame({})
    print("Using " + benchmark +" as the benchmark")
    for ret in returns_col:
        X = sm.add_constant(benchmark_data[benchmark])
        reg = sm.OLS(benchmark_data[ret], X, missing='drop').fit()
        coef_IR.loc[ret, 'Alpha'] = reg.params[0]
        coef_IR.loc[ret, 'Beta'] = reg.params[1]
        coef_IR.loc[ret, 'R_Squared'] = reg.rsquared
        coef_IR.loc[ret, 'Information Ratio'] = reg.params[0] / (reg.resid.std())
    return coef_IR


def downside_beta(benchmark,benchmark_data):
    """
    Return the downside beta statistics versus the appropriate benchmarks

    :param benchmark
    :return:  downside beta statistics
    """

    returns_col = ['PTF_Val', 'Equally_Weighted_Return', 'Equal_Risk_Contribution']
    coeff = pd.DataFrame({})
    print("Using " + benchmark + " as the benchmark")
    for ret in returns_col:
        X = sm.add_constant(benchmark_data[benchmark_data[benchmark] < 0][benchmark])

        reg = sm.OLS(benchmark_data[benchmark_data[benchmark] < 0][ret], X, missing='drop').fit()
        coeff.loc[ret, 'Downside Beta'] = reg.params[1]
        coeff.loc[ret, 'p-value'] = reg.pvalues[1]
    return coeff

def cointegration_test(hedging_instrument):

    """:param: Multi index dataframe

    Returns the cointegration stats for all the coins versus the hedging instrument
    BTCFUT

    """
    crypto_data_strat = final_data_frame()
    ret_BTC_FUT = returns(crypto_data_strat,'BTCFUT')
    crypto_list = ['BTC-USD', 'ADA-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD',
                   'ATOM-USD', 'LINK-USD', 'DOGE-USD', 'SHIB-USD']
    def cointegration_stats(x, y, threshold=0.05):
        result = ts.coint(x, y)
        s = pd.DataFrame()
        s.loc['Engle_Granger Stat', 'Coefficient'] = result[0]
        s.loc['P-value', 'Coefficient'] = result[1]
        if result[1] < threshold:
            s.loc['Result', 'Coefficient'] = 'Cointegrated'
        else:
            s.loc['Result', 'Coefficient'] = 'Not Cointegrated'
        return s

    # ret = returns(crypto_data_strat, crypto_list)
    df_list = []
    for c in crypto_list:
        #print(c)
        data_coint = pd.concat(
            [crypto_data_strat.loc[c][['Close']].rename(columns={'Close': c}).pct_change(), ret_BTC_FUT['Close']],
            axis=1, join='inner').dropna()
        y = cointegration_stats(data_coint[c], data_coint[hedging_instrument]).rename(columns={'Coefficient': c})
        df_list.append(y)
        coint_df = pd.concat(df_list, axis=1)
    return coint_df

def PL_Plot(PL_Plot):
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.suptitle('Strategy LTD Returns, with Net PL in Red and Costs in the Highlighted Region')
    ax1.plot(PL_Plot[['PL_LTD']], 'g', linewidth=0.75)
    ax1.plot(PL_Plot[['PL_LTD_Net']], 'r', linewidth=0.5)
    ax1.legend(['PL_LTD', 'PL_LTD_Net'], loc='upper left')
    plt.fill_between(PL_Plot[['PL_LTD']].index, PL_Plot['PL_LTD'], PL_Plot['PL_LTD_Net'], color='slategray',
                     alpha=0.2)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('LTD PL in USD ($)')
    plt.show()