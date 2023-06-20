# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 18:20:38 2023

@author: vinig
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
import yfinance as yf

# Atualizar os tickers do DWJ e IBOV

tickers = ['^BVSP','ABEV3.SA','B3SA3.SA','BBSE3.SA','BRML3.SA','BBDC3.SA','BBDC4.SA','BRAP4.SA','BBAS3.SA','BRKM3.SA','BRFS3.SA','CCRO3.SA','CMIG4.SA','CIEL3.SA','CPLE6.SA','CSAN3.SA','CPFE3.SA','CYRE3.SA','ECOR3.SA','EMBR3.SA', 'ENBR3.SA', 'EQTL3.SA', 'YDUQ3.SA','FIBR3.SA','GGBR4.SA','GOAU4.SA','HYPE3.SA','ITSA4.SA','ITUB4.SA','JBSS3.SA','KLBN11.SA','COGN3.SA','RENT3.SA','AMER3.SA','LREN3.SA','MRFG3.SA', 'MRVE3.SA','MULT3.SA','NTCO3.SA','PCAR3.SA','PETR3.SA','PETR4.SA','QUAL3.SA','RADL3.SA','RAIL3.SA','SBSP3.SA','SANB11.SA','CSNA3.SA','GOLL4.SA','SUZB3.SA','VIVT3.SA','TIMS3.SA','EGIE3.SA','UGPA3.SA','USIM5.SA','VALE3.SA','VALE5.SA','WEGE3.SA']
#tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DD','XOM','GE','GS','HD','INTC','IBM','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','TRV','UNH','VZ','V','WMT','DIS']


df = pd.DataFrame()

for company in tickers :
    
    hist = yf.Ticker(company)
    hist = hist.history(start='1994-07-05', end='2022-12-16',interval='1d')
    hist = hist['Close']
    df[company]=hist


retorno_diario = df.pct_change()

retorno_diario.head()
retorno_diario = retorno_diario.iloc[1:]
retorno_diario.head()

retorno_anual = retorno_diario.mean()*250

cov_diario = retorno_diario.cov()

cov_diario
cov_anual = cov_diario*250

port_returns = []

port_volatility = []

port_sharpe = []

stock_weights = []

num_assets = len(tickers)

num_portfolios = 30000


peso = np.random.random(num_assets)
peso /= np.sum(peso)
peso
np.sum(peso)

for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, retorno_anual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_anual, weights)))
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)
    sharpe = port_returns[single_portfolio]/port_volatility[single_portfolio]
    port_sharpe.append(sharpe)
    portfolio = {'Retornos': port_returns, 'Volatilidade': port_volatility, 'Sharpe': port_sharpe}

for counter,symbol in enumerate(tickers):
    portfolio[symbol+' peso'] = [weight[counter] for weight in stock_weights]
    df = pd.DataFrame(portfolio)
    df.head()
    retornos = df.sort_values(by = ['Retornos'], ascending = False)
    retornos.head()
    plt.style.use('seaborn')

df.plot.scatter(x = 'Volatilidade', y = 'Retornos', figsize = (10,10), grid = True)

plt.xlabel('Volatilidade')

plt.ylabel('Retornos Esperados')

plt.title('Fronteira Eficiente')

plt.show()


a = df[np.argmax(port_sharpe):np.argmax(port_sharpe)+1]
print ('A carteira com maior Sharpe é:', a.T)

