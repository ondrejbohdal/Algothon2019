import quandl
import pandas as pd
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import statsmodels.api as sm
import statsmodels
from matplotlib import pyplot
import sys

df = pd.read_csv('price_data/TTC.csv')

y = df['Adj_Close']

class TimeSeries(object):

    def __init__(self, y):
        self.y = y
    
    def determineLagLength(self, diff_counter):
        arima_model = statsmodels.tsa.arima_model.ARIMA(endog=self.y, order=(1, diff_counter, 0))
        print(dir(arima_model))

    def performDF(self, y, diff_counter):
        df_result = sm.tsa.stattools.adfuller(y, maxlag=None, autolag='AIC', regression='ct')
        mc_p = df_result[1]
        print(mc_p)
        if mc_p > 0.05:
            print("more")
            y_diff =  y.diff()
            y_diff = y_diff.dropna()
            diff_counter = diff_counter + 1
            self.performDF(y_diff, diff_counter)
        else:
            print("less")
            print(y.head())
            print(diff_counter)
        return diff_counter

time_series = TimeSeries(y)
diff_counter = time_series.performDF(time_series.y, 0)
time_series.determineLagLength(diff_counter)
# print(diff_counter)