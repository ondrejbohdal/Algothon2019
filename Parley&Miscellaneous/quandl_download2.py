import quandl
import pandas as pd

correct_tickers = pd.read_csv("correct_tickers.csv", header=0)

quandl.ApiConfig.api_key = 'REMOVED'
quandl_datasets = []

for index, row in correct_tickers.iterrows():
    ticker_string = 'EOD/' + row[0]
    print(index)
    print(row[0])
    quandl_series = quandl.get(ticker_string, start_date='2000-01-01', end_date='2019-10-01')
    print(type(quandl_series))
    print(quandl_series.head(2))
    quandl_series.to_csv('quandl_download_py2/'+row[0] + '.csv', index=True)
    quandl_datasets.append(quandl_series)


