import quandl
import pandas as pd
import os

# quandl.ApiConfig.api_key = 'REMOVED'

current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder

american_stocks_file_location = os.path.join(current_file, '../algothon102019/american_stocks.csv')
eod_metadata_file_location = os.path.join(current_file, '../algothon102019/EOD_metadata.csv')

american_stocks_df = pd.read_csv(american_stocks_file_location, header=None)
eod_metadata_df = pd.read_csv(eod_metadata_file_location, header=None)

eod_metadata_df_curr = eod_metadata_df.copy()
eod_metadata_df_curr = eod_metadata_df_curr[[0]]
eod_metadata_series = eod_metadata_df_curr.ix[:,0].tolist()

correct_tickers = []
wrong_tickers = []

for index, row in american_stocks_df.iterrows():
    if row[1] in eod_metadata_series:
        correct_tickers.append(row[1])
    else:
        wrong_tickers.append(row[1])

print(len(correct_tickers))
print(len(wrong_tickers))

correct_tickers_df = pd.DataFrame(correct_tickers, columns=["ticker"])
wrong_tickers_df = pd.DataFrame(wrong_tickers, columns=["ticker"])

correct_tickers_df.to_csv('correct_tickers.csv', index=False)
wrong_tickers_df.to_csv('wrong_tickers.csv', index=False)
