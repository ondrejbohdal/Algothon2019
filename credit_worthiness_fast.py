import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

credit_data = pd.read_csv('./datasets/DNB_SIRA.csv', low_memory=False)
credit_data['datepll'] = pd.to_datetime(credit_data['datepll'], format='%Y%m')
credit_data = credit_data.set_index(['datepll']).sort_index().reset_index()
credit_data = credit_data.set_index(['datepll','ticker'])

credit_data = credit_data.drop(columns=['businessname', 'isin', 'cusip6'])

credit_data_f_company = credit_data[credit_data['basecat'] == 'R'] # only keep data about full company
prop_missing = credit_data_f_company.isna().sum(axis=0) / credit_data_f_company.shape[0]

columns_keep = prop_missing[prop_missing < 0.1].index.tolist() # Threshold
credit_data_low_missing = credit_data_f_company[columns_keep]

prop_missing_low = credit_data_low_missing.isna().sum(axis=0) / credit_data_low_missing.shape[0]

low_missing_cols = prop_missing_low[prop_missing_low > 0].index.tolist()

final_cols = credit_data_low_missing[low_missing_cols].dtypes[~(credit_data_low_missing[low_missing_cols].dtypes == 'object')].index.tolist()

credit_data_low_missing = credit_data_low_missing[final_cols]

# Use mean to fill missing values
credit_data_clean = credit_data_low_missing.fillna(credit_data_low_missing.mean())

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
print('Running')
# Apply the KPCA transformation to the ADL dataset = GT variables and their one-month lags
df = credit_data_clean

df_google_only = df.copy()

# Standard Scale the data
df_google_only_st = StandardScaler().fit_transform(df_google_only)

# transformer = PCA()
transformer = KernelPCA(kernel='poly', gamma=10, n_components=10)
transformer = transformer.fit(df_google_only_st)

train_img = transformer.transform(df_google_only_st)

train_img = pd.DataFrame(data = train_img)

# gather data into a csv
df_after_pca = train_img.copy()

df_after_pca.to_csv('credit_worthiness_after_pca_fast.csv')