import random
import numpy as np
import torch
import pandas as pd
# multivariate data preparation
from numpy import array
from numpy import hstack

best_ticker = ''
best_ticker_size = 0

# import os
# current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder

# # correct_tickers_df = pd.read_csv(os.path.join(current_file, 'correct_tickers.csv'), header=0)
# # correct_tickers_list = correct_tickers_df['ticker'].to_list()
# # print(correct_tickers_list)

# # for ticker in correct_tickers_list:
# #     try:
# #         print("No error")
# #         df = pd.read_csv('combined_ticker_vectors/%s.csv'%ticker)
# #         number_of_rows = df.shape[0]
# #         if best_ticker_size < number_of_rows:
# #             best_ticker_size = number_of_rows
# #             best_ticker = ticker
# #     except:
# #         print("Error occured")

# # print(best_ticker)

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


prices_data = pd.read_csv('price_data/TTC.csv', header=0)
prices_data.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split',
       'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']
prices_data = prices_data[['date', 'Adj_Close']]

df = pd.read_csv('combined_ticker_vectors/TTC.csv', header=0)
df = prices_data.merge(df, left_on='date', right_on='date', how='inner')
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
df = df.drop(columns=['date'])
print(df.columns)

dataset_list = []

for column in df.columns:
    column_array = df[column].values
    dataset_list.append(column_array)
    
dataset_tuple = tuple(dataset_list)
dataset = hstack(dataset_tuple)



class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):        
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)


# Initialization

n_features = 6 # this is number of parallel inputs
n_timesteps = 7 # this is number of timesteps

# convert dataset into input/output
X, y = split_sequences(dataset, n_timesteps)
print(X.shape, y.shape)

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)

train_episodes = 500
batch_size = 16

# training

mv_net.train()
for t in range(train_episodes):
    for b in range(0,len(X),batch_size):
        inpt = X[b:b+batch_size,:,:]
        target = y[b:b+batch_size]    

        x_batch = torch.tensor(inpt,dtype=torch.float32)    
        y_batch = torch.tensor(target,dtype=torch.float32)

        mv_net.init_hidden(x_batch.size(0))
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(x_batch) 
        loss = criterion(output.view(-1), y_batch)  

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())