import torch 
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import numpy as np

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ticker", "--ticker", help="stock ticker symbol")
parser.add_argument("-train_file", "--train_file")
parser.add_argument("-trade_file", "--trade_file")
parser.add_argument("-model", "--model", help="trained model to be used for forecasting")
parser.add_argument("-save_loc", "--save_loc", help="location to save results")
args = parser.parse_args()

trade = pd.read_csv(args.trade_file)
data = trade[["close"]]
min_val = min(pd.read_csv(args.train_file)["close"])
max_val = max(pd.read_csv(args.train_file)["close"])

scaler = MinMaxScaler()
data = scaler.fit_transform(data['close'].values.reshape(-1,1))


input_size = 1
hidden_size = 32
num_layers = 2
sequence_length = 10
output_size = 5


def create_sequences(data, sequence_length):
    xs = []
    ytrue = []
    for i in range(len(data)-sequence_length-1-5):
        x = data[i:(i+sequence_length)]
        xs.append(x)
        ytrue.append((data[i+sequence_length]*(max_val - min_val))+min_val)
    return torch.tensor(xs), ytrue


train_size = len(data)
train_data = data[:train_size, :]
x, ytrue = create_sequences(train_data, sequence_length)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

model = LSTM(input_size, hidden_size, num_layers, output_size)

model.load_state_dict(torch.load(args.model))

y_test_pred = model(x.float())


s = x.size()
x_input = scaler.inverse_transform(x.reshape([s[0],s[1]]).detach().numpy())
y_output = scaler.inverse_transform(y_test_pred.detach().numpy())




cash_bal = 1000000
initial_cash = cash_bal
hmax = 100
stock_held = 0


raw_actions = []
cash_mem = [initial_cash]

for i in range(len(x_input)):
    current_price = x_input[i][-1]
    future_forecast = y_output[i][-1]
    
    action = 0
    
    if future_forecast > current_price: #increasing trend => buy
        raw_actions.append(1)
        if current_price*hmax < cash_bal: #have enough cash to buy
            stock_held += hmax
            cash_bal -= current_price*hmax
            action = 1
        else:
            num_to_buy = int(cash_bal/current_price) #num can afford
            stock_held += num_to_buy
            cash_bal -= current_price*num_to_buy
            if num_to_buy > 0:
                action = 1
    else: #decreasing trend => sell
        raw_actions.append(-1)
        if stock_held > hmax:
            stock_held -= hmax
            cash_bal += current_price*hmax
            action = -1
        else:
            num_to_sell = stock_held
            stock_held -= num_to_sell
            cash_bal += current_price*num_to_sell
            if num_to_sell > 0:
                  action = -1
    cash_mem.append(cash_bal + stock_held*current_price)


           


print(cash_bal)
print(stock_held)
print((cash_bal + stock_held*current_price) - initial_cash)
        

import pandas as pd
actions_file = args.save_loc + "/actions.csv"
account_bal_file = args.save_loc + "/account_balance.csv"
pd.DataFrame({"raw_actions": raw_actions}).to_csv(actions_file)
pd.DataFrame({"account_value": cash_mem}).to_csv(account_bal_file)













