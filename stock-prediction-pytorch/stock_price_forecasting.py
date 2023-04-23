import torch 
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ticker", "--ticker", help="stock ticker symbol")
parser.add_argument("-train_file", "--train_file", help="location of training data")
parser.add_argument("-model_loc", "--model_loc", help="location to save trained model")
parser.add_argument("-lr", "--lr", type=float, help="learning rate for training")
parser.add_argument("-epochs", "--epochs", type=int, help="number of epochs for training")
parser.add_argument("-layers", "--layers", type=int, help="number of hidden layers")
parser.add_argument("-hidden", "--hidden", type=int, help="number of nodes in hidden layers")
parser.add_argument("-bs", "--bs", type=int, help="batch size")
parser.add_argument("-save_loc", "--save_loc", help="location to save results")
args = parser.parse_args()

data = pd.read_csv(args.train_file)[["close"]]


min_val = min(data["close"])
max_val = max(data["close"])

# preprocessing
scaler = MinMaxScaler()
data = scaler.fit_transform(data['close'].values.reshape(-1,1))


# define constants
input_size = 1
hidden_size = args.hidden #32
num_layers = args.layers #2
sequence_length = 10
output_size = 5
batch_size = args.bs #64
num_epochs = args.epochs #5000

# split data into training and testing sets
train_size = len(data) #int(len(data) * 0.8)
test_size = len(data) #- train_size
train_data = data[:train_size, :]
# test_data = data[train_size:, :]
test_data = data[:train_size, :]

# create input and output sequences for training and testing sets
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length-1-5):
        x = data[i:(i+sequence_length)]
        y = data[(i+sequence_length):(i+sequence_length+output_size)]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)

x_train, y_train = create_sequences(train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)

# define LSTM model
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

# initialize model and optimizer
model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# train model
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        inputs = x_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]
        
        # forward pass
        outputs = model(inputs.float())
        
        s = labels.size()
        loss = criterion((outputs*(max_val-min_val))+min_val, ((labels.reshape([s[0], s[1]])*(max_val-min_val))+min_val).float())
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


torch.save(model.state_dict(), args.model_loc)
           
#test model
model.eval()
with torch.no_grad():
    test_inputs = x_test.float()
    test_labels = y_test.float()
    test_outputs = model(test_inputs)
    
    s = test_labels.size()
    test_loss = criterion((test_outputs*(max_val-min_val))+min_val, (test_labels.reshape([s[0], s[1]])*(max_val-min_val))+min_val)
    print('Test Loss: {:.4f}'.format(test_loss.item()))
    
# invert scaling to get original data
test_predict = scaler.inverse_transform(test_outputs.detach().numpy())
test_labels = scaler.inverse_transform(test_labels.reshape([s[0], s[1]]).detach().numpy())

for i in range(10):
    print(test_predict[i])
    print(test_labels[i])
    print("-------------------")
    print()









