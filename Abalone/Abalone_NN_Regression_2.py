# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from scipy.stats import zscore

pd.set_option('display.max_columns', None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
def rmsle(y_pred, y_true):
    return torch.sqrt(torch.mean(torch.square(torch.log(y_pred + 1) - torch.log(y_true + 1))))

def detect_outliers_and_create_feature(data, columns, threshold=3):
    for col in columns:
        # Calculate Z-scores for the selected column
        data[col + '_zscore'] = zscore(data[col])
        
        # Create a new binary feature indicating outliers
        data[col + '_is_outlier'] = (np.abs(data[col + '_zscore']) > threshold).astype(int)
        
    return data

# %%
test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')

enc=OneHotEncoder(drop='first')

test_enc=enc.fit_transform(test[['Sex']])
test_encoded_df = pd.DataFrame(test_enc.toarray(), columns=enc.get_feature_names_out(['Sex']))
test_df = pd.concat([test, test_encoded_df], axis=1)
test_df=test_df.drop(columns=['Sex'])

train_enc=enc.fit_transform(train[['Sex']])
train_encoded_df = pd.DataFrame(train_enc.toarray(), columns=enc.get_feature_names_out(['Sex']))
train_df = pd.concat([train, train_encoded_df], axis=1)
train_df=train_df.drop(columns=['Sex'])

# %%
train_df=detect_outliers_and_create_feature(train_df,columns=['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1',
       'Whole weight.2', 'Shell weight'],threshold=3)

test_df=detect_outliers_and_create_feature(test_df,columns=['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1',
       'Whole weight.2', 'Shell weight'],threshold=3)

# %%
feature_names=[
    'Length',
    'Diameter', 
    'Height', 
    'Whole weight', 
    'Whole weight.1',
    'Whole weight.2', 
    'Shell weight', 
    'Sex_I', 
    'Sex_M',
    'Length_is_outlier',
    'Diameter_is_outlier',
    'Height_is_outlier',
    'Whole weight_is_outlier',
    'Whole weight.1_is_outlier',
    'Whole weight.2_is_outlier',
    'Shell weight_is_outlier'
       ]

X=train_df[feature_names].values
y=train_df.Rings.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

X_train=torch.tensor(X_train, dtype=torch.float32)
y_train=torch.tensor(y_train, dtype=torch.float32)

X_test=torch.tensor(X_test, dtype=torch.float32)
y_test=torch.tensor(y_test, dtype=torch.float32)


# %%
X_test.shape

# %%
### Simple NN CPU prototyping

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, output_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.rnn(x)
        x = self.fc(h)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = self.fc(h[-1, :, :])
        return x

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        x = self.fc(h[-1, :, :])
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SimpleRBFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(hidden_size, input_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1) - self.centers.unsqueeze(0)
        x = torch.norm(x, dim=-1)
        x = torch.exp(-self.beta * x)
        x = self.fc(x)
        return x

input_size = 16
output_size = 1
hidden_size = 64

# Instantiate models
feedforward_nn = SimpleFeedForwardNN(input_size, hidden_size, output_size)
cnn = SimpleCNN(input_channels=3, output_size=output_size)
rnn = SimpleRNN(input_size, hidden_size, output_size)
lstm = SimpleLSTM(input_size, hidden_size, output_size)
gru = SimpleGRU(input_size, hidden_size, output_size)
transformer = SimpleTransformer(input_size, output_size)
autoencoder = SimpleAutoencoder(input_size, hidden_size, output_size)
rbfn = SimpleRBFN(input_size, hidden_size, output_size)

models = {
    'FeedforwardNN': feedforward_nn,
    # 'CNN': cnn,
    # 'RNN': rnn,
    # 'LSTM': lstm,
    # 'GRU': gru,
    # 'Transformer': transformer,
    'Autoencoder': autoencoder,
    'RBFN': rbfn
}

for model_name, model in models.items():
    # Set model to evaluation mode
    model.eval()

    # Pass validation data through the model
    with torch.no_grad():
        val_outputs = model(X_test)

    # Calculate RMSLE
    rmsle_score = rmsle(val_outputs, y_test)
    r2=r2_score(val_outputs,y_test)
    # #Convert predictions and actual values to numpy arrays for calculating R2 score
    # y_pred = val_outputs.numpy()
    # y_true = y_test.numpy().squeeze()
    # print("Shape of y_true:", y_true.shape)
    # print("Values of y_true:", y_true)
    # print("Shape of y_pred:", y_pred.shape)
    # print("Values of y_pred:", y_pred)
    # # Calculate R2 score
    # r2 = r2_score(y_true, y_pred)

    # Print results
    print(f"Model: {model_name}")
    print(f"RMSLE: {rmsle_score}")
    print(f"R2 Score: {r2}")
    print()

# %%
def get_train_length(dataset, batch_size, test_percent):
    length = len(dataset)
    length *= 1 - test_percent
    train_length_values = []
    for x in range(int(length) - 100, int(length)):  # Assuming the last 100 samples are for testing
        modulo = x % batch_size
        if modulo == 0:
            train_length_values.append(x)
            print(x)  # Debugging print
    print("Train length values:", train_length_values)  # Debugging print
    return max(train_length_values)

# %%
get_train_length(X,batch_size=128,test_percent=.4)

# %%
X_train.shape

# %%
rnn.eval()
with torch.no_grad():
    val_outputs=rnn(X_test)
rmsle_score = rmsle(val_outputs, y_test)
r2=r2_score(val_outputs,y_test)
print(rmsle_score)
print(r2)

# %%
feedforward_nn.eval()
with torch.no_grad():
    val_outputs=feedforward_nn(X_test)
rmsle_score = rmsle(val_outputs, y_test)
r2=r2_score(val_outputs,y_test)
print(rmsle_score)
print(r2)

# %%
autoencoder.eval()
with torch.no_grad():
    val_outputs=autoencoder(X_test)
rmsle_score = rmsle(val_outputs, y_test)
r2=r2_score(val_outputs,y_test)
print(rmsle_score)
print(r2)

# %%
class Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# %%
params = {
    'hidden_size': [64, 128, 256],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'loss_function': ['MSELoss', 'L1Loss', 'SmoothL1Loss']  # Add additional regression loss functions here
}

# %%
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

# %%
# Define your neural network architectures as classes
class SimpleRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ComplexRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# %%
### CPU accelerated prototyping

input_size = 16  # Number of input features
output_size = 1  # Number of output/target values
num_epochs = 10

best_score = None
best_params = None

# Define your params for different neural network architectures
params = {
    'architectures': [SimpleRegressor, ComplexRegressor],
    'hidden_sizes': [64, 128, 256],
    'learning_rates': [0.001, 0.01, 0.1],
    'num_epochs': 10,
    'batch_size': 64
}

# Split your data into train and test sets (you may need to adjust this based on your specific data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Train and evaluate different neural network architectures
best_score = None
best_params = None
for architecture in params['architectures']:
    if architecture == SimpleRegressor:
        for hidden_size in params['hidden_sizes']:
            for learning_rate in params['learning_rates']:
                # Initialize model, loss function, optimizer
                model = architecture(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Move model and data to GPU
                # model.to(device)
                # criterion.to(device)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_test_tensor)
                    val_loss = val_loss.item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs, y_test_tensor)

                # Update best_score and best_params based on score
                if best_score is None or score < best_score:
                    best_score = score
                    best_params = {
                        'architecture': architecture.__name__,
                        'hidden_size': hidden_size,
                        'learning_rate': learning_rate,
                        'score': score
                    }
    else:  # For ComplexRegressor
        for hidden_size1 in params['hidden_sizes']:
            for hidden_size2 in params['hidden_sizes']:
                for learning_rate in params['learning_rates']:
                    # Initialize model, loss function, optimizer
                    model = architecture(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    # Move model and data to GPU
                    # model.to(device)
                    # criterion.to(device)

                    # Create DataLoader with the specified batch size
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                    # Training loop
                    for epoch in range(params['num_epochs']):
                        model.train()
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            outputs = outputs.squeeze(dim=1)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()

                    # Evaluation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test_tensor)
                        val_outputs = val_outputs.squeeze(dim=1)
                        val_loss = criterion(val_outputs, y_test_tensor)
                        val_loss = val_loss.item()

                    # Calculate RMSLE or any other metric
                    score = rmsle(val_outputs, y_test_tensor)

                    # Update best_score and best_params based on score
                    if best_score is None or score < best_score:
                        best_score = score
                        best_params = {
                            'architecture': architecture.__name__,
                            'hidden_size1': hidden_size1,
                            'hidden_size2': hidden_size2,
                            'learning_rate': learning_rate,
                            'score': score
                        }


print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### GPU accelerated prototyping 

input_size = 16  # Number of input features
output_size = 1  # Number of output/target values
num_epochs = 10

best_score = None
best_params = None

# Convert data to PyTorch tensors and move them to GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Train and evaluate different neural network architectures
best_score = None
best_params = None
for architecture in params['architectures']:
    if architecture == SimpleRegressor:
        for hidden_size in params['hidden_sizes']:
            for learning_rate in params['learning_rates']:
                # Initialize model, loss function, optimizer and move them to GPU
                model = architecture(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1).to(device)
                criterion = nn.MSELoss().to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_test_tensor)
                    val_loss = val_loss.item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs, y_test_tensor)

                # Update best_score and best_params based on score
                if best_score is None or score < best_score:
                    best_score = score
                    best_params = {
                        'architecture': architecture.__name__,
                        'hidden_size': hidden_size,
                        'learning_rate': learning_rate,
                        'score': score
                    }
    else:  # For ComplexRegressor
        for hidden_size1 in params['hidden_sizes']:
            for hidden_size2 in params['hidden_sizes']:
                for learning_rate in params['learning_rates']:
                    # Initialize model, loss function, optimizer and move them to GPU
                    model = architecture(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
                    criterion = nn.MSELoss().to(device)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    # Create DataLoader with the specified batch size
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                    # Training loop
                    for epoch in range(params['num_epochs']):
                        model.train()
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            outputs = outputs.squeeze(dim=1)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()

                    # Evaluation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test_tensor)
                        val_outputs = val_outputs.squeeze(dim=1)
                        val_loss = criterion(val_outputs, y_test_tensor)
                        val_loss = val_loss.item()

                    # Calculate RMSLE or any other metric
                    score = rmsle(val_outputs, y_test_tensor)

                    # Update best_score and best_params based on score
                    if best_score is None or score < best_score:
                        best_score = score
                        best_params = {
                            'architecture': architecture.__name__,
                            'hidden_size1': hidden_size1,
                            'hidden_size2': hidden_size2,
                            'learning_rate': learning_rate,
                            'score': score
                        }
print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### CPU
### batch size = 32, adam, 17m 35s Best Parameters: {'hidden_size': 64, 'learning_rate': 0.001, 'batch_size': 32} 0.1555922120809555
### batch size = 128, adam, 6m 58s Best Parameters: {'hidden_size': 256, 'learning_rate': 0.001, 'batch_size': 128} 0.1553862363100052

input_size = 16  # Number of input features
output_size = 1  # Number of output/target values
num_epochs = 10

best_score = None
best_params = None
for hidden_size in params['hidden_size']:
    for learning_rate in params['learning_rate']:
        for batch_size in params['batch_size']:
            scores = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Convert numpy arrays to PyTorch tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

                # Initialize model, loss function, optimizer
                model = Regressor(input_size, hidden_size, output_size)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Training loop
                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)  # Squeeze the predicted tensor to remove singleton dimension
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_loss = val_loss.item()

                # Calculate RMSLE
                score = rmsle(val_outputs, y_val_tensor)
                scores.append(score.item())  # Append the score as a Python scalar, not a tensor

            mean_score = np.mean(scores)
            if best_score is None or mean_score < best_score:
                best_score = mean_score
                best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'batch_size': batch_size}

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### GPU 
### batch size = 32, adam, 18m 18s Best Parameters: {'hidden_size': 128, 'learning_rate': 0.01, 'batch_size': 32} Best Score (RMSLE): 0.15574217736721038
### batch size = 128, adam 6m 35s Best Parameters: {'hidden_size': 256, 'learning_rate': 0.001, 'batch_size': 128} 0.15597270727157592

input_size = 16  # Number of input features
output_size = 1  # Number of output/target values
num_epochs = 10

best_score = None
best_params = None
for hidden_size in params['hidden_size']:
    for learning_rate in params['learning_rate']:
        for batch_size in params['batch_size']:
            scores = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Convert numpy arrays to PyTorch tensors and move them to GPU
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

                # Initialize model, loss function, optimizer
                model = Regressor(input_size, hidden_size, output_size)
                model.to(device)
                criterion = nn.MSELoss()
                criterion.to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Training loop
                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)  # Squeeze the predicted tensor to remove singleton dimension
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_loss = val_loss.item()

                # Calculate RMSLE
                score = rmsle(val_outputs, y_val_tensor)
                scores.append(score.item())  # Append the score as a Python scalar, not a tensor

            mean_score = np.mean(scores)
            if best_score is None or mean_score < best_score:
                best_score = mean_score
                best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'batch_size': batch_size}

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)


# %%
### GPU 
### batch size = 32, adam, 18m 18s Best Parameters: {'hidden_size': 128, 'learning_rate': 0.01, 'batch_size': 32} 0.15574217736721038
### batch size = 128, SGD, 6m 8s Best Parameters: {'hidden_size': 64, 'learning_rate': 0.001, 'batch_size': 128} 0.15696884393692018

input_size = 16  # Number of input features
output_size = 1  # Number of output/target values
num_epochs = 10

best_score = None
best_params = None
for hidden_size in params['hidden_size']:
    for learning_rate in params['learning_rate']:
        for batch_size in params['batch_size']:
            scores = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Convert numpy arrays to PyTorch tensors and move them to GPU
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

                # Initialize model, loss function, optimizer
                model = Regressor(input_size, hidden_size, output_size)
                model.to(device)
                criterion = nn.MSELoss()
                criterion.to(device)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Training loop
                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)  # Squeeze the predicted tensor to remove singleton dimension
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_loss = val_loss.item()

                # Calculate RMSLE
                score = rmsle(val_outputs, y_val_tensor)
                scores.append(score.item())  # Append the score as a Python scalar, not a tensor

            mean_score = np.mean(scores)
            if best_score is None or mean_score < best_score:
                best_score = mean_score
                best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'batch_size': batch_size}

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)



