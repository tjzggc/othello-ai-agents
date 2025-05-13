import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import DataLoader, TensorDataset

# Dataset file location
othello_games_file = 'othello_dataset.csv'

# Initial Values
initial_board = '000000000000000000000000000-+000000+-000000000000000000000000000'

# Conversion values
who = ('Draw', 'Black', 'White')
marker = {'0': 0, '+': 1, '-': -1,
          0: '0', 1: '+', -1: '-',
          }
training_value = {'+': 1.0, '-': 0.0, '0': 0.5}
letter_conv = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
               'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
               }
increments = ((-1, -1), (-1, 0), (-1, 1),
              (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1),
              )

# Convert position formats.
def a1_num(pos):
    return (int(pos[1]) - 1) * 8 + letter_conv[pos[0]]


def a1_rc(pos):
    return int(pos[1]) - 1, letter_conv[pos[0]]


# Convert the character string representation of the board to an array.
def txt_training(brd):
    result = []
    for b in brd:
        result.append(training_value[b])
    return result


# Return the value of the board position given row/column coordinates.
def chk(brd, r, c):
    if 0 <= r < 8 and 0 <= c < 8:
        return marker[brd[r * 8 + c]]
    else:
        return 99


# Update the board position given row/column coordinates.
def upd(brd, r, c, player):
    return brd[:r * 8 + c] + marker[player] + brd[r * 8 + c + 1:]


# In order to know the board configurations in the game logs,
# you have to 'play' each move.
def move(brd, pos, player):
    r, c = pos
    if chk(brd, r, c) != 0:
        return brd

    for inc in increments:
        inc_r, inc_c = inc
        i = 1
        while chk(brd, r + inc_r * i, c + inc_c * i) == -player:
            i += 1
        if i > 1 and chk(brd, r + inc_r * i, c + inc_c * i) == player:
            i -= 1
            while i >= 0:
                brd = upd(brd, r + inc_r * i, c + inc_c * i, player)
                i -= 1
    return brd


# Convert numeric player information to text.
def conv_winner(x):
    return who[int(x)]


# Add the board configurations and player information to the game log.
# Note: A player can play twice in a row if the other player does not have
# a valid move.
def conv_log(log):
    player = 1
    b0 = initial_board
    result = []
    for i in range(0, len(log), 2):
        b1 = move(b0, a1_rc(log[i:i + 2]), player)
        if b1 == b0:
            player *= -1
            b1 = move(b0, a1_rc(log[i:i + 2]), player)
        result.append((who[player], b0, log[i:i + 2], b1))
        b0 = b1
        player *= -1

    return tuple(result)

def initialize_weights(m):
    """Applies Xavier initialization to model weights."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        self.fc1 = nn.Linear(64, 128) # Converted to four layers to test out.
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)  # Output layer (64 possible moves)
        #self.fc1 = nn.Linear(64, 128)
        #self.fc2 = nn.Linear(128, 128)
        #self.fc3 = nn.Linear(128, 64)  # Output layer (64 possible moves)
        self.relu = nn.ReLU()        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  
        return x

def train_model():
    
    # Read the historic game logs
    historic_game_data = pd.read_csv(othello_games_file,
                                     header=0,
                                     skipfooter=1,  # Skips the last row
                                     engine='python',  # Needed for skipfooter
                                     names=['eOthello Game ID',
                                            'Winner',
                                            'Log',
                                            ],
                                     converters={'Winner': conv_winner,
                                                 'Log': conv_log,
                                                 },
                                     index_col=['eOthello Game ID'], )
    
    # Moves of interest are moves that a player made in games that they won.
    winning_moves_list = []
    for game in list(historic_game_data[historic_game_data['Winner'] == 'Black'].Log):
        for game_move in game:
            if game_move[0] == 'Black':
                winning_moves_list.append(('Black', game_move[1], a1_num(game_move[2])))
    for game in list(historic_game_data[historic_game_data['Winner'] == 'White'].Log):
        for game_move in game:
            if game_move[0] == 'White':
                winning_moves_list.append(('White', game_move[1], a1_num(game_move[2])))
    
    # For machine learning:
    #   - board values are transformed to values between 0 and 1
    #   - move values are transformed to integers between 0 and 63
    training_df = pd.DataFrame(winning_moves_list, columns=['Player', 'Feature - Board', 'Label - Move'])
    training_df['Feature - Board'] = training_df['Feature - Board'].apply(txt_training)
    
    # Convert to NumPy arrays
    board_states = np.vstack(training_df['Feature - Board']).astype(np.float32)  # Board states as input
    best_moves = np.array(training_df['Label - Move']).astype(np.int64)  # Best move (0-63) as output
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(board_states)
    y_train_tensor = torch.tensor(best_moves)
    
    # Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloNet().to(device)
    
    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64 # Split dataset into this many sets.
    epochs = 25     # Number times through the dataset. 
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.apply(initialize_weights)
    
    # Training loop
    for epoch in range(epochs):
        
        model.train() 
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move to GPU if available
            optimizer.zero_grad()  # Reset gradients
            output = model(batch_X)  # Forward pass
            loss = criterion(output, batch_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)  # Compute average loss
        
        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "4LayerBackProp.pth")


train_model()