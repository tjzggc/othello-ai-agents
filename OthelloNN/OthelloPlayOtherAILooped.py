from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import numpy as np
import csv
import torch.nn as nn

# Load your Othello AI model (modify this based on your implementation)
import torch

# Load DQN AI Model
class DQN(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # No activation since we use Q-values
    
# Used to load 3 or 4 Layer AI Models
class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        self.fc1 = nn.Linear(64, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        # Comment out fc4 when doing 3 layer testing
        #self.fc4 = nn.Linear(64, 64)  # Output layer (64 possible moves)
        self.relu = nn.ReLU()        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # Comment out this line and uncomment other two lines for four layer testing.
        x = self.fc3(x) 
        # Comment out these two lines when doing 3 layer testing 
        #x = self.relu(self.fc3(x))
        #x = self.fc4(x)  
        return x

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
folder_path = "ModelSaves"
model.load_state_dict(torch.load(folder_path + "/" + "DQN_agent1_updated.pth", map_location=device))
model.eval()

options = Options()
#options.add_argument("--headless")  # Run in headless mode if you don't need to see the browser
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Open the Othello game
driver.get("https://othello-rust.web.app/")
time.sleep(5)  # Wait for the page to load
driver.maximize_window()

# Function to parse the board state
def get_board():
    cells = driver.find_elements(By.CLASS_NAME, "cell")
    board = np.zeros((8, 8), dtype=int)
    
    for i, cell in enumerate(cells):
        class_names = cell.get_attribute("class")
        row, col = divmod(i, 8)
        if "black" in class_names:
            board[row][col] = 1  # Black (AI)
        elif "white" in class_names:
            board[row][col] = -1  # White (Opponent)
    
    return board

def is_valid_move(board, x, y, player):
    if board[y, x] != 0:
        return False
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        captured = []
        while 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == -player:
            captured.append((nx, ny))
            nx += dx
            ny += dy
        if captured and 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == player:
            return True
    return False

def get_valid_moves(board, player):
    return [(x, y) for x in range(8) for y in range(8) if is_valid_move(board, x, y, player)]

def ai_choose_move(board):
    state = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state)

    valid_moves = get_valid_moves(board, 1)
    if not valid_moves:
        return None

    best_move = max(valid_moves, key=lambda move: q_values[0, move[1] * 8 + move[0]].item())
    return best_move

# Function to make a move
def make_move(row, col):
    index = row * 8 + col
    cells = driver.find_elements(By.CLASS_NAME, "cell")
    cells[index].click()
    time.sleep(2)  # Wait for move to register

    # Function to count final scores
def get_final_score():
    board = get_board()
    score_black = np.sum(board == 1)
    score_white = np.sum(board == -1)
    return score_black, score_white


def check_game_over():
    try:
        result_element = driver.find_element(By.CLASS_NAME, "title")  # Adjust if necessary
        
        if(result_element.is_displayed):
            result_text = result_element.text.lower()

            if(result_text == ""):
                return False
            
            winner = "Unknown"
            black_score, white_score = get_final_score()  # Get final score

            if "you won" in result_text:
                winner = "DQNRust"
            elif "you lost" in result_text:
                winner = "AI Opponent - Normal"
            elif "draw" in result_text:
                winner = "Draw"
            else:
                print("Game Over: Unknown result.")

            print(f"Game Over: {winner} | Final Score - DQNRust: {black_score}, Opponent (White): {white_score}")
            
            filepath = "Output"
            filename = "DQN_agent1_vs_ai_normal.csv"

            with open(filepath + "/" + filename, "a", newline ="") as f:
                writer = csv.writer(f)
                
                # Append game result    
                writer.writerow([winner, black_score, white_score])

            return True  # Game is over

        else:
            return False  # Game is still ongoing
    except:
        return False


current_player = -1
opponent_player = 1
our_ai_player = -1
running = True

num_games = 100

for game in range(num_games):
    
    print(f"Starting game {game + 1}")
    game_running = True

    while game_running:
        if check_game_over():
            try:
                # Click the OK button to dismiss the result
                ok_button = driver.find_element(By.CLASS_NAME, "ok")
                ok_button.click()
                time.sleep(2)  # Wait for the board to reset
                try:
                    # Click New Game button
                    newgame_button = driver.find_element(By.ID, "new-game")
                    newgame_button.click()
                    time.sleep(2)
                except Exception as e:
                    print(f"Could not click New Game button: {e}")
            except Exception as e:
                print(f"Could not click OK button: {e}")

            game_running = False
            break  # Exit current game loop

        board = get_board()
        ai_move = ai_choose_move(board)

        if ai_move:   
            make_move(ai_move[1], ai_move[0])

        time.sleep(2)  # Give time between moves

print("Finished all games.")
driver.quit()