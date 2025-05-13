from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import numpy as np
import csv
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
# Set up Selenium WebDriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

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

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
folder_path = "ModelSaves"
model_path = os.path.join(folder_path, "DQN_agent1.pth")
model_path_update = os.path.join(folder_path, "DQN_agent1_updated.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Experience Replay
replay_buffer = deque(maxlen=5000)
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995  # Decay rate per game
epsilon = epsilon_start
target_update_freq = 100
win_log = []

options = Options()
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

def get_legal_moves(board, player=None):

    legal_moves = []
    for row in range(8):
        for col in range(8):
            if is_valid_move(board, col, row, player):
                legal_moves.append(row * 8 + col)  # Convert 2D index to 1D
    return legal_moves

def ai_choose_move(board, valid_moves):
    state = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

    # Epsilon Greedy
    if random.random() < epsilon:
        action_index = random.choice(valid_moves) # 0-63 
    else:
        q_values = model(state)
        q_values = q_values.detach().cpu().numpy()
        q_values = q_values[0]
        action_index = max(valid_moves, key=lambda move: q_values[move])  # Best move (0-63)

    return divmod(action_index, 8)  # Convert (0-63) to (x, y)

# Function to make a move
def make_move(row, col):
    index = row * 8 + col
    cells = driver.find_elements(By.CLASS_NAME, "cell")
    driver.execute_script("arguments[0].click();", cells[index]) # to try and fix issue I have been having with bottom left corner
    #cells[index].click()
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
        
        if(result_element.is_displayed()):
            result_text = result_element.text.lower()

            if(result_text == ""):
                return False, None
            
            winner = "Unknown"
            black_score, white_score = get_final_score()  # Get final score

            if "you won" in result_text:
                winner = "DQN2"
            elif "you lost" in result_text:
                winner = "AI Opponent - Normal"
            elif "draw" in result_text:
                winner = "Draw"
            else:
                print("Game Over: Unknown result.")

            print(f"Game Over: {winner} | Final Score - DQN2: {black_score}, Opponent (White): {white_score}")
            
            filepath = "Output"
            filename = "DQN2train_vs_ai.csv"

            with open(filepath + "/" + filename, "a", newline ="") as f:
                
                # Append game result    
                csv.writer(f).writerow([winner, black_score, white_score])

            return True, winner  # Game is over

        else:
            return False, None  # Game is still ongoing
    except:
        return False, None


def train_dqn(step_count):
    if len(replay_buffer) >= batch_size and step_count % 4 == 0:
        
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * gamma * max_next_q

        loss = loss_fn(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Reward Class
class OthelloRewarder:
    def __init__(self, board_getter, move_checker):
        self.get_board = board_getter  # Callable to get board
        self.get_valid_moves = move_checker  # Callable to get valid moves

    def is_game_over(self, board):
        return not (self.get_valid_moves(board, 1) or self.get_valid_moves(board, -1))

    def get_score(self, board):
        black_score = np.sum(board == 1)
        white_score = np.sum(board == -1)
        return black_score, white_score

    def count_flipped_pieces(self, x, y, player, board):
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        flipped = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            captured = []
            while 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == -player:
                captured.append((nx, ny))
                nx += dx
                ny += dy
            if captured and 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == player:
                flipped += len(captured)
        return flipped

    def get_reward(self, player, move, board):
        if self.is_game_over(board=board):
            black_score, white_score = self.get_score(board=board)
            if (player == 1 and black_score > white_score) or (player == -1 and white_score > black_score):
                return 5  # Win
            else:
                return -5  # Loss

        if move is None:
            return -1  # Pass

        x, y = move
        flipped_pieces = self.count_flipped_pieces(x, y, player, board)
        opponent_moves = self.get_valid_moves(board, -player)
        opponent_mobility_penalty = 0.05 * len(opponent_moves)
        corner_penalty = 3 if any(m in [(0,0),(0,7),(7,0),(7,7)] for m in opponent_moves) else 0

        return (0.1 * flipped_pieces) - opponent_mobility_penalty - corner_penalty


def is_dqn_turn():
    
    html_class = driver.find_element(By.TAG_NAME, "html").get_attribute("class")

    if "player-turn" in html_class:
        turn = True
    elif "ai-turn" in html_class:
        turn = False
    else:
        turn = None
    return turn  


# Main Loop
running = True
rewarder = OthelloRewarder(get_board, get_valid_moves)
action = ActionChains(driver)

# Train on 1500 games switching to harder difficulty after every 500 games.
num_games = 500
step_count = 0

for game in range(num_games):

    # Change game difficulty to Normal
    """ if game == 0:
        driver.find_element(By.ID, "settings-button").click()
        time.sleep(1)
        slider = driver.find_element(By.CSS_SELECTOR, "input[type='range']")
        action.click_and_hold(slider).move_by_offset(25, 0).release().perform()
        driver.find_element(By.ID, "close-settings").click() """

    # Change game difficulty to Hard
    """ if game == 0:
        driver.find_element(By.ID, "settings-button").click()
        time.sleep(1)
        slider = driver.find_element(By.CSS_SELECTOR, "input[type='range']")
        action.click_and_hold(slider).move_by_offset(50, 0).release().perform()
        driver.find_element(By.ID, "close-settings").click() """

    # Change game difficulty to Very Hard
    if game == 0:
        driver.find_element(By.ID, "settings-button").click()
        time.sleep(1)
        slider = driver.find_element(By.CSS_SELECTOR, "input[type='range']")
        action.click_and_hold(slider).move_by_offset(100, 0).release().perform()
        driver.find_element(By.ID, "close-settings").click()

    print(f"Starting game {game + 1}")
    game_running = True

    while game_running:
        over, winner = check_game_over()
        if over:
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

            # Log results
            if winner == "DQN2":
                win_log.append(1)
            elif winner == "AI Opponent - Easy":
                win_log.append(0)
            else:
                win_log.append(0.5)
            if game % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
                print("Target network updated.")
            if game % 100 == 0:
                torch.save(model.state_dict(), model_path_update)
                model.load_state_dict(torch.load(model_path_update, map_location=device)) # Load the newly saved model to continue training.
                print("Model saved.")
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(f"Epsilon after game {game + 1}: {epsilon:.4f}")
            
            break # Exit current game loop

        if not is_dqn_turn():
            time.sleep(2)
            continue

        board = get_board()
        valid_moves = get_legal_moves(board, 1)
        if not valid_moves:
            time.sleep(2)
            continue

        ai_move = ai_choose_move(board, valid_moves) 
        #print(f"Valid moves: {valid_moves}")
        #print(f"Selected move: {ai_move}")
        if ai_move:   
            row, col = ai_move 
            prev_state = board.flatten().astype(np.float32)
            make_move(row, col) 
            next_board = get_board()
            reward = rewarder.get_reward(player=1, move=ai_move, board=next_board)
            next_state = get_board().flatten().astype(np.float32)
            done, winner = check_game_over()

            action_index = row * 8 + col
            replay_buffer.append((prev_state, action_index, reward, next_state, done))
            step_count += 1
            train_dqn(step_count)
        else:
            # Pass turn: penalize
            prev_state = board.flatten().copy().astype(np.float32)
            reward = rewarder.get_reward(player=1, move=None, board=board)
            next_state = board.flatten().copy().astype(np.float32)
            replay_buffer.append((prev_state, 0, reward, next_state, False))

        time.sleep(3)  # Give time between moves

torch.save(model.state_dict(), model_path_update)
model.load_state_dict(torch.load(model_path_update, map_location=device)) # Load the newly saved model to continue training.
print("Model saved.")
print("Finished all games.")
driver.quit()