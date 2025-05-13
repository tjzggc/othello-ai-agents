import pygame
import torch
import numpy as np
import csv
import os

# Load AI Model
class Othello(torch.nn.Module):
    def __init__(self):
        super(Othello, self).__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Othello().to(device)
folder_path = "ModelSaves"
model.load_state_dict(torch.load(folder_path + "/" + "3Layer.pth", map_location=device))
model.eval()

# Pygame Setup
pygame.init()
WIDTH, HEIGHT = 600, 600
CELL_SIZE = WIDTH // 8
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Othello - Human vs AI")

# Initialize Board
board = np.zeros((8, 8), dtype=int)
board[3, 3], board[4, 4] = 1, 1
board[3, 4], board[4, 3] = -1, -1
current_player = -1  # Human plays as Black (-1)
# Initialize Font
pygame.font.init()
font = pygame.font.Font(None, 36)  # Default Pygame font, size 36

def draw_board():
    screen.fill(GREEN)
    for x in range(8):
        for y in range(8):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 2)

            if board[y, x] == -1:  # Black Piece
                pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 3)
            elif board[y, x] == 1:  # White Piece
                pygame.draw.circle(screen, WHITE, rect.center, CELL_SIZE // 3)
    
    # Show turn
    turn_text = f"Turn: {'Black' if current_player == -1 else 'White'}"
    text_surface = font.render(turn_text, True, WHITE if current_player == 1 else BLACK)
    screen.blit(text_surface, (20, 20))

    pygame.display.flip()

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

def apply_move(board, x, y, player):
    board[y, x] = player
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    for dx, dy in directions:
        captured = []
        nx, ny = x + dx, y + dy
        while 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == -player:
            captured.append((nx, ny))
            nx += dx
            ny += dy
        if captured and 0 <= nx < 8 and 0 <= ny < 8 and board[ny, nx] == player:
            for cx, cy in captured:
                board[cy, cx] = player

def display_winner():
    black_score = np.sum(board == -1)  # Count black pieces
    white_score = np.sum(board == 1)   # Count white pieces

    # Determine winner
    if black_score > white_score:
        result_text = f"You Win! (Black: {black_score} - White: {white_score})"
    elif white_score > black_score:
        result_text = f"AI Wins! (Black: {black_score} - White: {white_score})"
    else:
        result_text = f"It's a Draw! (Black: {black_score} - White: {white_score})"

    # Render and display text
    screen.fill(GREEN)  # Clear the screen
    text_surface = font.render(result_text, True, WHITE)
    screen.blit(text_surface, (WIDTH // 2 - 150, HEIGHT // 2))  # Center text
    print(result_text)
    pygame.display.flip()
    save_game_result(black_score, white_score)

    pygame.time.delay(5000)  # Show result for 5 seconds before exiting

def ai_choose_move(board):
    state = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state)

    valid_moves = get_valid_moves(board, 1)
    if not valid_moves:
        return None

    best_move = max(valid_moves, key=lambda move: q_values[0, move[1] * 8 + move[0]].item())
    return best_move

def save_game_result(black_score, white_score):
    
    if black_score > white_score:
        winner = "Human"
    elif white_score > black_score:
        winner = "AI"
    else:
        winner = "Draw"

    filename = "game_results.csv"

    # Check if file exists.
    file_exists = os.path.isFile(filename)

    with open(filename, "a", newline ="") as f:
        writer = csv.writer(f)

        # Write header if file doesn't already exist.
        if not file_exists:
            writer.writerow(["Winner", "Black Count", "White Count"])

        # Append game result    
        writer.writerow([winner, black_score, white_score])


# Game Loop
running = True
ai_player = 1
human_player = -1

while running:
    draw_board()
    
    # Create no valid move section for human and one for AI.
    # Should just switch to the next player. Want to find a way to display player turn on screen.
    human_moves = get_valid_moves(board, human_player)
    ai_moves = get_valid_moves(board, ai_player)

    # If both players cannot move, end the game
    if not human_moves and not ai_moves:
        display_winner()
        print("Game Over!")
        break  

    # If it's the human's turn and they cannot move, switch to AI
    if current_player == human_player and not human_moves:
        print("No valid moves for you. Skipping turn...")
        current_player = ai_player  # AI's turn
        continue  # Skip to AI's turn

    # If it's the AI's turn and it cannot move, switch to the human
    if current_player == ai_player and not ai_moves:
        print("AI has no valid moves. Skipping turn...")
        current_player = human_player  # Human's turn
        continue  # Skip to Human's turn


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and current_player == human_player:
            mx, my = pygame.mouse.get_pos()
            x, y = mx // CELL_SIZE, my // CELL_SIZE

            if is_valid_move(board, x, y, current_player):
                apply_move(board, x, y, current_player)
                draw_board()
                current_player = ai_player  # Switch to AI

        if current_player == ai_player:
            pygame.time.delay(1000)  # Short delay for AI to "think"
            ai_move = ai_choose_move(board)
            if ai_move:
                apply_move(board, ai_move[0], ai_move[1], current_player)
                draw_board()
            
            current_player = human_player  # Switch back to human

pygame.quit()
