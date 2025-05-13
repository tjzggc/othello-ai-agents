import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # No activation since we use Q-values

class OthelloDQN:
    def __init__(self, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN().to(self.device)
        self.target_dqn = DQN().to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.update_target_network()

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def select_action(self, state, legal_moves):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        if random.random() < self.epsilon:
            action_index = random.choice(legal_moves)  # Select random move (0-63)
        else:
            q_values = self.dqn(state_tensor).detach().cpu().numpy()
            action_index = max(legal_moves, key=lambda move: q_values[move])  # Best move (0-63)

        return divmod(action_index, 8)  # Convert (0-63) to (x, y)


    def store_experience(self, state, action, reward, next_state, done):
        action_index = action[0] * 8 + action[1]  # Convert (x, y) to index
        self.memory.append((state, action_index, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


def train_ai_vs_ai(agent1, agent2, env, episodes=10000, gamma=0.99):
    wins = {"agent1": 0, "agent2": 0, "draws": 0}

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards = {"agent1": 0, "agent2": 0}
        current_agent = agent1
        current_agent_name = "agent1"
        pass_count = 0

        while not done:
            legal_moves = env.get_legal_moves(env.current_player)  # always be explicit

            if not legal_moves:
                pass_count += 1
                if pass_count >= 2:
                    done = True
                    break
                state = env.get_state()
                env.current_player *= -1  # switch player in environment
                current_agent, current_agent_name = (
                    agent2, "agent2") if current_agent == agent1 else (agent1, "agent1")
                continue
            
            pass_count = 0  # reset if a legal move exists 


            perspective = 1 if current_agent == agent2 else -1
            normalized_state = (np.array(state) * perspective).tolist()
            action = current_agent.select_action(normalized_state, legal_moves)
            #print(f"Move: {action}, Player: {env.current_player}")

            next_state, reward, done = env.step(action)
            total_rewards[current_agent_name] += reward

            next_normalized_state = (np.array(next_state) * perspective).tolist()
            current_agent.store_experience(normalized_state, action, reward, next_normalized_state, done)
            current_agent.train(batch_size=32)

            current_agent, current_agent_name = (agent2, "agent2") if current_agent == agent1 else (agent1, "agent1")
            state = next_state

        black_score, white_score = env.get_score()
        if black_score > white_score:
            wins["agent1"] += 1
        elif white_score > black_score:
            wins["agent2"] += 1
        else:
            wins["draws"] += 1

        if episode % 100 == 0:
            print(f"Episode {episode}")

        if episode % 1000 == 0:
            total_games = sum(wins.values())
            win_rate_1 = (wins["agent1"] / total_games) * 100 if total_games > 0 else 0
            win_rate_2 = (wins["agent2"] / total_games) * 100 if total_games > 0 else 0
            draw_rate = (wins["draws"] / total_games) * 100 if total_games > 0 else 0

            print(f"Episode {episode} | Agent1 Wins: {win_rate_1:.2f}% | Agent2 Wins: {win_rate_2:.2f}% | Draws: {draw_rate:.2f}%")
            
            folder_path = "ModelSaves"
            torch.save(agent1.dqn.state_dict(), folder_path + "/" + "DQN_agent1.pth")
            torch.save(agent2.dqn.state_dict(), folder_path + "/" + "DQN_agent2.pth")

    print("Training complete. Models saved.")
    print(f"Final Win Rates -> Agent1: {wins['agent1']}, Agent2: {wins['agent2']}, Draws: {wins['draws']}")



class OthelloEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)  # 0: empty, 1: player, -1: opponent
        self.board[3, 3], self.board[4, 4] = 1, 1
        self.board[3, 4], self.board[4, 3] = -1, -1
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        return self.board.flatten()  # Convert 8x8 to 1D (64 values)
    
    def make_move(self, x, y):
        if not self.is_valid_move(x, y, self.current_player):
            return False
        self.board[x, y] = self.current_player
        self.flip_pieces(x, y)
        return True
    
    def is_game_over(self):
        return not (self.get_legal_moves(1) or self.get_legal_moves(-1))
    
    def get_legal_moves(self, player=None):
        if player is None:
            player = self.current_player  # Default to the current player
    
        legal_moves = []
        for x in range(8):
            for y in range(8):
                if self.is_valid_move(x, y, player):
                    legal_moves.append(x * 8 + y)  # Convert 2D index to 1D
        return legal_moves
    
    def _captures_in_direction(self, x, y, dx, dy, player):
        nx, ny = x + dx, y + dy
        found_opponent = False
        while 0 <= nx < 8 and 0 <= ny < 8:
            if self.board[nx, ny] == -player:
                found_opponent = True
            elif self.board[nx, ny] == player:
                return found_opponent
            else:
                break
            nx, ny = nx + dx, ny + dy
        return False

    def is_valid_move(self, x, y, player):
        if self.board[x, y] != 0:
            return False
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            i, j, found_opponent = x + dx, y + dy, False
            while 0 <= i < 8 and 0 <= j < 8:
                if self.board[i, j] == -player:
                    found_opponent = True
                elif self.board[i, j] == player and found_opponent:
                    return True
                else:
                    break
                i += dx
                j += dy
        return False
    
    def step(self, action):
        """Apply action, return new state, reward, and done flag."""
        x, y = action
        if not self.make_move(x, y):
            return self.get_state(), -1, False  # Penalty for invalid moves
        
        reward = self.get_reward(self.current_player, action)  # reward for current player
        self.current_player *= -1  # switch after reward is calculated
        done = self.is_game_over()
        return self.get_state(), reward, done

    
    def flip_pieces(self, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            i, j, to_flip = x + dx, y + dy, []
            while 0 <= i < 8 and 0 <= j < 8 and self.board[i, j] == -self.current_player:
                to_flip.append((i, j))
                i += dx
                j += dy
            if 0 <= i < 8 and 0 <= j < 8 and self.board[i, j] == self.current_player:
                for fx, fy in to_flip:
                    self.board[fx, fy] = self.current_player

    def evaluate_board(self):
        return np.sum(self.board) * self.current_player  # Positive if winning, negative if losing

    def get_score(self):
        black_score = np.sum(self.board == -1)
        white_score = np.sum(self.board == 1)
        return black_score, white_score    

    def get_reward(self, player, move):
        """Calculate the reward for a given move in Othello."""
        if self.is_game_over():
            black_score, white_score = self.get_score()
            if (player == -1 and black_score > white_score) or (player == 1 and white_score > black_score):
                return 5  # Winning reward
            else:
                return -5  # Losing penalty

        if move is None:
            return -1  # Passing turn penalty

        x, y = move
        flipped_pieces = self.count_flipped_pieces(x, y, player)

        opponent_moves = self.get_legal_moves(-player)

        opponent_mobility = len(opponent_moves)
        opponent_mobility_penalty = 0.05 * opponent_mobility

        corner_penalty = 3 if any(move in [0, 7, 56, 63] for move in opponent_moves) else 0
        
        return (0.1 * flipped_pieces) - opponent_mobility_penalty - corner_penalty # Reward for own flips, penalty for opponent's mobility and corner moves

    
    def count_flipped_in_direction(self, x, y, dx, dy, player):
        """Count flipped pieces in a given direction."""
        nx, ny = x + dx, y + dy
        count = 0

        while 0 <= nx < 8 and 0 <= ny < 8:
            if self.board[nx, ny] == -player:
                count += 1
            elif self.board[nx, ny] == player:
                return count  # Return only if a valid sandwich is found
            else:
                break
            nx, ny = nx + dx, ny + dy

        return 0  # No pieces flipped in this direction

    def count_flipped_pieces(self, x, y, player):
        """Count the number of opponent pieces flipped after a move."""
        flipped_count = 0
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            flipped_count += self.count_flipped_in_direction(x, y, dx, dy, player)
    
        return flipped_count

env = OthelloEnv()
agent1 = OthelloDQN()  # AI for Black
agent2 = OthelloDQN()  # AI for White

train_ai_vs_ai(agent1, agent2, env, episodes=50000)
