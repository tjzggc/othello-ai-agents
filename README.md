# Othello AI Agents

## Requirements
There is a file included Requirements.txt that has all of the libraries that are needed for this project. It also includes what each library was used for in each of the files.

## OthelloBackPropNN

### Initialization
Beginning of this file is initial conversion files to deal with training.
Then convert the dataset letter and number positions to a single integer from 0 to 63. 
Functions to check position passed to it is valid. update the board based on a move, and make a move from training.

### Logging and Model Setup
Create a log of moves since a player can move twice if the other player does not have a valid move.
To create the model I used a Xavier initializtion for the weights and bias. Then there are lines you can comment in or out to get a 3 layer or 4 layer network.
The weights are chosen the same for almost all of my networks. It just needs to have 64 positions as it's output.
It is currently setup for a four layer network with back propagation.

### Training
Reads the data logs. We find the winner and separate all of the moves that the winner made to train with. 
The move data is put into pytorch tensors that then go into a dataset and dataloader.
Split the dataset into 64 sets and run through it 25 times in training. 


## OthelloDQN

### Initialization
Model is setup with 3 layers and the same situation as the other file where it needs the weights to finish at 64 coming out.
Model is setup with a target model as well for training. Using Adam optimizer and mean squared error for loss. Model is also setup with memory buffer for training and epsilon decay. 
choose_move uses either epsilon greedy for exploration or the max q value when making a choice.

### OthelloEnv
Class is used to setup the environment for the model to train on. It has many functions that are similar to what was in the BackProp file.
Step and getReward are  the most important functions in this class.
Step - applies the action from make_move, then gets the new state, the reward for making the move, and finds out if the game is over.
getReward - Hands out the reward to the model. In this case +/-5 for win/loss. -1 for a turn being skipped. +0.1 for any pieces the models flips.
-3 for giving an opponent the ability to take a corner spot.
-0.05 for the amount of spots the model gives the opponent the ability to play.

### Training
While playing the game in train_ai_vs_ai() it will switch from player to player in the environment class. 
The AI will choose a move, collecting information from the step function adn storing that information.
Then using that information it will train in self play.
The training went for 50,000 games, and every 1000 games the two AI agents playing were saved.
At the end the win rate was calculated and the version with the best rate was used in testing.


## OthelloPlayHuman
This class is used to play against an AI that is created by loading the AI with a .pth file at line 24.
The class was created using open AI pygame. I used chatGPT to help me create the class so that I didn't have to create the board for testing.
The file is currently setup to use the 3 layer model at line 8.
The model will choose the best move out of the valid moves that it is given when looking at the current board on it's turn.
When running the program will check whose turn it is, take the players input if it is valid and redraw the board.
When it is the AI's turn the AI will choose the best move of the valid moves, apply that move, and draw the board.
When the game is over the winner is saved with the counts for each of the players in a csv file for logging.


## OthelloPlayOtherAILooped

### Functions Explained
This class was used as the main testing ground for the AI agents that were created.
At the top of the file both versions of the model can be loaded separately using DQN or OthelloNet using the .pth file on line 52 and changing the call on line 50.
Next I setup selenium to open the rust web app at https://othello-rust.web.app/.
The next functions are used to parse the board as well as check for valid moves. The squares are setup 0 to 63 from the top left to the right and then down.
When choosing a move the model takes in the state and choosing the best move of the available. The move has to then be changed to a single value instead of row and column.
When checking for game over it looks for specific text the web app gives and then tallies the board. This data is then saved into a csv file. 

### Main Loop
You can change the number of games being played at line 171. All of the testing was done on Easy difficulty so there is no changing the web app difficulty right now.
The main loop checks for a game over. If not one then it gets the board and lets our AI choose and make a move.
There are sleep() calls throughout the file so that our AI does not step on the web app.

## OthelloDQNRustTrain

### Functions and File Changes
This file was only used to train the Deep Q Network that was trained in OthelloDQN. It uses the .pth file from that file and continues to train it using the web app.
This file is a mix of OthelloDQN and OthelloPlayOtherAILooped. The model is setup the same using the same loss function, epsilon decay, and choose move function. 
When interacting with the board it gets the index from 0 to 63 converts to row and column for the model and then converts back to an integer for selecting the move.
The reward function is setup in the same way as it was in OthelloDQN.

### Main Loop
I could not get the game to change difficulty so the amount of games is set to 500.
I used the blocks from line 276 to 297 to change difficulty after runnning the game 500 times through. All blocks commented out for Easy and the commenting in a block to train the other difficulties.
After training once I also changed line 39 to use model_path_update so that it was using the most up to date model when training on the harder difficulties. 
The main loop is the same as OthelloPlayOtherAILooped except with training added.

#### Main Loop Training
After each 100 game is finished the target network was updated and a new model was saved and loaded to be used.
After each game the epsilon was decayed.
While a game was happening and it is the AI Agent's turn the agent would make a move and get reward and states for before and after putting them into a replay buffer. 
After the agent makes 4 moves it will train using a random selection from the replay buffer.
After all 500 games are played there will be a csv file with the wins and losses and the model will save one more time.


## Steps
1. Trained AI Agent using both the OthelloDQN and OthelloBackPropNN files to get .pth files and then OthelloDQNRustTrain.
2. Tested by playing against myself with OthelloPlayHuman.
3. Tested against Rust Web App with OthelloPlayOtherAILooped.


