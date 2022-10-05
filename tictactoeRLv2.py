from matplotlib.style import available
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env

class gameEnv(gym.Env):
    def __init__(self):
        super(gameEnv,self).__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete(np.array([3]* 9))
        self.board = np.array([1,1,1,1,1,1,1,1,1]) 
        self.number = 2 #X player
        self.emptySpaces = []
        self.opponentNumber = 0

    def render(self, mode = "console"):
        reshapedBoard = np.reshape(self.board,(3,3))
        for i in range(3):
            print("|", end = "")
            for j in range(3):
                if reshapedBoard[i][j] == 1:
                    print("",end = " |")
                if reshapedBoard[i][j] == 0:
                    print("", end = "O|")
                if reshapedBoard[i][j] == 2:
                    print("", end = "X|")
            print("")
                

    def reset(self):
        self.board = np.array([1,1,1,1,1,1,1,1,1])
        return (self.board)

    def checkWinner(self, board):
        for i in range(3): #row and column 
            if sum(board[3*i:(i+1)*3]) == 3:
                return 1
            if sum(board[3*i:(i+1)*3]) == -3:
                return -1
            if (board[i] + board[i+3] + board[i+6]) == 3:
                return 1
            if (board[i] + board[i+3] + board[i+6]) == -3:
                return -1
        
        #diagonals
        if (board[0] + board[4] + board[8]) == 3:
            return 1
        if (board[0] + board[4] + board[8]) == -3:
            return -1
        if (board[2] + board[4] + board[6]) == 3:
            return 1
        if (board[2] + board[4] + board[6]) == -3:
            return -1
        
        #tie
        emptySpaces = list(self.emptySpaces)
        for j in board:
            if j == 0:
                emptySpaces.append(j)
        emptySpaces.pop(0)
        if not emptySpaces:
            return 0

    def returnEmptySpaces(self):
        emptySpaces = list(self.emptySpaces)
        for j in self.board:
            if j == 1:
                emptySpaces.append(j)

    def step(self,action):
        self.emptySpaces = np.where(self.board == 1)
        self.board[action] = self.number
        boardCopy = self.board - 1

        if(action in np.asarray(self.emptySpaces)):
            if(self.checkWinner(boardCopy) == 1):
                return self.board, 1, True, {}
            elif (self.checkWinner(boardCopy) == 0):
                return self.board, 0.2, True, {}
            
            else:
                self.emptySpaces = np.where(self.board == 1)
                array1 = np.asarray(self.emptySpaces)
                randomRow1 = [0]
                if array1.size != 0: #bot
                    opponentAction = np.random.choice(array1[randomRow1[0], :])
                    self.board[opponentAction] = self.opponentNumber
                if (self.checkWinner(boardCopy) == -1):
                    return self.board, -1, True, {}
                elif (self.checkWinner(boardCopy) == 0):
                    return self.board, 0.2, True, {}
                else:
                    return self.board, 0, False, {'opponentAction' : opponentAction}
        
        else: #invalid move
            return self.board, -10, True, {}

env = gameEnv()
check_env(env)
env = make_vec_env(lambda: env, n_envs = 1)
model = PPO('MlpPolicy', env, verbose=1).learn(total_timesteps = 120000)

#playing against bot; win percentage
wins = 0
losses = 0
misfires = 0
ties = 0
print("Training complete! Now playing...")
for i in range(5000):
    obs = env.reset()
    finished = False
    while(not finished):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if(done):
            finished = True
            if reward == 1:
                wins += 1
            if reward == -1:
                losses += 1
            if reward == 0.2:
                ties += 1
            if reward == -10:
                misfires += 1
    
print('Wins: ' + str(wins))
print('Losses: ' + str(losses))
print('Invalid action attempts: ' + str(misfires))
print('Ties: ' + str(ties))
print('Winrate: ' + str((wins/5000)))

#sample game
# obs = env.reset()
# finished = False
# while(not finished):
#     env.render()
#     action, _ = model.predict(obs)
#     print("Model chooses to move " + str(action))
#     obs, reward, done, info = env.step(action)
#     if reward == 1: 
#         print("Model Wins!")
#         finished = True
#     if reward == -1: 
#         print("Opponent Wins!")
#         finished = True
#     if reward == 0.2:
#         print("Tie!")
#         finished = True
#     if reward == -10:
#         print("Model chose invalid move!")
#         finished = True
#     print("Opponent chooses:" + str(info[0]['opponentAction']))


print("Winrate computed! Now user can play against agent")

#game against human opponent

stillPlay = True
while(stillPlay):
    finished = False
    board = np.array([1,1,1,1,1,1,1,1,1]) 
    x = 0
    while(not finished):
        reshapedBoard = np.reshape(board,(3,3))
        action,_ = model.predict(board)
        print("Model chooses to move: " + str(action))
        board[action] = 2
        emptySpaces = np.where(board == 1)
        emptySpacesArray = np.asarray(emptySpaces)

        for i in range(3): #render
            print("|", end = "")
            for j in range(3):
                if reshapedBoard[i][j] == 1:
                    print("",end = " |")
                if reshapedBoard[i][j] == 0:
                    print("", end = "O|")
                if reshapedBoard[i][j] == 2:
                    print("", end = "X|")
            print("")

        boardCopy = board - 1

        for i in range(3): #row and column 
            if sum(boardCopy[3*i:(i+1)*3]) == 3:
                print("Model Wins!")
                finished = True
            if sum(boardCopy[3*i:(i+1)*3]) == -3:
                print("Player Wins!")
                finished = True
            if (boardCopy[i] + boardCopy[i+3] + boardCopy[i+6]) == 3:
                print("Model Wins!")
                finished = True
            if (boardCopy[i] + boardCopy[i+3] + boardCopy[i+6]) == -3:
                print("Player Wins!")
                finished = True
            
            #diagonals
            if (boardCopy[0] + boardCopy[4] + boardCopy[8]) == 3:
                print("Model Wins!")
                finished = True
                
            if (boardCopy[0] + boardCopy[4] + boardCopy[8]) == -3:
                print("Player Wins!")
                finished = True
                
            if (boardCopy[2] + boardCopy[4] + boardCopy[6]) == 3:
                print("Model Wins!")
                finished = True
                
            if (boardCopy[2] + boardCopy[4] + boardCopy[6]) == -3:
                print("Player Wins!")
                finished = True
                
            
            #tie
            emptySpacesList = list(emptySpaces)
            for j in boardCopy:
                if j == 0:
                    emptySpacesList.append(j)
            emptySpacesList.pop(0)
            if not emptySpacesList:
                print("Tie!")
                finished = True
                
            if finished:
                break
                
        if finished:
            break
        
        while(x == 0):
            val = input("Enter your value: ")
            if int(val) in (emptySpacesArray[0, :]):
                board[int(val)] = 0
                x = 1
            else:
                print("Please enter in a valid space")

        x = 0
        for i in range(3): #render
            print("|", end = "")
            for j in range(3):
                if reshapedBoard[i][j] == 1:
                    print("",end = " |")
                if reshapedBoard[i][j] == 0:
                    print("", end = "O|")
                if reshapedBoard[i][j] == 2:
                    print("", end = "X|")
            print("")

        boardCopy = board - 1
        for i in range(3): #row and column 
            if sum(boardCopy[3*i:(i+1)*3]) == 3:
                print("Model Wins!")
                finished = True
            if sum(boardCopy[3*i:(i+1)*3]) == -3:
                print("Player Wins!")
                finished = True
            if (boardCopy[i] + boardCopy[i+3] + boardCopy[i+6]) == 3:
                print("Model Wins!")
                finished = True
            if (boardCopy[i] + boardCopy[i+3] + boardCopy[i+6]) == -3:
                print("Player Wins!")
                finished = True
            
            #diagonals
            if (boardCopy[0] + boardCopy[4] + boardCopy[8]) == 3:
                print("Model Wins!")
                finished = True
            if (boardCopy[0] + boardCopy[4] + boardCopy[8]) == -3:
                print("Player Wins!")
                finished = Trueye
            if (boardCopy[2] + boardCopy[4] + boardCopy[6]) == 3:
                print("Model Wins!")
                finished = True
            if (boardCopy[2] + boardCopy[4] + boardCopy[6]) == -3:
                print("Player Wins!")
                finished = True
            
            #tie
            emptySpacesList = list(emptySpaces)
            for j in boardCopy:
                if j == 0:
                    emptySpacesList.append(j)
            emptySpacesList.pop(0)
            if not emptySpacesList:
                print("Tie!")
                finished = True
            
            if finished:
                break

    repeat = input("Do you want to play again? ")
    if repeat.lower() == "yes":
        stillPlay = True
    else:
        stillPlay = False
