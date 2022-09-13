import numpy as np
import random
from tqdm import tqdm

class gameEnv():

    def __init__(self):
        self.board = np.zeros((3,3))
        self.states = []
        self.xstate_values = {}
        self.ystate_values = {}

    def returnBoard(self):
        #print(self.board)
        return(self.board)    

    def editBoard(self, row, column, playerNumber):
        if((int(row),int(column)) in self.availablePositions()):
            self.board[row][column] = playerNumber
            #print(self.board)
        else:
            return 0 #if spot is already taken

    def checkWinner(self):
        for i in range(3): #row and column
            if sum(self.board[i,:]) == 3:
                return 1
            if sum(self.board[i,:]) == -3:
                return -1
            if sum(self.board[:,i]) == 3:
                return 1
            if sum(self.board[:,i]) == -3:
                return -1

        #diagonal
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == 3:
            return 1
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == -3:
            return -1
        if self.board[0][2] + self.board[1][1] + self.board[2][0] == 3:
            return 1
        if self.board[0][2] + self.board[1][1] + self.board[2][0] == -3:
            return -1
        
        #tie
        if len(self.availablePositions()) == 0:
            return 0

    def returnHash(self):
        hash = str(self.board.reshape(9))
        return hash
    
    def addState(self,array): #adds a state to an array
        self.states.append(array)
    
    def returnStates(self): #returns the array; accessed when backpropogating
        #print(str(self.states))
        return self.states

    def clearStates(self):
        emptyList = []
        self.states = emptyList

    def updateXReward(self,states,reward): #updates the different values for X Agent
       for i in reversed(states):
            if self.xstate_values.get(i) is None:
                self.xstate_values[i] = 0           
            else:
                self.xstate_values[i] += 0.2*(0.9*reward - self.xstate_values[i])
            reward = self.xstate_values[i]

    def updateYReward(self,states,reward): #updates the different values in the Y Agent
       for i in reversed(states):
            if self.ystate_values.get(i) is None:
                self.ystate_values[i] = 0           
            else:
                self.ystate_values[i] += 0.2*(0.9*reward - self.ystate_values[i])
            reward = self.ystate_values[i]

    def returnXStateValue(self):
        return self.xstate_values

    def returnYStateValue(self):
        return self.ystate_values

    def clearBoard(self):
        newBoard = np.zeros((3,3))
        self.board = newBoard

    def availablePositions(self): #returns array of available positions on the board
        emptySpaces = []
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == 0:
                    emptySpaces.append((x,y))
        return(emptySpaces)

#HUMAN PLAYER IF WANT TO PLAY
class Player: 
    def __init__(self, humanPlayer):
        self.humanPlayer = humanPlayer
        self.playerNumber = 0

    def chooseNumber(self): #assigns player number
        x = 1
        while x > 0:
            val = input("Player 1 choose X or O: ")
            print(val)
            if val == "X" or val == "x":
                self.playerNumber = 1
                x = -1
            elif val == "O" or val == "o":
                self.playerNumber = -1
                x = -1
            else:
                print('"Please enter in "X" or "O"')

    def assignNumber(self, number): #assigns player number
        self.playerNumber = number

    def returnNumber(self): #returns player number
        return(self.playerNumber)

    def move(self):
        if self.playerNumber == 1: #X player
            row = input("X Player enter row: ")
            column = input("X Player enter column: ")
        if self.playerNumber == -1: #Y Player
            row = input("O Player enter row: ")
            column = input("O Player enter column: ")
        return row, column

class Agent: 
    def __init__(self, agentNumber):
        self.states = []
        self.agentNumber = agentNumber
        self.state_values = {}

    def assignNumber(self, number):
        self.agentNumber = number

    def returnNumber(self):
        return self.agentNumber

    def chooseAction(self, availablePositions, valueDict, currentBoard, agentNumber, greed):
        highestList = []
        if np.random.uniform(0,1) <= greed:
            randomPosition = np.random.choice(len(availablePositions))
            action = availablePositions[randomPosition]
            return action
        else:
            maxValue = -1
            for i in availablePositions: #format of availablePositions is not same as valueDict
                currentBoardCopy = currentBoard.copy()
                currentBoardCopy[i] = agentNumber
                currentBoardCopyHash = str(currentBoardCopy.reshape(9))
                if valueDict.get(currentBoardCopyHash) is None:
                    value = 0
                else:
                    value = valueDict.get(currentBoardCopyHash)
                if value > maxValue:
                    maxValue = value
                    highestList.clear()
                    highestList.append(i)
                elif value == maxValue:
                    highestList.append(i)
                action = random.choice(highestList)
        return action

#TIC-TAC-TOE BOT MAKING RANDOM CHOICES           
class randomPlayer():
    def __init__(self, number):
        self.playerNumber = number
        
    def move(self, availablePositions):
        action = random.choice(availablePositions)
        return action

gameEnd = False
playerturn = 1
game = gameEnv()
agent1 = Agent(0)
agent1.assignNumber(1)
bot = randomPlayer(-1)

#training
print("training...")
for i in tqdm(range(50000)):
    while gameEnd == False:
        if playerturn == 1:
            row, column = agent1.chooseAction(game.availablePositions(),game.returnXStateValue(),game.returnBoard(),agent1.returnNumber(),0.26)
            if(game.editBoard(int(row),int(column),agent1.returnNumber())) == 0:
                pass
            else:
                playerturn = -1
                game.addState(game.returnHash())
                if game.checkWinner() == 1:
                    game.updateXReward(game.returnStates(),1)
                    game.clearBoard()
                    game.clearStates() 
                    break
                if game.checkWinner() == 0:
                    game.updateXReward(game.returnStates(),0.15)
                    game.clearBoard()
                    game.clearStates()
                    break
                if game.checkWinner() == -1:
                    game.updateXReward(game.returnStates(),0)
                    game.clearBoard()
                    game.clearStates()
                    break
        if playerturn == -1:
            row, column = bot.move(game.availablePositions())
            if(game.editBoard(int(row),int(column),bot.playerNumber)) == 0:
                pass
            else:
                playerturn = 1
                game.addState(game.returnHash())
                if game.checkWinner() == -1:
                    game.updateYReward(game.returnStates(),1)
                    game.clearBoard()
                    game.clearStates()
                    break
                if game.checkWinner() == 0:
                    game.updateYReward(game.returnStates(),0.15)
                    game.clearBoard()
                    game.clearStates()
                    break
                if game.checkWinner() == 1:
                    game.updateYReward(game.returnStates(),0)
                    game.clearBoard()
                    game.clearStates()
                    break


print("training complete! now playing...")
wins = 0
losses = 0
ties = 0
for i in range(2000):    
    while gameEnd == False:
        if playerturn == 1:
            row, column = agent1.chooseAction(game.availablePositions(),game.returnXStateValue(),game.returnBoard(),agent1.returnNumber(),0)
            if(game.editBoard(int(row),int(column),agent1.returnNumber())) == 0:
                print("Please choose an open space")
            else:
                playerturn = -1
                if game.checkWinner() == 1:
                    wins += 1
                    game.clearBoard()
                    game.clearStates() 
                    break
                if game.checkWinner() == 0:
                    ties += 1
                    game.clearBoard()
                    game.clearStates()
                    break
        if playerturn == -1:
            row, column = bot.move(game.availablePositions())
            if(game.editBoard(int(row),int(column),bot.playerNumber)) == 0:
                print("Please choose an open space")
            else:
                playerturn = 1
                if game.checkWinner() == -1:
                    losses += 1
                    game.clearBoard()
                    game.clearStates()
                    break
                if game.checkWinner() == 0:
                    ties += 1
                    game.clearBoard()
                    game.clearStates()
                    break

#State-Value Dict
#print(game.returnXStateValue)

print('Wins: ' + str(wins))
print('Losses: ' + str(losses))
print('Ties: ' + str(ties))
print('Winrate: ' + str((wins/2000)))