import random
from directkeys import PressKey, ReleaseKey, Keys
import numpy as np
import time
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from mss import mss
import pyautogui
import cv2


class gameBoard():

    colourTile = {(180, 193, 205): 0,
                  (218, 228, 238): 1,
                  (200, 224, 237): 2,
                  (121, 177, 242): 3,
                  (99, 149, 245): 4,
                  (95, 124, 246): 5,
                  (59, 94, 246): 6,
                  (114, 207, 237): 7,
                  (97, 204, 237): 8,
                  (80, 200, 237): 9,
                  (63, 197, 237): 10,
                  (46, 194, 237): 11}


    def setMonitor(self, x, y, width, height):
        self.monitor = {"left": x, "top": y, "width": width, "height": height}


    def screenshot(self):
        return np.asarray(mss().grab(self.monitor))[:,:,:3]


    def gameBoardArr(img):
        scores = []
        resizedBoard = cv2.resize(img, (100, 100))
        maskedBoard =  ~ cv2.inRange(resizedBoard, (120, 170, 185), (165, 210, 235))
        tileContours = cv2.findContours(maskedBoard, 0, 2)[1]
        tileContours = sorted(tileContours, key=lambda z: (cv2.boundingRect(z)[1] // 25 , cv2.boundingRect(z)[0] // 25))
        for tile in tileContours:
            tileX, tileY, tileWidth, tileHeight = cv2.boundingRect(tile)
            if (tileWidth > 15 and tileHeight > 15):
                roi = resizedBoard[tileY:tileY+tileHeight, tileX:tileX+tileWidth]
                colour = tuple(roi[3][3])
                for col in gameBoard.colourTile:
                    if abs(colour[0] - col[0]) <= 2 and abs(colour[1] - col[1]) <= 2 and abs(colour[2] - col[2]) <= 2:
                        scores.append(gameBoard.colourTile[col])
        if len(scores) != 16:
            return None
        return np.reshape(scores, [1, 16])


    def gameBoardScore(board):
        counter = 0
        for tile in board[0]:
            if tile == 0:
                counter += 1
        return counter


    def displayBoard(board):
        rowPlaced = 0
        for i in range(len(board)):
            print(board[i], end = (5 - len(str(board[i]))) * " ")
            rowPlaced += 1
            if rowPlaced % 4 == 0:
                rowPlaced = 0
                print("\n")
        print("\n\n\n")


class Agent():

    def __init__(self):
        self.window = gameBoard()
        self.window.setMonitor(425, 135, 500, 500)
        self.validKeys = list(Keys)[:4]
        self.blankTiles = None


    def get_action(self):
        return random.randint(0, len(self.validKeys) - 1)


    def reset(self, games):
        time.sleep(1.5)
        if games % 50 == 0:
            input("Game has ended, press enter when new game is ready")
        else:
            pyautogui.moveTo(640, 380, 0.2)
            pyautogui.click()
        time.sleep(3)
        sct = self.window.screenshot()
        state = gameBoard.gameBoardArr(sct)
        self.blankTiles = gameBoard.gameBoardScore(state)
        return state


    def step(self, prevState, action):
        
        # Perform the action
        PressKey(Keys[self.validKeys[action]])
        ReleaseKey(Keys[self.validKeys[action]])

        # Get the new state
        sct = self.window.screenshot()
        state = gameBoard.gameBoardArr(sct)

        # Check if the game is over
        done = True
        if state is not None:
            done = False
        else:
            state = prevState

        # Get the new reward
        reward = 0
        if not done:
            oldScore = self.blankTiles
            self.blankTiles = gameBoard.gameBoardScore(state)
            reward = self.blankTiles - oldScore

        if np.array_equal(prevState, state):
            reward -= 2

        
        return state, reward, done


class DQN2048Solver():
    
    def __init__(self, n_episodes=1000, gamma=1.0, epsilon=1.0, epsilon_min=0.01,
                 epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64):

        self.timesRun = 0
        self.memory = deque(maxlen=100000)
        self.env = Agent()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        # Init model
        self.model = Sequential()
        self.model.add(Dense(48, input_dim=16, activation='relu'))
        self.model.add(Dense(96, activation='relu'))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(4, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def choose_action(self, state, epsilon):
        return self.env.get_action() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))


    def get_epsilon(self):
        return max(self.epsilon_min, self.epsilon)


    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def run(self):
        scores = deque(maxlen=5)

        for e in range(self.n_episodes):
            state = self.env.reset(e)
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon())
                next_state, reward, done = self.env.step(state, action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                time.sleep(0.3)
                i += 1
                
            scores.append(i)
            mean_score = np.mean(scores)
            if e % 5 == 0:
                print('[Episode {}] - Mean survival time over last 5 episodes was {} ticks. Epsilon: {}'.format(e, mean_score, self.get_epsilon()))

            self.replay(self.batch_size)
            self.timesRun += 1
            
        return e


    def play(self):
        state = self.env.reset(0)
        done = False
        while not done:
            actions = np.sort(self.model.predict(state))
            idx = 3
            while idx >= 0:
                next_state, reward, done = self.env.step(state, idx)
                time.sleep(1)
                if np.array_equal(state, next_state):
                    idx -= 1
                else:
                    break

            
            state = next_state


    def save(self, modelName):

        # Save the weights of the model
        self.model.save_weights(modelName)

        # Save the items in memory
        with open("memory.txt", "w") as file:
            for item in self.memory:
                file.write(str(item) + "\n")

        # Print current epsilon
        print(self.epsilon)

    def load(self, modelName, epsilon, loadMemory = True):
        self.epsilon = epsilon
        self.model.load_weights(modelName)
        if loadMemory:
            array = np.array
            with open("memory.txt", "r") as file:
                for line in file:
                    batch = eval(line)
                    agent.memory.append(batch)


if __name__ == '__main__':
    agent = DQN2048Solver(n_episodes = 100)
    agent.load("recent5.h5", 0.5344229416520513)
    agent.run()
    # agent.play()
    input()
