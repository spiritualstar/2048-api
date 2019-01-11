import numpy as np
import keras
from keras.models import load_model

model1=load_model('2048_new2_2048.h5')
#my_model32=load_model('2048_32_1.h5')
#my_model64=load_model('2048_64_1.h5')
#my_model128=load_model('2048_128.h5')
#my_model256=load_model('2048_256.h5')
#my_model512=load_model('2048_512.h5')
#my_model1024=load_model('2048_1024.h5')
class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
 
    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyOwnAgent(Agent):
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
    def step(self):
       
        x_train=np.array(self.game.board)
        x=x_train
        x=np.log2(x+1)
        x=np.trunc(x)
        x = keras.utils.to_categorical(x,12)
        x = x.reshape(1, 4, 4, 12)
        pred=model1.predict(x,batch_size=128)
        r=pred[0]
        r1=r.tolist()
        direction2=r1.index(max(r1))
        return direction2            