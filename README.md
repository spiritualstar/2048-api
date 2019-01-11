# learning.py
```
用来学习的程序
首先用keras建立一个有六个卷积层，3个全连层的模型，然后用这个模型展开训练。
  首先用强Agent跑10次游戏，将跑出来的棋盘和方向保存起来，作为训练的数据。
其中，棋盘和方向均采用one_hot编码。利用train_on_batch进行训练，得到一个初始模型。
  然后建立一个大循环，不断地用训练出来地模型跑新的棋盘，然后再用这些棋盘来训练新的模型，即所谓地online。
循环采用while，每次跑50次棋盘，当50次里面跑到48次2048，则退出循环，程序结束。
  循环中训练时，棋盘下一步的方向由新跑出来的模型给出，但用来训练的方向则由强agent给出。
同样对棋盘和方向进行one_hot编码，使用train_on_batch训练。
具体过程可参考代码learning.py，里面有详细注释。
```
# 2048_new2_2048_1011.h5
一个最好的keras模型，该模型最高可达到平均分1011（跑50次）
# MyOwnagent_cif.cif
用自己的agent跑的gif视频结果，差一点跑到2048
# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read [here](EE369.md).
