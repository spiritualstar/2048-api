#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:36:06 2018

@author: spiritualstar
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras

BATCH_SIZE = 128
NUM_CLASSES = 4
NUM_EPOCHS = 20

from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import ExpectiMaxAgent,MyOwnAgent
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split


display1 = Display()
display2 = IPythonDisplay()
input_shape = (4, 4, 12)
# define the model
model=Sequential()
model.add(Conv2D(filters= 128, kernel_size=(4,1),kernel_initializer='he_uniform', padding='Same', activation='relu',input_shape=input_shape))  # 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小 4*4*12
model.add(Conv2D(filters= 128, kernel_size=(1,4),kernel_initializer='he_uniform', padding='Same', activation='relu')) 

#model.add(Dropout(0.25)) 
model.add(Conv2D(filters= 128, kernel_size=(1,1),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(2,2),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(3,3),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(4,4),kernel_initializer='he_uniform', padding='Same', activation='relu'))  

model.add(Flatten()) 
model.add(BatchNormalization())
model.add(Dense(256, kernel_initializer='he_uniform',activation='relu')) 
model.add(BatchNormalization())
model.add(Dense(128, kernel_initializer='he_uniform',activation='relu')) 
model.add(Dense(4, activation='softmax')) 

#建立一个六个卷积层，三个全连层的模型

model.summary()
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])
image=[]
label=[]
for i in range(0,10):
    game = Game(4, score_to_win=2048, random=False)
    agent = ExpectiMaxAgent(game, display=display1)
    
    while game.end==False:
        
        direction=agent.step()
        image.append(game.board)
        label.append(direction)
        game.move(direction)
       
    display1.display(game)
#用强Agent跑10次游戏，将跑出来的棋盘和方向记录在数组中
x_train=np.array(image)
y_train=np.array(label)


x_train=np.log2(x_train+1)
x_train=np.trunc(x_train)
x_train = keras.utils.to_categorical(x_train, 12)

#对棋盘进行12位one-hot编码

print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
#对方向标签进行4位one-hot编码
model.train_on_batch(x_train, y_train)       
#用train_on_batch进行训练
# evaluate
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100,score_train[1]*100))

model.save('2048_new2_2048.h5')
#实时记录模型，保存下来
image=[]
label=[]
#清空棋盘和方向的矩阵
count1=0
while count1<1:
    count=0
    for j in range(0,50):
        game = Game(4, score_to_win=2048, random=False)
        agent1 = ExpectiMaxAgent(game, display=display1)
        #agent2 = MyOwnAgent(game, display=display1)
        while game.end==False:
            direction1=agent1.step()#用强Agent判断方向
            
            x=np.array(game.board)
            
            x=np.log2(x+1)
            x=np.trunc(x)
            x = keras.utils.to_categorical(x, 12)
            x = x.reshape(1, 4, 4, 12)
            pred=model.predict(x,batch_size=128)
            r=pred[0]
            r1=r.tolist()
            direction2=r1.index(max(r1))
            #用新跑出来的模型判断方向
            image.append(game.board)
            label.append(direction1)#将用强Agent跑出来的方向作为训练的标签
            game.move(direction2)#用新模型判断的方向执行下一步
            
        display1.display(game)
        x=np.array(game.board)
        if np.amax(x)==2048:
            count+=1
        #如果本次游戏跑出2048，count+1
    if count>=48:
        break
    #如果50次游戏跑出48次2048，则循环结束
    x_train=np.array(image)
    y_train=np.array(label)
    x_train=np.log2(x_train+1)
    x_train=np.trunc(x_train)
    x_train = keras.utils.to_categorical(x_train, 12)
    #对棋盘进行12位one-hot编码


    print(x_train.shape)
   
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    #对方向标签进行4位one-hot编码
    
    model.train_on_batch(x_train, y_train)
    #用train_on_batch进行训练
    # evaluate
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100,score_train[1]*100))
    
    model.save('2048_new2_2048.h5')
    #实时记录模型，保存下来
    image=[]
    label=[]
    #清空棋盘和方向的矩阵
    print(i)
    print(count)
    
    