
import gym
from matplotlib import pyplot as plt


#创建环境
#is_slippery控制会不会滑
#map_name决定地图的尺寸,还可以取8x8
#desc决定地形
env = gym.make('FrozenLake-v1',
               render_mode='rgb_array',
               is_slippery=False,
               map_name='4x4',
               desc=['SFFF', 'FHFH', 'FFFH', 'HFFG'])
env.reset()

#解封装才能访问状态转移矩阵P
env = env.unwrapped


#打印游戏
def show():
    plt.imshow(env.render())
    plt.show()


show()
#查看冰湖这个游戏的状态列表
#一共4*4=16个状态
#每个状态下都可以执行4个动作
#每个动作执行完，都有概率得到3个结果
#(0.3333333333333333, 0, 0.0, False)这个数据结构表示(概率，下个状态，奖励，是否结束)
len(env.P), env.P[0]
import numpy as np

#初始化每个格子的价值
values = np.zeros(16)

#初始化每个格子下采用动作的概率
pi = np.ones([16, 4]) * 0.25

#两个算法都是可以的
algorithm = '策略迭代'
algorithm = '价值迭代'

values, pi
#计算qsa
def get_qsa(state, action):
    value = 0.0

    #每个动作都会有三个不同的结果，这里要按概率把他们加权求和
    for prop, next_state, reward, over in env.P[state][action]:

        #计算下个状态的分数,取values当中记录的分数,再打个折扣
        next_value = values[next_state] * 0.9

        #如果下个状态是终点或者陷阱，则下个状态的分数是0
        if over:
            next_value = 0

        #动作的分数就是reward,和下个状态的分数相加就是最终的分数了
        next_value += reward

        #因为下个状态是概率出现了,所以这里要乘以概率
        next_value *= prop

        value += next_value

    return value


get_qsa(0, 0)
#策略评估
def get_values():
    #初始化一个新的values,重新评估所有格子的分数
    new_values = np.zeros([16])

    #遍历所有格子
    for state in range(16):

        #计算当前格子4个动作分别的分数
        action_value = np.zeros(4)

        #遍历所有动作
        for action in range(4):
            action_value[action] = get_qsa(state, action)

        if algorithm == '策略迭代':
            #每个动作的分数和它的概率相乘
            action_value *= pi[state]
            #最后这个格子的分数,等于该格子下所有动作的分数求和
            new_values[state] = action_value.sum()

        if algorithm == '价值迭代':
            #求每一个格子的分数，等于该格子下所有动作的最大分数
            new_values[state] = action_value.max()

    return new_values


get_values()
#策略提升
def get_pi():
    #重新初始化每个格子下采用动作的概率,重新评估
    new_pi = np.zeros([16, 4])

    #遍历所有格子
    for state in range(16):

        #计算当前格子4个动作分别的分数
        action_value = np.zeros(4)

        #遍历所有动作
        for action in range(4):
            action_value[action] = get_qsa(state, action)

        #计算当前state下，达到最大分数的动作有几个
        count = (action_value == action_value.max()).sum()

        #让这些动作均分概率
        for action in range(4):
            if action_value[action] == action_value.max():
                new_pi[state, action] = 1 / count
            else:
                new_pi[state, action] = 0

    return new_pi


get_pi()
#循环迭代策略评估和策略提升，寻找最优解
for _ in range(10):
    for _ in range(100):
        values = get_values()
    pi = get_pi()

values, pi
#打印每个格子的策略
def print_pi():
    #遍历所有格子
    for row in range(4):

        line = ''

        for col in range(4):
            state = row * 4 + col

            if (row == 1 and col == 1) or (row == 1 and col == 3) or (
                    row == 2 and col == 3) or (row == 3 and col == 0):
                line += '○'
                continue

            if row == 3 and col == 3:
                line += '❤'
                continue

            #line += '□'
            line += '←↓→↑'[pi[state].argmax()]

        print(line)


print_pi()
from IPython import display
import time


def play():
    env.reset()

    #起点在0
    index = 0

    #最多玩N步
    for i in range(200):
        #选择一个动作
        action = np.random.choice(np.arange(4), size=1, p=pi[index])[0]

        #执行动作
        index, reward, terminated, truncated, _ = env.step(action)

        #打印动画
        display.clear_output(wait=True)
        time.sleep(0.1)
        show()

        #获取当前状态，如果状态是终点或者掉陷阱则终止
        if terminated or truncated:
            break

    print(i)


play()