from IPython import display
import time
import gym
from matplotlib import pyplot as plt

#创建月球着陆
env = gym.make('LunarLander-v2', render_mode='rgb_array')

#初始化游戏
state = env.reset()

for i in range(300):
    action = env.action_space.sample()  # randomly choose actions
    state, reward, terminated, truncated, info = env.step(action)
    over = terminated or truncated


    #打印动画
    display.clear_output(wait=True)
    plt.imshow(env.render())
    plt.show()

         #游戏结束了就重置
    if over:
        state, info = env.reset()