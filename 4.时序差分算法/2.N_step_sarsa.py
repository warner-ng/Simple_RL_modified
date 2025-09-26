

def get_state(row, col):
    if row != 3:
        return 'ground'

    if row == 3 and col == 0:
        return 'ground'

    if row == 3 and col == 11:
        return 'terminal'

    return 'trap'



#在一个格子里做一个动作
def move(row, col, action):
    #如果当前已经在陷阱或者终点，则不能执行任何动作
    if get_state(row, col) in ['trap', 'terminal']:
        return row, col, 0

    #↑
    if action == 0:
        row -= 1

    #↓
    if action == 1:
        row += 1

    #←
    if action == 2:
        col -= 1

    #→
    if action == 3:
        col += 1

    #不允许走到地图外面去
    row = max(0, row)
    row = min(3, row)
    col = max(0, col)
    col = min(11, col)

    #是陷阱的话，奖励是-100，否则都是-1
    reward = -1
    if get_state(row, col) == 'trap':
        reward = -100

    return row, col, reward



import numpy as np

#初始化Q值（也就是action_value）,初始化都是0,因为没有任何的知识
Q = np.zeros([4, 12, 4])

"""改动的地方：初始化3个list，这样可以存储n步sarsa种产生的数据，等到n步之后要回溯这些数据？"""
state_list = []
action_list = []
reward_list  =[]

import random



def policy_update(row, col):
    #有小概率选择随机动作 explore
    if random.random() < 0.1:
        return random.choice(range(4))

    #否则选择分数最高的动作 exploit ,而且这里用的仍然是greedy算法
    return Q[row, col].argmax()



def q_value_update_N_step_sarsa(row, col, action, reward, next_row, next_col, next_action):

    TD_target = Q[next_row, next_col, next_action] 

    TD_target_list = []

    # 为什么是反着的：回溯的原因是为了从已知的终点开始逐步计算每个状态的期望回报
    # 因为贝尔曼方程本身是递归的，需要用未来的信息来更新当前状态的价值。
    # 正向计算时，无法直接利用已知的未来信息，
    # 因此使用回溯来逐步更新价值更符合强化学习的学习目标
    for i in reversed(range(5)):
        TD_target = 0.9 * TD_target + reward_list[i]
        TD_target_list.append(TD_target)

    #再返回来
    target_list = list(reversed(target_list))

    action_value_list = []
    for i in range(5):
        row, col = state_list[i]
        action = action_list[i]
        action_value_list.append(Q[row, col, action])

    TD_error_list = []
    for i in range(5):

        TD_error = TD_target_list[i] - action_value_list[i]
        learning_rate = 0.1
        TD_error_list.append(learning_rate * TD_error)

    return TD_error_list
        


    


def train():
    #Epoch 是在机器学习和深度学习中常用的术语，表示训练模型一次
    for epoch in range(1500):

        row = random.choice(range(4))
        col = 0

        action = policy_update(row, col)
        reward_sum = 0

        state_list.clear()
        action_list.clear()
        reward_list.clear()

        while get_state(row, col) not in ['terminal', 'trap']:

            next_row, next_col, reward = move(row, col, action)
            reward_sum = reward_sum + reward

            '''交替进行policy_update和sarsa(value_update)'''
            next_action = policy_update(next_row, next_col)

            # 记录历史数据
            state_list.append([row, col])
            action_list.append(action)
            reward_list.append(reward)





            #积累到5步以后再开始更新参数
            if len(state_list) == 5:
                
                TD_error_list = q_value_update_N_step_sarsa(row, col, action, reward, next_row, next_col, next_action)

                #只更新第一步的分数
                row, col = state_list[0]
                action = action_list[0]
                TD_error = TD_error_list[0]

                Q[row, col, action] += TD_error

                #移除第一步，这样在下一次循环时保持列表是5个元素
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)

            




            # 更新当前位置
            row = next_row
            col = next_col
            action = next_action





        #走到终点以后，更新剩下步数的update
        for i in range(len(state_list)):
            row, col = state_list[i]
            action = action_list[i]
            TD_error = TD_error_list[i]
            Q[row, col, action] += TD_error

        # 顺便打印一个迭代进度
        if epoch % 150 == 0:
            print(epoch, reward_sum)

'''
相比起一步sarsa，n步的sarsa因为要记录n步（举个例子：5步），
所以改变有下面几个：

① 创建几个list来存放历史数据：state action reward action_value TD_target TD_error
② 等待迭代了五次之后，更新（本代码和学习到的算法更新原理不一样，这里的代码又是有问题的，因为它只是用最后1次来更新第一次的，但是中间那些他全部不要了）

批判思考，成功用问题发现了代码是不对的，
代码没有按照n-step-sarsa的更新原则来，
但是数据结构方面是值得学习的（设置了很多list）
'''

train()

#打印所有格子的动作倾向
for row in range(4):
    line = ''
    for col in range(12):
        action = Q[row, col].argmax()
        action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
        line += action
    print(line)







'''
for i in reversed(range(5))
state_list = []
'''