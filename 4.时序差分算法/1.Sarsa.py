

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



import random



def policy_update(row, col):
    #有小概率选择随机动作 explore
    if random.random() < 0.1:
        return random.choice(range(4))

    #否则选择分数最高的动作 exploit ,而且这里用的仍然是greedy算法
    return Q[row, col].argmax()



def q_value_update_sarsa(row, col, action, reward, next_row, next_col, next_action):

    TD_target = 0.9 * Q[next_row, next_col, next_action]  # 0.9 denotes gamma in the bellman equation(decay rate)
    TD_target = TD_target + reward

    q_k = Q[row, col, action]

    learning_rate = 0.1

    TD_error = TD_target - q_k
    
    q_k_plus_1 = q_k + learning_rate * TD_error

    return q_k_plus_1
    


def train():
    #Epoch 是在机器学习和深度学习中常用的术语，表示训练模型一次
    for epoch in range(1500):

        row = random.choice(range(4))
        col = 0
        action = policy_update(row, col)
        reward_sum = 0

        while get_state(row, col) not in ['terminal', 'trap']:

            next_row, next_col, reward = move(row, col, action)
            reward_sum = reward_sum + reward

            '''交替进行policy_update和sarsa(value_update)'''
            # action 是由 policy A 产生的（行为策略）
            next_action = policy_update(next_row, next_col) 
            # TD_Target 也是由 policy A 产生的（目标策略）
            Q[row, col, action] += q_value_update_sarsa(row, col, action, reward, next_row, next_col, next_action)
            '''两个策略一致，故称作on policy'''
            
            # 更新当前位置
            row = next_row
            col = next_col
            action = next_action

        # 顺便打印一个迭代进度
        if epoch % 150 == 0:
            print(epoch, reward_sum)


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
new grammar:

    row = random.choice(range(4))
    从 range(4) 生成的序列中随机选择一个整数，并将其赋值给变量 row

    while get_state(row, col) not in ['terminal', 'trap']:
    非常的自然语言啊

    action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
    把原来action[]数组里面的数字通过这样的“字典”替换成箭头
'''