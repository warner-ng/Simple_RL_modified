import numpy as np



def get_state(row,col):  # initialize the map
    if row != 3:
        return 'ground'

    if row == 3 and col == 0:
        return 'ground'

    if row == 3 and col == 11:
        return 'terminal'

    return 'trap'




def move(row, col, action):   # action and its resultk
    if get_state(row, col) in ['trap', 'terminal']:
        return row, col, 0  #  get a NONE feedback

    if action == 0:  # go up
        row -=1
    
    if action == 1:  # go down
        row +=1

    if action == 2:  # go left
        col -=1

    if action == 3:  # go right
        col +=1


    row = max(0,row)
    row = min(3,row)
    col = max(0,col)
    col = min(11,col)

    reward = -1  # no matter what, your rewards gets punished, cause 
                 # we don't want it to walk too long
    if get_state(row,col) == 'trap':
        reward = -100

    return row, col, reward




def get_qsa(row, col, action,): # get your action value

    next_row, next_col, reward = move(row, col, action)

    value = values[next_row, next_col] * 0.9  # a recurrent calculation

    if get_state(next_row, next_col) in ['trap','terminal']:
        value = 0

    return value + reward




'''important part 1'''
def policy_evaluation():

    new_values = np.zeros([4,12])

    for row in range(4):
        for col in range(12):

            action_value = np.zeros(4)  # initialize a policy /Pi_0

            for action in range(4):
                
                action_value[action] = get_qsa(row, col, action)

            new_values[row, col] = (action_value * pi[row, col]).sum() # the state value
                                                      # is calculated by
                                                      # adding every action value * its policy (probability) 
            
    return new_values



'''important part 2'''
def policy_update():

    new_policy = np.zeros([4,12,4])

    for row in range(4):
        for col in range(12):

            action_value = np.zeros(4)

            for action in range(4):
                action_value[action] = get_qsa(row, col, action)

            count = (action_value == action_value.max()).sum()  # in case there are many actions with a same value

            for action in range(4):
                if action_value[action] == action_value.max():
                    new_policy[row, col, action] = 1 / count  # Greedy algo
                else:
                    new_policy[row, col, action] = 0       # Greedy algo

    return new_policy
'''
搞笑一幕：一开始我的policy update 在count那几行没有缩进到for循环里，导致策略都没更新，全是0
'''


values = np.zeros([4,12])

pi = np.ones([4,12,4]) * 0.25

for _ in range(4):  # 3次刚好是临界点，你可以多训练几次，就收敛了

    for _ in range(100):  # 这就是truncated policy update

        values = policy_evaluation()

    pi = policy_update()


def debug_policy(pi):
    for row in range(4):
        for col in range(12):
            prob_sum = np.sum(pi[row, col])
            if prob_sum != 1:
                print(f"Warning: Probabilities at ({row}, {col}) do not sum to 1. Sum = {prob_sum}")
            else:
                print(f"Probabilities at ({row}, {col}): {pi[row, col]} (Sum = {prob_sum})")

def test():
    #起点在0,0
    row = 0
    col = 0

    #最多玩200步，以防超时，但其实这么简单的格子，一下就收敛了
    for _ in range(200):

        #选择一个动作
        action = np.random.choice(np.arange(4), size=1, p=pi[row, col])[0]

        #执行动作
        row, col, reward = move(row, col, action)

        #获取当前状态，如果状态是终点或者掉陷阱则终止
        if get_state(row, col) in ['trap', 'terminal']:
            break

    #打印所有格子的动作倾向
    for row in range(4):
        line = ''
        for col in range(12):
            action = pi[row, col].argmax()
            action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
            line += action  # line is a string, which select the first maximum action
        print(line)

debug_policy(pi)

test()

'''new grammar learned:
点记法 new_values[row, col] = action_value.sum() 


'''