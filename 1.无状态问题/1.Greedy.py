import numpy as np
import random

probs = np.random.uniform(size=10)  # generate an array, every element 0-1
rewards = [[1] for _ in range(10)]  # ten machines, all rewards set to be 1

def choose_one():
    played_count = sum([len(i) for i in rewards])

    if random.random() < 1/ played_count:
        return random.randint(0, 9)  # exploit
        
    reward_mean = [np.mean(i) for i in rewards]  # calculate rewards_mean for every machine
    return np.argmax(reward_mean)




def UCB_choose_one():
    '''
UCB, Upper Confidence Bound
goal: explore more that hasn't been explored
      explore less that has been explored
'''
    played_count = sum([len(i) for i in rewards])

    numerator = played_count ** 0.5
    denumerator = played_count * 2
    ucb = numerator / denumerator
    ucb = ucb**0.5   # keep ucb in a range

    #计算每个老虎机的奖励平均
    rewards_mean = [np.mean(i) for i in rewards]
    rewards_mean = np.array(rewards_mean)

    #ucb和期望求和
    ucb += rewards_mean

    return ucb.argmax()

'''写了中文的这几步骤是在干什么，我不知道'''
 



def try_and_play(): 
    i = choose_one()

    reward = 0
    if random.random() < probs[i]:
        reward = 1

    rewards[i].append(reward)  # Fix: rewards[i], not reward[i]



def play_n_times():
    n = int(input("How many times do you want it to train"))  # Convert input to int
    for _ in range(n):
        try_and_play()

    target = probs.max() * n

    result = sum([sum(i) for i in rewards])

    return target, result



target, result = play_n_times()

print(f"Target: {target}, Result: {result}")

"""
new grammar:
probs = np.random.uniform(size=10)
rewards = [[1] for _ in range(10)]
result = sum([sum(i) for i in rewards])

target, result = play_n_times()

play_count = sum([len(i) for i in rewards])

""" 