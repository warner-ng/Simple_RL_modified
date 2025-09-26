import numpy as np
import random

probs = np.random.uniform(size=10)  # generate an array, every element 0-1
rewards = [[1] for _ in range(10)]  # ten machines, all rewards set to be 1

def Thompson_choose_one():
    #求出每个老虎机出1的次数+1
    count_1 = [sum(i) + 1 for i in rewards]

    #求出每个老虎机出0的次数+1
    count_0 = [sum(1 - np.array(i)) + 1 for i in rewards]

    #按照beta分布计算奖励分布,这可以认为是每一台老虎机中奖的概率
    beta = np.random.beta(count_1, count_0)

    return beta.argmax()


def try_and_play(): 
    i = Thompson_choose_one()

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
