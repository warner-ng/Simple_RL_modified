import numpy as np

class SARSA:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        # 初始化一些超参数
        self.n_actions = n_actions  # 动作空间大小
        self.n_states = n_states  # 状态空间大小
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率

        # Q表，初始化为0
        self.Q = np.zeros((n_states, n_actions))

    def epsilon_greedy(self, state):
        """epsilon-greedy策略选择动作"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)  # 随机选择动作，探索
        else:
            return np.argmax(self.Q[state])  # 选择当前Q值最大的动作，利用

    def update(self, state, action, reward, next_state, next_action):
        """根据SARSA更新规则更新Q值"""
        td_target = reward + self.gamma * self.Q[next_state, next_action]  # 计算目标
        td_error = td_target - self.Q[state, action]  # 计算时序差分误差
        self.Q[state, action] += self.alpha * td_error  # 更新Q值

    def train(self, env, episodes=1000):
        """训练方法，通过与环境的交互来更新Q表"""
        for _ in range(episodes):
            state = env.reset()  # 环境初始化
            action = self.epsilon_greedy(state)  # 选择初始动作

            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)  # 执行动作，获得反馈
                next_action = self.epsilon_greedy(next_state)  # 选择下一个动作

                # 更新Q值
                self.update(state, action, reward, next_state, next_action)

                state, action = next_state, next_action  # 更新状态和动作

