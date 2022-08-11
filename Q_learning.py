import numpy as np
import time
import gym
import os
import pickle
import env_sheep


class QLearning:
    def __init__(self, learning_rate=0.1, reward_decay=0.99, e_greedy=0.8):
        # self.target                     # 目标状态（终点）
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 回报衰减率
        self.epsilon = e_greedy  # 探索/利用 贪婪系数
        self.env=env_sheep.dogSheepEnv()
        self.num_cos = 10  # 分为多少份
        self.num_sin = 10
        self.num_dot = 10
        self.num_actions = 10
        # self.actions = self.toBins(-2.0, 2.0, self.num_actions)  # 可以选择的动作空间  离散化
        # 对变化的动作空间进行离散化
        self.actions=self.toBins(self.env.action_space.low,self.env.action_space.high,self.num_actions)
        # q_table是一个二维数组  # 离散化后的状态共有num_pos*num_vel中可能的取值，每种状态会对应一个行动# q_table[s][a]就是当状态为s时作出行动a的有利程度评价值
        self.q_table = np.random.uniform(low=-1, high=1,
                                         size=(self.num_cos * self.num_sin * self.num_dot, self.num_actions))  # Q值表
        self.cos_bins = self.toBins(-1.0, 1.0, self.num_cos)
        self.sin_bins = self.toBins(-1.0, 1.0, self.num_sin)
        self.dot_bins = self.toBins(-8.0, 8.0, self.num_dot)

    # 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
    def choose_action(self, state):
        # 假设epsilon=0.9，下面的操作就是有0.9的概率按Q值表选择最优的，有0.1的概率随机选择动作
        # 随机选动作的意义就是去探索那些可能存在的之前没有发现但是更好的方案/动作/路径
        if np.random.uniform() < self.epsilon:
            # 选择最佳动作（Q值最大的动作）
            action = np.argmax(self.q_table[state])
        else:
            # 随机选择一个动作
            action = np.random.choice(self.actions)
        action = -2 + 4 / (self.num_actions - 1) * action  # 从离散整数变为范围内值
        return action

    # 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，  如[1,10]分为5.5
    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]  # 第一项到倒数第一项

    # 分别对各个连续特征值进行离散化  如[1,10]分为5.5  小于5.5取0  大于取5.5取1
    def digit(self, x, bin):
        n = np.digitize(x, bins=bin)
        return n

    # 将观测值observation离散化处理
    def digitize_state(self, observation):
        # 将矢量打散回4个连续特征值
        cart_sin, cart_cos, cart_dot = observation
        # 分别对各个连续特征值进行离散化（分箱处理）
        digitized = [self.digit(cart_sin, self.cos_bins),
                     self.digit(cart_cos, self.sin_bins),
                     self.digit(cart_dot, self.dot_bins), ]
        # 将离散值再组合为一个离散值，作为最终结果
        return (digitized[1] * self.num_cos + digitized[0]) * self.num_dot + digitized[2]

    # 学习，主要是更新Q值
    def learn(self, state, action, r, next_state):
        action = self.digit(action, self.actions)
        next_action = np.argmax(self.q_table[next_state])
        q_predict = self.q_table[state, action]
        q_target = r + self.gamma * self.q_table[next_state, next_action]  # Q值的迭代更新公式
        self.q_table[state, action] += self.lr * (q_target - q_predict)  # update


def train():
    env = gym.make('Pendulum-v0')
    print(env.action_space)
    agent = QLearning()
    # with open(os.getcwd()+'/tmp/Pendulum.model', 'rb') as f:
    #     agent = pickle.load(f)
    action = [0]  # 输入格式要求 要是数组
    for i in range(20000):  # 训练次数
        observation = env.reset()  # 状态  cos(theta), sin(theta) , thetadot角速度
        state = agent.digitize_state(observation)  # 状态标准化
        for t in range(100):  # 一次训练最大运行次数
            action[0] = agent.choose_action(state)  # 动作 -2到2
            observation, reward, done, info = env.step(action)
            next_state = agent.digitize_state(observation)
            if done:
                reward -= 200  # 对于一些直接导致最终失败的错误行动，其报酬值要减200
            if reward >= -1:  # 竖直时时reward接近0  -10到0
                reward += 40  # 给大一点
                print('arrive')
            # print(action,reward,done,state,next_state)
            agent.learn(state, action[0], reward, next_state)
            state = next_state
            if done:  # done   重新加载环境
                print("Episode finished after {} timesteps".format(t + 1))
                break
            # env.render()    # 更新并渲染画面
    print(agent.q_table)
    env.close()
    # 保存
    with open(os.getcwd() + '/tmp/Pendulum.model', 'wb') as f:
        pickle.dump(agent, f)


def test():
    env = gym.make('Pendulum-v0')
    print(env.action_space)
    with open(os.getcwd() + '/tmp/Pendulum.model', 'rb') as f:
        agent = pickle.load(f)
    agent.epsilon = 1  # 测试时取1  每次选最优结果
    observation = env.reset()  #
    state = agent.digitize_state(observation)  # 状态标准化
    action = [0]  # 输入格式要求 要是数组
    for t in range(200):  # 一次训练最大运行次数
        action[0] = agent.choose_action(state)  #
        observation, reward, done, info = env.step(action)
        next_state = agent.digitize_state(observation)
        print(action, reward, done, state, next_state)
        print(observation)
        if reward >= -1:  # 竖直时时reward接近0  -10到0
            print('arrive')
        agent.learn(state, action[0], reward, next_state)
        state = next_state
        env.render()  # 更新并渲染画面
        time.sleep(0.02)
    env.close()


def run_test():
    env = gym.make('Pendulum-v0')
    action = [0]
    observation = env.reset()  # 状态
    print(env.action_space)
    print(observation)
    actions = np.linspace(-2, 2, 10)
    for t in range(100):  #
        # action[0] =  random.uniform(-2,2)   #力矩  -2到2
        action[0] = 2
        observation, reward, done, info = env.step(action)
        print(action, reward, done)

        # print('observation:',observation)
        # print('theta:',env.state)
        env.render()
        time.sleep(1)
    env.close()


if __name__ == '__main__':
    train()
    test()
    # run_test()

