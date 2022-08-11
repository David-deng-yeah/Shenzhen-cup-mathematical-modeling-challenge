import math

import numpy as np

from env import dogSheepEnv
from rl import DDPG
import matplotlib.pyplot as plt
import time

MAX_EPISODES = 200# 比赛次数
MAX_EP_STEPS = 2000# 每把比赛的步数
ON_TRAIN = False# 控制程序是进行训练还是进行测试
sigma=10 # 碰撞精度

# reward_list=[]# 准备画图
# ep_reward_list=[]
thetaP_list=[]
thetaP2_list=[]
thetaE_list=[]
rE_list=[]

# 设置环境
env = dogSheepEnv()
# 设置维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

# 设置强化学习模型
rl = DDPG(action_dim, state_dim, action_bound)

# 判断羊是否给抓住
def catch(R,theta1,theta2,theta3):
    x=R*np.cos(theta1)
    y=R*np.sin(theta1)
    a=R*np.cos(theta2)
    b=R*np.sin(theta2)
    A=R*np.cos(theta3)
    B=R*np.sin(theta3)
    len1=math.sqrt((x-a)*(x-a)+(y-b)*(y-b))
    len2=math.sqrt((x-A)*(x-A)+(y-B)*(y-B))
    if len1 <= sigma and len2 <= sigma:
        return True
    else:
        return False

def trans(tmp):
    return 360*(tmp/(2*np.pi))

# def dis(R,theta1,theta2):
#     x=R*np.cos(theta1)
#     y=R*np.sin(theta1)
#     a=R*np.cos(theta2)
#     b=R*np.sin(theta2)
#     len=math.sqrt((x-a)*(x-a)+(y-b)*(y-b))
#     if len <= sigma:
#         return False
#     else:
#         return True

# 训练过程
'''
env和算法的交互
传state
将一次学习的经历装进记忆库

rl模型一直将学习经历填进记忆库，直到记忆库满了才开始学习
填充记忆库的过程中，环境不断地交互
'''
def train():
    for i in range(MAX_EPISODES):
        print('i: ',i)
        state = env.reset()
        ep_reward = 0.# 单局比赛的总reward
        for j in range(MAX_EP_STEPS):
            env.render()# 画图
            action = rl.choose_action(state)# 算法预测下一个动作
            action=np.random.normal(action,scale=0.01) # 随机一下
            # 这里限制一下动作空间
            action=np.clip(action,env.state[0]-np.pi/2,env.state[0]+np.pi/2)
            action=(action+2*np.pi)%(2*np.pi)
            _state, reward, done = env.step(action) # 和环境交互
            # reward_list.append(reward)
            # print('reward: ',reward)
            # print('i: ',i,' choose_action: ', trans(action[0]),' reward: ',reward,' state: ',_state)
            rl.store_transition(state, action, reward, _state)# 把这次经历装进记忆库
            ep_reward += reward
            # 记忆模块填完之后算法开始学习
            if rl.memory_full:
                rl.learn()
            state = _state
            # time.sleep(0.2)
            if done or j == MAX_EP_STEPS-1: # 结束
                # if env.state[1] >= env.R and dis(env.R,env.state[0],env.state[2]):
                if (env.state[1] >= env.R) and (not catch(env.R, env.state[0], env.state[2],env.state[3])):
                    print('sheep win')
                else:
                    print('dog win')
                # ep_reward_list.append(ep_reward)
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_reward, j))
                break
    rl.save() # 保存模型

# 测试
def eval():
    rl.restore()# 提取模型
    # env.render()
    # env.viewer.set_vsync(True)
    # while True:
    #     # print('新的一次')
    #     state = env.reset()
    #     for _ in range(1000):
    #         env.render()
    #         action = rl.choose_action(state)
    #         action = np.random.normal(action, scale=0.01)  # 随机一下
    #         # 这里限制一下动作空间
    #         action = np.clip(action, env.state[0] - np.pi / 2, env.state[0] + np.pi / 2)
    #         action = (action + 2 * np.pi) % (2 * np.pi)
    #         # print('choose action: ',action,'state: ',env.state)
    #         state, reward, done = env.step(action)
    #         thetaE_list.append(state[0])
    #         rE_list.append(state[1])
    #         thetaP_list.append(state[2])
    #         if done:
    #             if env.state[1] >= env.R and dis(env.R,env.state[0],env.state[2]):
    #                 print('sheep win')
    #             else:
    #                 print('dog win')
    #             break
    state = env.reset()
    print('thetaP: ',state[2])
    print('thetaP2: ', state[3])
    for _ in range(1000):
        env.render()
        action = rl.choose_action(state)
        # 这里限制一下动作空间
        action = np.clip(action, env.state[0] - np.pi / 2, env.state[0] + np.pi / 2)
        action = (action + 2 * np.pi) % (2 * np.pi)
        state, reward, done = env.step(action)
        thetaE_list.append(state[0])
        rE_list.append(state[1])
        thetaP_list.append(state[2])
        thetaP2_list.append(state[3])
        # print('choose action: ', action,' reward: ',reward, 'state: ', env.state)
        if done:
            break
    input('input: ')


if ON_TRAIN:
    train()
else:
    eval()

# 画reward图
# plt.figure()
# len2=len(ep_reward_list)
# plt.plot(list(range(len2)),ep_reward_list)
# plt.title('reward convergence trend ')
# plt.xlabel('steps')
# plt.ylabel("reward")
# plt.show()

# 画犬1的图
plt.figure()
plt.plot(list(range(len(thetaP_list))),thetaP_list)
plt.title('pursuer1 theta')
plt.xlabel('steps')
plt.ylabel("theta")
plt.show()

# 画犬2的图
plt.figure()
plt.plot(list(range(len(thetaP2_list))),thetaP2_list)
plt.title('pursuer2 theta')
plt.xlabel('steps')
plt.ylabel("theta")
plt.show()

# 画羊的极角
plt.figure()
plt.plot(list(range(len(thetaE_list))),thetaE_list)
plt.title('escaper theta')
plt.xlabel('steps')
plt.ylabel("theta")
plt.show()

# 画羊的极径
plt.figure()
plt.plot(list(range(len(rE_list))),rE_list)
plt.title('escaper radius')
plt.xlabel('steps')
plt.ylabel("radius")
plt.show()