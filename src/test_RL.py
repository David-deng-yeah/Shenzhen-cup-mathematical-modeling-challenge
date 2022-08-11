import gym
import env_sheep
import numpy as np
import time
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy

def trans(tmp):
    return 360*(tmp/(2*np.pi))

env = env_sheep.dogSheepEnv()
model = DDPG(MlpPolicy, env=env)
model.learn(total_timesteps=10000)

print('训练完毕')
for i_episode in range(20):
    # 重新初始化环境
    observation=env.reset()
    print('第',i_episode,'次')
    for t in range(100):
        action = env.action_space.sample()
        # action, state = model.predict(observation=observation)
        print('action: ',action[0])
        observation, reward, done, info = env.step(action[0])
        print('reward: ',reward)
        env.render()
        time.sleep(0.1)
        # 追逃游戏结束
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()