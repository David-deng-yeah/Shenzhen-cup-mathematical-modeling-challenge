'''
自己设置的环境类
智能体：羊
环境：羊、犬和圆形草地，犬采用最优围堵策略围堵羊，若羊在一段时间内逃出圈则胜利，这段时间内没逃出或者被犬抓到则失败；
状态空间：整个圆组成的点集，是二维的；
动作空间：羊每一步可采取的动作的集合
回报的设计：参照pendulum-v0游戏环境源码中的回报的设计方案。
'''
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math
from gym.envs.classic_control import rendering

def trans(tmp):
    return 360*(tmp/(2*np.pi))


class dogSheepEnv(gym.Env):
    # metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.dt = 0.1  # 采样时间
        self.thetaP=np.pi/2# 狗的极坐标
        self.wP=np.pi/15# 狗的角速度
        self.vE=10# 羊的速度
        self.thetaE=np.pi/2# 羊的极坐标
        self.radiusE=0# 羊的极坐标半径
        self.R=100# 圆的半径
        self.state=np.array([self.thetaE,self.radiusE,self.thetaP])# 环境的初始状态
        self.viewer = rendering.Viewer(400, 400)# 画板

        # 自定义动作空间，观察空间
        self.action_space = spaces.Box(
            # 羊的动作空间即为转动的角度，会根据当前位置进行变化
            # low=np.array([(self.thetaE-np.pi/2+2*np.pi)%(2*np.pi)])
            # ,high=np.array([(self.thetaE+np.pi/2+2*np.pi)%(2*np.pi)])
            # 由于怕出现low比high还大的情况，我们的action_space就不做周期处理，用的时候取余2pi就行
            low=np.array([(self.state[0] - np.pi / 2)])
            , high=np.array([(self.state[0] + np.pi / 2)])
            ,dtype=np.float32
        )
        self.observation_space = spaces.Box(
            # 状态空间为 theta_E,R_E,theta_P
            low=np.array([0,0,0])
            ,high=np.array([2*np.pi,self.R,2*np.pi])
            ,dtype=np.float32
        )

    '''
    羊接受一个动作进行位移: 使用PG算法的choose_action
    犬沿劣弧进行位移
    接着判断游戏是否结束
    评价这个动作的回报
    '''
    def step(self, action):# u为action
        _action=(action+2*np.pi)%(2*np.pi)# 对action进行周期化处理
        print('_action: ',_action)
        # 根据action（即θ_E'来计算新的状态）
        reward = self._get_reward(_action)
        self.state = self._get_observation(_action)
        done = self._get_done()
        info={}
        return self.state,reward,done,info


    # 获取reward
    def _get_reward(self,action):
        delta_theta=0
        if action > self.state[2]:
            delta_theta=action-self.state[2]
            if delta_theta > np.pi:
                delta_theta=2*np.pi-action+self.state[2]
        else:
            delta_theta=self.state[2]-action
            if delta_theta > np.pi:
                delta_theta=2*np.pi-action+self.state[2]
        print('thetaE: ',trans(action),'thetaP: ',trans(self.state[2]),'delta: ',trans(delta_theta))
        return self.state[1]+delta_theta # 羊距圆周越近越好，羊与犬的夹角越大越好
    # 判断游戏是否结束
    def _get_done(self):
        if self.state[1]>=self.R:
            return True
        else:
            return False
    # 根据action修改环境，改变状态
    def _get_observation(self,action):
        # 已知现在的位置，首先计算位移后羊的极坐标
        xb=self.state[1]*np.cos(self.state[0])+self.vE*self.dt*np.cos(action)
        yb=self.state[1]*np.sin(self.state[0])+self.vE*self.dt*np.sin(action)
        # print('(x:{},y:{})'.format(xb,yb))
        new_radiusE=math.sqrt(xb*xb+yb*yb)
        new_thetaE=math.atan2(yb,xb)# 返回弧度pi
        # 根据羊的action，选择狼的位移方向并位移
        new_thetaP =self.state[2]
        delta_theta=self.wP*self.dt
        # 修改犬的状态
        # print('犬:',self.state[2])
        # print('羊:', self.state[0])
        if self.state[0] > self.state[2]:
            if self.state[0] - self.state[2] >= np.pi:
                new_thetaP = (new_thetaP - delta_theta + 2 * np.pi) % (2 * np.pi)
                # print('1')
            else:
                new_thetaP = (new_thetaP + delta_theta + 2 * np.pi) % (2 * np.pi)
                # print('2')
        elif self.state[0] < self.state[2]:
            if self.state[2] - self.state[0] >= np.pi:
                new_thetaP = (new_thetaP + delta_theta + 2 * np.pi) % (2 * np.pi)
                # print('3')
            else:
                new_thetaP = (new_thetaP - delta_theta + 2 * np.pi) % (2 * np.pi)
                # print('4')
        else:
            new_thetaP = (new_thetaP + delta_theta + 2 * np.pi) % (2 * np.pi)
            # print('0')
        return np.array([new_thetaE,new_radiusE,new_thetaP])

    # 重置羊和犬的状态
    def reset(self):
        self.state=np.array([np.pi/2,0,np.pi/2],dtype=float)
        return np.array(self.state)

    # def _get_obs(self):
    #     theta, thetadot = self.state
    #     return np.array([np.cos(theta), np.sin(theta), thetadot])

    # 画画显示犬和羊的状态
    def render(self):
        # 清空轨迹
        self.viewer.geoms.clear()
        # 绘制大圆
        ring = rendering.make_circle(radius=self.R,res=50,filled=False)
        transform1 = rendering.Transform(translation=(200, 200))  # 相对偏移
        ring.add_attr(transform1)# 让圆添加平移这个属性
        self.viewer.add_geom(ring)

        # 绘制犬
        xP,yP=self.R*np.cos(self.state[2]),self.R*np.sin(self.state[2])
        ringP = rendering.make_circle(radius=3, res=50, filled=True)
        ringP.set_color(0,0,1)
        transform_P = rendering.Transform(translation=(200+xP, 200+yP))  # 相对偏移
        ringP.add_attr(transform_P)  # 让圆添加平移这个属性
        self.viewer.add_geom(ringP)

        # 绘制羊
        xE, yE = self.state[1] * np.cos(self.state[0]), self.state[1] * np.sin(self.state[0])
        ringE = rendering.make_circle(radius=3, res=50, filled=True)
        ringE.set_color(1, 0, 0)
        transform_E = rendering.Transform(translation=(200+xE, 200+yE))  # 相对偏移
        ringE.add_attr(transform_E)  # 让圆添加平移这个属性
        self.viewer.add_geom(ringE)

        return self.viewer.render()

