import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class CircleDrive(gym.Env):
    def __init__(self, render: bool = False):
        self._render = render
        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([-10., 10.]),
            high=np.array([10., 10.]),
            dtype=np.float32
        )

        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([0., 0.]),
            high=np.array([100., np.pi])
        )

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        # 圆周运动的半径
        self.target_radius = 3.
        # 计数器
        self.step_num = 0

    def __apply_action(self, action):
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        left_v, right_v = action
        # 裁剪防止输入动作超出动作空间
        left_v = np.clip(left_v, -10., 10.)
        right_v = np.clip(right_v, -10., 10.)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[3, 2],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[10., 10.],
            physicsClientId=self._physics_client_id
        )

    def __get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        # 先获取距离
        distance = np.linalg.norm(np.array(basePos))
        # 再获取夹角
        matrix = p.getMatrixFromQuaternion(baseOri, physicsClientId=self._physics_client_id)
        direction_vector = np.array([matrix[0], matrix[3], matrix[6]])
        position_vector = np.array(basePos)
        d_L2 = np.linalg.norm(direction_vector)
        p_L2 = np.linalg.norm(position_vector)
        if d_L2 == 0 or p_L2 == 0:
            return np.array([distance, np.pi])
        angle = np.arccos(np.dot(direction_vector, position_vector) / (d_L2 * p_L2))
        return np.array([distance, angle])

    def reset(self):
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8)
        self.robot = p.loadURDF("./miniBox.urdf", basePosition=[0., 0., 0.2], physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        return self.__get_observation()

    def step(self, action):
        self.__apply_action(action)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.step_num += 1
        state = self.__get_observation()
        reward = -1 * np.abs(state[0] - self.target_radius) - 0.1 * np.abs(state[1] - np.pi / 2.)
        if state[0] > 5. or self.step_num > 36000:
            done = True
        else:
            done = False
        info = {}
        return state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1