# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import warnings
from typing import Tuple, Optional, ClassVar, Dict, Generic, Type, TypeVar

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gym 

import numpy as np

T = TypeVar('T')


class ObjectFactory(Generic[T]):
    def __init__(
        self,
        clz: Type[T],
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None
        ):
        self._clz = clz
        self._args = args
        self._kwargs = kwargs

    def make_object(
        self
        ) -> T:
        if self._args is not None and self._kwargs is not None:
            return self._clz(*self._args, **self._kwargs)
        elif self._args is not None:
            return self._clz(*self._args)
        elif self._kwargs is not None:
            return self._clz(**self._kwargs)
        else:
            return self._clz()

    @property
    def clz(
        self
        ) -> Type:
        return self._clz

    @property
    def args(
        self
        ) -> Optional[Tuple]:
        return self._args

    @property
    def kwargs(
        self
        ) -> Optional[Dict]:
        return self._kwargs


class MujucoPolicyFunc:
    ANT_ENV: ClassVar[Tuple[str, float, float, int]] = ('Ant-v2', -1.0, 1.0, 1)
    SWIMMER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Swimmer-v2', -1.0, 1.0, 5)
    HALF_CHEETAH_ENV: ClassVar[Tuple[str, float, float, int]] = ('HalfCheetah-v2', -1.0, 1.0, 5)
    HOPPER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Hopper-v2', -1.4, 1.4, 5)
    WALKER_2D_ENV: ClassVar[Tuple[str, float, float, int]] = ('Walker2d-v2', -1.8, 0.9, 5)
    HUMANOID_ENV: ClassVar[Tuple[str, float, float, int]] = ('Humanoid-v2', -1.0, 1.0, 5)

    ENV_CP = {
        ANT_ENV[0]         : 10.0,
        SWIMMER_ENV[0]     : 30.0,
        HALF_CHEETAH_ENV[0]: 10.0,
        HOPPER_ENV[0]      : 100.0,
        WALKER_2D_ENV[0]   : 50.0,
        HUMANOID_ENV[0]    : 20.0
    }

    def __init__(
        self,
        policy_file: str,
        env: str,
        lb: float,
        ub: float,
        num_rollouts
        ):
        lin_policy = np.load(policy_file, allow_pickle=True)
        lin_policy = lin_policy['arr_0']
        self._policy = lin_policy[0]
        self._mean = lin_policy[1]
        self._std = lin_policy[2]
        self._dims = len(self._policy.ravel())
        self._lb = np.full(self._dims, lb)
        self._ub = np.full(self._dims, ub)
        self._env_name = env
        self._env = gym.make(env)
        self._num_rollouts = num_rollouts
        self._render = False

    @property
    def lb(
        self
        ) -> np.ndarray:
        return self._lb

    @property
    def ub(
        self
        ) -> np.ndarray:
        return self._ub

    @property
    def dims(
        self
        ) -> int:
        return self._dims

    @property
    def is_minimizing(
        self
        ) -> bool:
        return False

    def __call__(
        self,
        x: np.ndarray
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        fx = np.zeros(len(x))
        for i, actions in enumerate(x):
            m = actions.reshape(self._policy.shape)
            rewards = []
            observations = []
            actions = []
            for _ in range(self._num_rollouts):
                obs, _ = self._env.reset()
                done = False
                total_reward = 0.
                steps = 0
                while not done:
                    action = np.dot(m, (obs - self._mean) / self._std)
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _, _ = self._env.step(action)
                    total_reward += r
                    steps += 1
                    if self._render:
                        self._env.render()
                rewards.append(total_reward)
            fx[i] = np.mean(rewards)
        return fx, None

    def __str__(
        self
        ):
        return f"Mujuco_{self._env_name}[{self.dims}]"


func_dir = os.path.dirname(os.path.abspath(__file__))
func_factories = {
    "ant"         : ObjectFactory(
        MujucoPolicyFunc,
        (f"{func_dir}/mujuco_policies/Ant-v1/lin_policy_plus.npz", *MujucoPolicyFunc.ANT_ENV)
        ),
    "half_cheetah": ObjectFactory(
        MujucoPolicyFunc,
        (f"{func_dir}/mujuco_policies/HalfCheetah-v1/lin_policy_plus.npz",
         *MujucoPolicyFunc.HALF_CHEETAH_ENV)
        ),
    "hopper"      : ObjectFactory(
        MujucoPolicyFunc,
        (f"{func_dir}/mujuco_policies/Hopper-v1/lin_policy_plus.npz",
         *MujucoPolicyFunc.HOPPER_ENV)
        ),
    "humanoid"    : ObjectFactory(
        MujucoPolicyFunc,
        (f"{func_dir}/mujuco_policies/Humanoid-v1/lin_policy_plus.npz",
         *MujucoPolicyFunc.HUMANOID_ENV)
        ),
    "swimmer"     : ObjectFactory(
        MujucoPolicyFunc,
        (f"{func_dir}/mujuco_policies/Swimmer-v1/lin_policy_plus.npz",
         *MujucoPolicyFunc.SWIMMER_ENV)
        ),
    "walker_2d"   : ObjectFactory(
        MujucoPolicyFunc,
        (f"{func_dir}/mujuco_policies/Walker2d-v1/lin_policy_plus.npz",
         *MujucoPolicyFunc.WALKER_2D_ENV)
        ),
}
