"""Implementation of a synchronous (for loop) vectorization method of any environment."""

from __future__ import annotations
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from model_engine.envs.base_env import Base_Env
from model_engine.engine import get_engine, MultiModelEngine
import model_engine.util as util

class SyncVectorEnv():

    def __init__(
        self,
        env_fns: Iterator[Callable[[], object]] | Sequence[Callable[[], object]],
    ):
        """Vectorized environment that serially runs multiple environments.
        """
        super().__init__()

        self.env_fns = env_fns
        self.autoreset_mode = None

        # Initialise all sub-environments
        self.envs = [env_fn() for env_fn in env_fns]

        # Define core attributes using the sub-environments
        self.num_envs = len(self.envs)

        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space

        # Initialise attributes used in `step` and `reset`
        self._env_obs = [None for _ in range(self.num_envs)]
        self._observations = np.zeros((self.num_envs,)+self.single_observation_space.shape)
        
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(
        self,
        *,
        options: dict[str, Any] | None = None,
    ):
        """Resets each of the sub-environments and concatenate the results together.
        """

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        infos = {}
        for i, env in enumerate(self.envs):
            self._env_obs[i], env_info = env.reset(
                options=options
            )

        # Concatenate the observations
        self._observations = torch.stack(self._env_obs)
        return self._observations, infos

    def step(self, actions):
        """Steps through each of the environments returning the batched results.
        """
        infos = {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                self._env_obs[i], _ = self.envs[i].reset()

                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (
                    self._env_obs[i],
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    _,
                ) = self.envs[i].step(action)

        # Concatenate the observations
        self._observations = self._observations = torch.stack(self._env_obs)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            self._observations,
            self._rewards,
            self._terminations,
            self._truncations,
            infos,
        )

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Calls a sub-environment method with name and applies args and kwargs.
        """
        results = []
        for env in self.envs:
            function = env.get_wrapper_attr(name)

            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Get a property from each parallel environment.
        """
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any, ...] | Any):
        """Sets an attribute of the sub-environments
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]

        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)

class UnifiedSyncVectorEnv(Base_Env):

    def __init__(
        self, num_envs:int=1, config=None, data=None
    ):
        """Vectorized environment that serially runs multiple environments.
        Unified to handle all data in base class
        """
        super().__init__(config, data)

        self.num_envs = num_envs
        self.num_models = 1 # Kept for compatibility if we ever do batches
        self.autoreset_mode = None

        self.process_data(data)

        self.params = config.params
        self.params_range = torch.tensor(np.array(self.config.params_range,dtype=np.float32)).to(self.device)
        self.model_constr = get_engine(self.config)
        self.envs = [self.model_constr(num_models=self.num_models, config=config['ModelConfig'], inputprovider=self.input_data, device=self.device) for _ in range(num_envs)]
        
        self.single_observation_space = np.empty(shape=(1 + len(self.output_vars) + len(self.input_vars),))
        self.single_action_space = np.empty(shape=(len(self.params),))

        # Initialize data storage
        self.curr_data = [None for _ in range(self.num_envs)]
        self.curr_val = [None for _ in range(self.num_envs)]
        self.curr_dates = [None for _ in range(self.num_envs)]
        self.batch_len = [None for _ in range(self.num_envs)]
        self.curr_day = [None for _ in range(self.num_envs)]
        # Initialise attributes used in `step` and `reset`
        self._env_obs = [None for _ in range(self.num_envs)]
        self._observations = np.zeros((self.num_envs,)+self.single_observation_space.shape)
        
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        init_params = self.envs[0].get_params()[0]
        if isinstance(self.envs[0], MultiModelEngine):
            self.init_params = torch.stack([init_params[k] for k in self.params] ).to(self.device).view(self.num_models, -1)

    def reset(self):
        """Resets each of the sub-environments and concatenate the results together.
        """

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        infos = {}
        for i, env in enumerate(self.envs):
            self.single_env_reset(i)
        self._observations = torch.stack(self._env_obs)
        return self._observations, infos

    def step(self, actions):
        """Steps through each of the environments returning the batched results.
        """
        infos = {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                self.single_env_reset(i)

                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
                print('#### NEW YEAR #### ')
            else:
                if isinstance(action, np.ndarray):
                    action = torch.tensor(action).to(self.device)
                if action.ndim == 1:
                    action = action.unsqueeze(0)

                params_predict = self.param_cast(action)
                #self.envs[i].set_model_params(params_predict, self.params)
                self.envs[i].set_model_params(self.init_params, self.params)
                output = self.envs[i].run(dates=self.curr_dates[i][:,self.curr_day[i]])
                # Normalize output 
                normed_output = util.tensor_normalize(output, self.output_range).detach()
                normed_output = normed_output.view(normed_output.shape[0],-1)
                obs = torch.cat((normed_output, self.curr_data[i][:,self.curr_day[i]]),dim=-1)
                
                reward = -torch.sum((normed_output != self.curr_val[i][:,self.curr_day[i]]) * (self.curr_val[i][:,self.curr_day[i]] != self.target_mask),axis=-1)
                self.curr_day[i] += 1

                trunc = np.zeros(self.num_models)
                done = np.tile(self.curr_day[i] >= self.batch_len[i], self.num_models)

                obs = obs.flatten()
                reward = reward.flatten()[0]

                self._env_obs[i] = obs
                self._rewards[i] = reward
                self._terminations[i] = done
                self._truncations[i] = trunc 

        # Concatenate the observations
        self._observations = torch.stack(self._env_obs)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            self._observations,
            self._rewards,
            self._terminations,
            self._truncations,
            infos,
        )
    
    def single_env_reset(self, i:int):
        """Reset a single environment"""
        # Shuffle data and record length
        inds = np.arange(len(self.data['train']))
        np.random.shuffle(inds)
        self.curr_data[i] = self.data['train'][inds[:self.num_models]]
        self.curr_val[i] = self.val['train'][inds[:self.num_models]]
        self.curr_dates[i] = self.dates['train'][inds[:self.num_models]]
        # Get current batch and sequence length
        self.batch_len[i] = self.curr_data[i].shape[1]
        self.curr_day[i] = 1

        output = self.envs[i].reset()
        # Cat waether onto obs
        normed_output = util.tensor_normalize(output, self.output_range).detach()
        normed_output = normed_output.view(normed_output.shape[0],-1)
        self._env_obs[i] = torch.cat((normed_output, self.curr_data[i][:,0]),dim=-1).flatten()

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Calls a sub-environment method with name and applies args and kwargs.
        """
        results = []
        for env in self.envs:
            function = env.get_wrapper_attr(name)

            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Get a property from each parallel environment.
        """
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any, ...] | Any):
        """Sets an attribute of the sub-environments
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]

        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)

    def param_cast(self, action):
        """Cast action to params"""
        params_predict = torch.tanh(action) + 1 # convert from tanh
        params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
        return params_predict

    