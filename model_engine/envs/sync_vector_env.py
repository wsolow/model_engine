"""Implementation of a synchronous (for loop) vectorization method of any environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch



__all__ = ["SyncVectorEnv"]


class SyncVectorEnv():

    def __init__(
        self,
        env_fns: Iterator[Callable[[], object]] | Sequence[Callable[[], object]],
        observation_mode: str = "same",
    ):
        """Vectorized environment that serially runs multiple environments.
        """
        super().__init__()

        self.env_fns = env_fns
        self.observation_mode = observation_mode
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

        Args:
            name (str): Name of the property to get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any, ...] | Any):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
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