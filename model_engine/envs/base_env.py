"""Base class for model env, handles processing data"""

import numpy as np 
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from model_engine import util

from model_engine.engine import MultiModelEngine, BatchModelEngine
import copy

class Base_Env():
    """
    Environment wrapper around model
    """

    def __init__(self, config=None, data=None, num_models=1):
        """
        Initialize gym environment with model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.config = config
        self.num_models = num_models
        self.target_mask = np.nan

    def run_till(self, i=None):
        """
        Run model i until the end of the sequence
        """
        if isinstance(self.envs, BatchModelEngine):
            curr_model_state = copy.deepcopy(self.envs.get_state())
            rollout_env = self.envs
            #rollout_env = copy.deepcopy(self.envs)
            curr_day = self.curr_day+1
            b_len = self.batch_len 
            output_tens = torch.zeros(size=(self.num_envs, b_len, len(self.output_vars))).to(self.device)
            while curr_day < b_len:
                output = rollout_env.run(dates=self.curr_dates[:,curr_day]).detach()
                output_tens[:,curr_day] = output.view(output.shape[0],-1)
                curr_day += 1

            # Reset model state back  
            self.envs.set_state(curr_model_state)
        else:
            #curr_model_state = self.envs[i].get_state(i=i ).clone()
            rollout_env = copy.deepcopy(self.envs[i])
            
            curr_day = self.curr_day[i]+1 
            b_len = self.batch_len[i] 
            output_tens = torch.zeros(size=(1, b_len, len(self.output_vars))).to(self.device)

            while curr_day < b_len:
                output = rollout_env.run(dates=self.curr_dates[i][:,curr_day]).detach()
                output_tens[:,curr_day] = output.view(output.shape[0],-1)
                curr_day += 1

            # Reset model state back    
            #self.envs[i].set_state(curr_model_state, i=i)
        return output_tens
    
    def param_cast(self, params):
        """
        To be implemented in subclasses, used to change parameters  
        """
        # Cast to range [0,2] from tanh activation and cast to actual parameter range
        params_predict = torch.tanh(params) + 1
        params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
        return params_predict
            
    def posbinary_reward(self, output, val, i=None):
        """Binary reward function"""
        return torch.sum((output == val) * (~torch.isnan(val)),axis=-1)
    
    def negbinary_reward(self, output, val, i=None):
        """Binary reward function"""
        return -torch.sum((output != val) * (~torch.isnan(val)),axis=-1)
    
    def todate_reward(self, output, val, i=None):
        """Reward for how much matches to date"""
        mask = ~torch.isnan(val)
        if i is None:
            self.reward_sum += torch.sum((output == val) * mask,axis=-1).flatten()[0] / mask.sum()

        else:
            self.reward_sum[i] += torch.sum((output == val) * mask,axis=-1).flatten()[0] / mask.sum()

        return self.reward_sum if i is None else self.reward_sum[i]

    def projectionsum_reward(self, output, val, i=None):
        """Reward is the projection sum of past and for params into the future"""
        if isinstance(self.envs, BatchModelEngine):
            mask = ~torch.isnan(val)
            self.reward_sum += torch.sum((output == val) * mask,axis=-1)

            output = self.run_till()
            mask2 = ~torch.isnan(self.curr_val[:,self.curr_day:])
            reward = self.reward_sum + \
                    torch.sum((output[:,self.curr_day:] == self.curr_val[:,self.curr_day:]) * \
                            mask2,axis=(-1,-2))
            return reward / self.batch_len
        else:
            mask = ~torch.isnan(val)
            self.reward_sum[i] += torch.sum((output == val) * mask,axis=-1).flatten()[0]
            output = self.run_till(i)
            mask2 = ~torch.isnan(self.curr_val[i][:,self.curr_day[i]:])
            reward = self.reward_sum[i] + \
                torch.sum((output[:,self.curr_day[i]:] == self.curr_val[i][:,self.curr_day[i]:]) * \
                        mask2,axis=(-1,-2)).flatten()[0]
            
            return reward / self.batch_len[i]
    
    def projection_reward(self, output, val, i=None):
        """Reward is the projection for params into the future"""
        if isinstance(self.envs, BatchModelEngine):
            mask = ~torch.isnan(self.curr_val[:,self.curr_day:])
            output = self.run_till()
            reward = torch.sum((output[:,self.curr_day:] == self.curr_val[:,self.curr_day:]) * \
                            mask,axis=(-1,-2))
            
            return reward / (mask.sum((-1,-2))+1)
        else:
            mask = ~torch.isnan(self.curr_val[i][:,self.curr_day[i]:])
            output = self.run_till(i)
            reward = torch.sum((output[:,self.curr_day[i]:] == self.curr_val[i][:,self.curr_day[i]:]) * \
                        mask,axis=(-1,-2)).flatten()[0]

            return reward / (mask.sum((-1,-2))+1)
        
    def todate_continuous_reward(self, output, val, i=None):
        """Reward for how much matches to date for MSE loss"""
        if i is None:
            self.reward_sum += torch.sum(((output - val) ** 2).nan_to_num(nan=0.0) ** 2 * (~torch.isnan(val)),axis=-1).flatten()[0] / self.batch_len
        else:
            self.reward_sum[i] += torch.sum(((output - val) ** 2).nan_to_num(nan=0.0) ** 2 * (~torch.isnan(val)),axis=-1).flatten()[0] / self.batch_len[i]

        return self.reward_sum if i is None else self.reward_sum[i]

    def projectionsum_continuous_reward(self, output, val, i=None):
        """Reward is the projection sum of past and for params into the future"""

        if isinstance(self.envs, BatchModelEngine):
            mask = ~torch.isnan(val)
            self.reward_sum += -torch.sum(((output - val) ** 2).nan_to_num(nan=0.0) * mask,axis=-1)

            output = self.run_till()
            mask2 = ~torch.isnan(self.curr_val[:,self.curr_day:])
            reward = self.reward_sum + \
                    -torch.sum(((output[:,self.curr_day:] - self.curr_val[:,self.curr_day:]) ** 2).nan_to_num(nan=0.0) * \
                            mask2,axis=(-1,-2))

            return reward / self.batch_len
        else:
            mask = ~torch.isnan(val)
            self.reward_sum[i] += -torch.sum(((output - val) ** 2).nan_to_num(nan=0.0) * mask,axis=-1).flatten()[0]

            output = self.run_till(i)
            mask2 = ~torch.isnan(self.curr_val[i][:,self.curr_day[i]:])
            reward = self.reward_sum[i] + \
                -torch.sum(((output[:,self.curr_day[i]:] - self.curr_val[i][:,self.curr_day[i]:]) ** 2).nan_to_num(nan=0.0) * \
                        mask2,axis=(-1,-2)).flatten()[0]

            return reward / self.batch_len[i]
        
    def projection_continuous_reward(self, output, val, i=None):
        """Reward is the projection for params into the future"""
        if isinstance(self.envs, BatchModelEngine):
            mask = ~torch.isnan(self.curr_val[:,self.curr_day:])
            output = self.run_till()
            reward = -torch.sum(((output[:,self.curr_day:] - self.curr_val[:,self.curr_day:]) ** 2).nan_to_num(nan=0.0) * \
                            (~torch.isnan(self.curr_val[:,self.curr_day:])),axis=(-1,-2))
            return reward / (mask.sum((-1,-2))+1)

        else:
            mask = torch.isnan(self.curr_val[i][:,self.curr_day[i]:])
            output = self.run_till(i) 
            reward = -torch.sum(((output[:,self.curr_day[i]:] - self.curr_val[i][:,self.curr_day[i]:]) **2).nan_to_num(nan=0.0) * \
                        (~torch.isnan(self.curr_val[i][:,self.curr_day[i]:])),axis=(-1,-2)).flatten()[0]

            return reward / (mask.sum((-1,-2))+1)

    def set_reward_func(self):
        """Set the reward function"""
        if self.config.PPO.reward == "posbinary":
            self.reward_func = self.posbinary_reward
        elif self.config.PPO.reward == "negbinary":
            self.reward_func = self.negbinary_reward
        elif self.config.PPO.reward == "todate":
            self.reward_func = self.todate_reward
        elif self.config.PPO.reward == "projectionsum":
            self.reward_func = self.projectionsum_reward
        elif self.config.PPO.reward == "projection":
            self.reward_func = self.projection_reward
        elif self.config.PPO.reward == "todate_continuous":
            self.reward_func = self.todate_continuous_reward
        elif self.config.PPO.reward == "projectionsum_continuous":
            self.reward_func = self.projectionsum_continuous_reward
        elif self.config.PPO.reward == "projection_continuous":
            self.reward_func = self.projection_continuous_reward
        else:
            raise NotImplementedError("Reward function not implemented")
        