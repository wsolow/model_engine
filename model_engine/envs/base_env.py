"""Base class for model env, handles processing data"""

import numpy as np 
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from model_engine import util
from model_engine.util import GRAPE_NAMES

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

    def process_data(self, data, split:int=3):
        """Process all of the initial data"""

        self.output_vars = self.config.ModelConfig.output_vars
        self.input_vars = self.config.ModelConfig.input_vars

        self.params = self.config.params
        self.params_range = torch.tensor(np.array(self.config.params_range,dtype=np.float32)).to(self.device)

        # Get normalized (weather) data 
        normalized_input_data, self.drange = \
            util.embed_and_normalize_minmax([d.loc[:,self.input_vars] for d in data]) if self.config.normalization_type == "minmax" else util.embed_and_normalize_zscore([d.loc[:,self.input_vars] for d in data])
        
        normalized_input_data = pad_sequence(normalized_input_data, batch_first=True, padding_value=0).to(self.device)
        self.drange = self.drange.to(torch.float32).to(self.device)
        
        # Get input data for use with model to avoid unnormalizing
        if "CULTIVAR" in data[0].columns:
            self.input_data = util.make_tensor_inputs(self.config, [d.loc[:,self.input_vars+["CULTIVAR"]] for d in data])
        else:
            self.input_data = util.make_tensor_inputs(self.config, [d.loc[:,self.input_vars] for d in data])
        # Get validation data
        normalized_output_data, self.output_range = util.embed_output([d.loc[:,self.output_vars] for d in data])
        # TODO: May want an offset output_range to handle when it is [[0,0]]
        normalized_output_data = pad_sequence(normalized_output_data, batch_first=True, padding_value=self.target_mask).to(self.device)
        self.output_range = self.output_range.to(torch.float32).to(self.device)

        # Get the dates
        dates = [d.loc[:,"DAY"].to_numpy().astype('datetime64[D]') for d in data]
        max_len = max(len(arr) for arr in dates)
        # Pad each array to the maximum length
        dates = [np.pad(arr, (0, max_len - len(arr)), mode='maximum') for arr in dates]

        # Shuffle to get train and test splits for data
        # 2:1 train/test split
        if len(self.config.withold_cultivars) == 0: 
            n = len(data)
            inds = np.arange(n)
            np.random.shuffle(inds)
            if split == 0:
                x = 0
            elif split == -1: # Withold 2 seasons
                x = 2
            else:
                assert split > 0, "Variable split must be greater than 0, or -1, 0"
                x = int(np.floor(n/split))
            self.data = {'train': torch.stack([normalized_input_data[i] for i in inds][x:]).to(torch.float32), 
                        'test': torch.stack([normalized_input_data[i] for i in inds][:x]).to(torch.float32) if x > 0 else torch.tensor([])}
            self.val = {'train': torch.stack([normalized_output_data[i] for i in inds][x:]).to(torch.float32), 
                        'test': torch.stack([normalized_output_data[i] for i in inds][:x]).to(torch.float32) if x > 0 else torch.tensor([])}
            self.dates = {'train': np.array([dates[i] for i in inds][x:]), 'test':np.array([dates[i] for i in inds][:x]) if x else np.array([])}
            # Get cultivar weather for use with embedding
            if "CULTIVAR" in data[0].columns:
                cultivar_data = torch.tensor([d.loc[0,"CULTIVAR"] for d in data]).to(torch.float32).to(self.device).unsqueeze(1)
                
                self.num_cultivars = len(torch.unique(cultivar_data))
                self.cultivars = {'train': torch.stack([cultivar_data[i] for i in inds][x:]).to(torch.float32), 
                        'test': torch.stack([cultivar_data[i] for i in inds][:x]).to(torch.float32)}
            else:
                self.num_cultivars = None
                self.cultivars = None
        else: 
            assert "CULTIVAR" in data[0].columns, "CULTIVAR is not in the data columns. Incorrect data file loaded."
            
            train_inds = np.empty(shape=(0,))
            test_inds = np.empty(shape=(0,))
            cultivar_data = np.array([d.loc[0,"CULTIVAR"] for d in data])

            for c, v in self.config.withold_cultivars.items():
                try:
                    model_name, model_num = self.config.ModelConfig.model_parameters.split(":")
                except:
                    raise Exception(f"Incorrectly specified model_parameters file `{self.config.ModelConfig.model_parameters}`")
                cultivar_inds = np.argwhere(GRAPE_NAMES[model_name].index(c) == cultivar_data).flatten()
                np.random.shuffle(cultivar_inds)
                test_inds = np.concatenate((test_inds, cultivar_inds[:v])).astype(np.int32)

            train_inds = np.array(list(set(np.arange(len(cultivar_data))) - set(test_inds)))
            np.random.shuffle(train_inds)
            np.random.shuffle(test_inds)
            self.data = {'train': torch.stack([normalized_input_data[i] for i in train_inds]).to(torch.float32), 
                        'test': (torch.stack([normalized_input_data[i] for i in test_inds]).to(torch.float32) if len(test_inds) > 0 else torch.tensor([]))}
            self.val = {'train': torch.stack([normalized_output_data[i] for i in train_inds]).to(torch.float32), 
                        'test': torch.stack([normalized_output_data[i] for i in test_inds]).to(torch.float32) if len(test_inds) > 0 else torch.tensor([])}
            self.dates = {'train': np.array([dates[i] for i in train_inds]), 'test':np.array([dates[i] for i in test_inds]) if len(test_inds) > 0 else np.array([])}

            cultivar_data = torch.tensor([d.loc[0,"CULTIVAR"] for d in data]).to(torch.float32).to(self.device).unsqueeze(1)
            self.num_cultivars = len(torch.unique(cultivar_data))
            self.cultivars = {'train': torch.stack([cultivar_data[i] for i in train_inds]).to(torch.float32), 
                    'test': torch.stack([cultivar_data[i] for i in test_inds]).to(torch.float32)}
    
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
                output = rollout_env.run(dates=self.curr_dates[:,curr_day])
                normed_output = util.normalize(output, self.output_range).detach()
                output_tens[:,curr_day] = normed_output.view(normed_output.shape[0],-1)
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
                output = rollout_env.run(dates=self.curr_dates[i][:,curr_day])
                normed_output = util.normalize(output, self.output_range).detach()
                output_tens[:,curr_day] = normed_output.view(normed_output.shape[0],-1)
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
        
    '''def set_param_cast(self):
        if self.config.PPO.ppo_type is not None:
            if self.config.PPO.ppo_type == "Discrete":
                def p_cast(self, action):
                    params_predict = self.params_range[:,0] + ( action / (self.param_bins - 1) ) * (self.params_range[:,1]-self.params_range[:,0]) 
                    return params_predict
                self.param_cast = p_cast.__get__(self)
            elif self.config.PPO.ppo_type == "Base" or self.config.PPO.ppo_type == "Reccur":
                def p_cast(self, action):
                    # Cast to range [0,2] from tanh activation and cast to actual parameter range
                    params_predict = torch.tanh(action) + 1
                    params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
                    return params_predict
                self.param_cast = p_cast.__get__(self)
            else:
                msg = "Unknown Output Type"
                raise Exception(msg)
        elif self.config.RNN.rnn_output is not None:
            if self.config.RNN.rnn_output == "Param":
                def p_cast(self, params):
                    return params
                self.param_cast = p_cast.__get__(self)
            elif self.config.RNN.rnn_output == "TanhParam":
                def p_cast(self, params):
                    # Cast to range [0,2] from tanh activation and cast to actual parameter range
                    params_predict = torch.tanh(params) + 1
                    params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
                    return params_predict
                self.param_cast = p_cast.__get__(self)
            elif self.config.RNN.rnn_output == "DeltaParam":
                def p_cast(self, params):
                    return self.curr_params + params
                self.param_cast = p_cast.__get__(self)
            elif self.config.RNN.rnn_output == "DeltaTanhParam":
                def p_cast(self, params):
                    params_predict = torch.tanh(params)
                    params_predict = params_predict * (self.params_range[:,1]-self.params_range[:,0])
                    return self.curr_params + params_predict
                self.param_cast = p_cast.__get__(self)
            else:
                msg = "Unknown Output Type"
                raise Exception(msg)
        else:
            msg = "Unknown Algorithm Type"
            raise Exception(msg)
    '''