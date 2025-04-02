"""Base class for model env, handles processing data"""

import numpy as np 
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from model_engine import util
from data_real.data_load import GRAPE_NAMES

from model_engine.engine import MultiModelEngine, BatchModelEngine

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

        # Get normalized (weather) data 
        normalized_input_data, self.drange = util.embed_and_normalize([d.loc[:,self.input_vars] for d in data])
        normalized_input_data = pad_sequence(normalized_input_data, batch_first=True, padding_value=0).to(self.device)
        self.drange = self.drange.to(torch.float32).to(self.device)
        
        # Get input data for use with model to avoid unnormalizing
        if "CULTIVAR" in data[0].columns:
            self.input_data = util.make_tensor_inputs(self.config, [d.loc[:,self.input_vars+["CULTIVAR"]] for d in data])
        else:
            self.input_data = util.make_tensor_inputs(self.config, [d.loc[:,self.input_vars] for d in data])
        # Get validation data
        normalized_output_data, self.output_range = util.embed_output([d.loc[:,self.output_vars] for d in data])
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
            else:
                x = int(np.floor(n/split))
            self.data = {'train': torch.stack([normalized_input_data[i] for i in inds][x:]).to(torch.float32), 
                        'test': torch.stack([normalized_input_data[i] for i in inds][:x]).to(torch.float32) if x > 0 else torch.Tensor([])}
            self.val = {'train': torch.stack([normalized_output_data[i] for i in inds][x:]).to(torch.float32), 
                        'test': torch.stack([normalized_output_data[i] for i in inds][:x]).to(torch.float32) if x > 0 else torch.Tensor([])}
            self.dates = {'train': np.array([dates[i] for i in inds][x:]), 'test':np.array([dates[i] for i in inds][:x]) if x else np.array([])}
            # Get cultivar weather for use with embedding
            if "CULTIVAR" in data[0].columns:
                cultivar_data = torch.tensor([d.loc[0,"CULTIVAR"] for d in data]).to(torch.float32).to(self.device).unsqueeze(1)
                self.num_cultivars = len(torch.unique(cultivar_data))
                self.cultivars = {'train': torch.stack([cultivar_data[i] for i in inds][x:]).to(torch.float32), 
                        'test': torch.stack([cultivar_data[i] for i in inds][:x]).to(torch.float32)}
        else: 
            assert "CULTIVAR" in data[0].columns, "CULTIVAR is not in the data columns. Incorrect data file loaded."
            
            train_inds = np.empty(shape=(0,))
            test_inds = np.empty(shape=(0,))
            cultivar_data = np.array([d.loc[0,"CULTIVAR"] for d in data])
            for c in self.config.withold_cultivars:
                try:
                    model_name, model_num = self.config.ModelConfig.model_parameters.split(":")
                except:
                    raise Exception(f"Incorrectly specified model_parameters file `{self.config.ModelConfig.model_parameters}`")
                test_inds = np.concatenate((test_inds, np.argwhere(GRAPE_NAMES[model_name].index(c) == cultivar_data).flatten())).astype(np.int32)
            train_inds = np.array(list(set(np.arange(len(cultivar_data))) - set(test_inds)))
            
            np.random.shuffle(train_inds)
            np.random.shuffle(test_inds)
            self.data = {'train': torch.stack([normalized_input_data[i] for i in train_inds]).to(torch.float32), 
                        'test': (torch.stack([normalized_input_data[i] for i in test_inds]).to(torch.float32) if len(test_inds) > 0 else torch.Tensor([]))}
            self.val = {'train': torch.stack([normalized_output_data[i] for i in train_inds]).to(torch.float32), 
                        'test': torch.stack([normalized_output_data[i] for i in test_inds]).to(torch.float32) if len(test_inds) > 0 else torch.Tensor([])}
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
            curr_model_state = self.envs.get_state()
            curr_day = self.curr_day+1
            b_len = self.batch_len 
            output_tens = torch.empty(size=(self.num_envs, b_len, len(self.output_vars))).to(self.device)
            while curr_day < b_len:
                output = self.envs.run(dates=self.curr_dates[:,curr_day])
                normed_output = util.tensor_normalize(output, self.output_range).detach()
                output_tens[:,curr_day] = normed_output.view(normed_output.shape[0],-1)
                curr_day += 1

            # Reset model state back    
            self.envs.set_state(curr_model_state)
        else:
            curr_model_state = self.envs[i].get_state(i=i) if i is not None else self.model.get_state()

            curr_day = self.curr_day[i]+1 if i is not None else self.curr_day+1
            b_len = self.batch_len[i] if i is not None else self.batch_len
            output_tens = torch.empty(size=(1, b_len, len(self.output_vars))).to(self.device)

            while curr_day < b_len:
                output = self.envs[i].run(dates=self.curr_dates[i][:,curr_day]) if i is not None else self.model.run(dates=self.curr_dates[:,curr_day])
                normed_output = util.tensor_normalize(output, self.output_range).detach()
                output_tens[:,curr_day] = normed_output.view(normed_output.shape[0],-1)
                curr_day += 1

            # Reset model state back    
            self.envs[i].set_state(curr_model_state, i=i) if i is not None else self.model.set_state(curr_model_state)

        return output_tens
            
    def posbinary_reward(self, output, val, i=None):
        """Binary reward function"""
        return torch.sum((output == val) * (val != self.target_mask),axis=-1)
    
    def negbinary_reward(self, output, val, i=None):
        """Binary reward function"""
        return -torch.sum((output != val) * (val != self.target_mask),axis=-1)
    
    def todate_reward(self, output, val, i=None):
        """Reward for how much matches to date"""
        if i is None:
            self.reward_sum += torch.sum((output == val) * (val != self.target_mask),axis=-1).flatten()[0] / self.batch_len
        else:
            self.reward_sum[i] += torch.sum((output == val) * (val != self.target_mask),axis=-1).flatten()[0] / self.batch_len[i]

        return self.reward_sum if i is None else self.reward_sum[i]

    def projectionsum_reward(self, output, val, i=None):
        """Reward is the projection sum of past and for params into the future"""
        if isinstance(self.envs, BatchModelEngine):
            self.reward_sum += torch.sum((output == val) * (val != self.target_mask),axis=-1)

            output = self.run_till()
            reward = self.reward_sum + \
                    torch.sum((output[:,self.curr_day:] == self.curr_val[:,self.curr_day:]) * \
                            (self.curr_val[:,self.curr_day:] != self.target_mask))
            return reward / self.batch_len
        else:
            if i is None:
                self.reward_sum += torch.sum((output == val) * (val != self.target_mask),axis=-1).flatten()[0] 
            else:
                self.reward_sum[i] += torch.sum((output == val) * (val != self.target_mask),axis=-1).flatten()[0]

            output = self.run_till(i)
            
            if i is None:
                reward = self.reward_sum + \
                    torch.sum((output[:,self.curr_day:] == self.curr_val[:,self.curr_day:]) * \
                            (self.curr_val[:,self.curr_day:] != self.target_mask)).flatten()[0] 
            else: 
                reward = self.reward_sum[i] + \
                    torch.sum((output[:,self.curr_day[i]:] == self.curr_val[i][:,self.curr_day[i]:]) * \
                            (self.curr_val[i][:,self.curr_day[i]:] != self.target_mask)).flatten()[0]

            return reward / self.batch_len if i is None else reward / self.batch_len[i]
    
    def projection_reward(self, output, val, i=None):
        """Reward is the projection for params into the future"""
        if isinstance(self.envs, BatchModelEngine):

            output = self.run_till()
            reward = self.reward_sum + \
                    torch.sum((output[:,self.curr_day:] == self.curr_val[:,self.curr_day:]) * \
                            (self.curr_val[:,self.curr_day:] != self.target_mask))
            return reward / self.batch_len

        else:
            output = self.run_till(i)
            
            if i is None:
                reward = torch.sum((output[:,self.curr_day:] == self.curr_val[:,self.curr_day:]) * \
                            (self.curr_val[:,self.curr_day:] != self.target_mask)).flatten()[0] 
            else: 
                reward = torch.sum((output[:,self.curr_day[i]:] == self.curr_val[i][:,self.curr_day[i]:]) * \
                            (self.curr_val[i][:,self.curr_day[i]:] != self.target_mask)).flatten()[0]

            return reward / self.batch_len if i is None else reward / self.batch_len[i]

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
        else:
            raise NotImplementedError("Reward function not implemented")