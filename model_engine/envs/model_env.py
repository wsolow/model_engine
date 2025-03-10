import numpy as np 
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from model_engine.engine import get_engine, MultiModelEngine
from model_engine import util

class Model_Env():
    """
    Environment wrapper around model
    """

    def __init__(self, config, data, num_models=1):
        """
        Initialize gym environment with model
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.config = config
        self.num_models = num_models
        self.target_mask = -1

        self.process_data(data)

        self.params = config.params
        self.params_range = np.array(config.params_range,dtype=np.float32)

        self.model = get_engine(self.config)(num_models=self.num_models, config=config['ModelConfig'], inputprovider=self.input_data, device=self.device)
        
        init_params = self.model.get_params()
        if isinstance(self.model, MultiModelEngine):
            self.init_params = torch.Tensor([[init_params[i][k] for k in self.params] for i in range(self.num_models)]).to(self.device)
        else: 
            self.init_params = torch.cat([init_params[k][:,None] for k in self.params], dim=-1).to(self.device).view(self.num_models, -1)

        self.observation_space = 1 + len(self.output_vars) + len(self.input_vars)
        self.action_space = len(self.params)

    def reset(self):
        """Reset Model with corresponding data"""

        # Shuffle data and record length
        inds = np.arange(len(self.data['train']))
        np.random.shuffle(inds)
        self.curr_data = self.data['train'][inds[:self.num_models]]
        self.curr_val = self.val['train'][inds[:self.num_models]]
        self.curr_dates = self.dates['train'][inds[:self.num_models]]
        self.curr_params = self.init_params.detach().clone().requires_grad_(requires_grad=True)
        # Get current batch and sequence length
        self.batch_len = self.curr_data.shape[1]
        self.curr_day = 1

        output = self.model.reset(self.num_models)
        # Cat waether onto obs
        normed_output = util.tensor_normalize(output, self.output_range).detach()
        normed_output = normed_output.view(normed_output.shape[0],-1)
        obs = torch.cat((normed_output, self.curr_data[:,0]),dim=-1)

        return obs, {}

    def step(self, action):
        """Take a step through the environment"""
        # Update model parameters and get weather
        self.model.set_model_params(action, self.params)
        output = self.model.run(dates=self.curr_dates[:,self.curr_day])
        # Normalize output 
        normed_output = util.tensor_normalize(output, self.output_range).detach()
        normed_output = normed_output.view(normed_output.shape[0],-1)
        obs = torch.cat((normed_output, self.curr_data[:,self.curr_day]),dim=-1)

        reward = -torch.sum((normed_output != self.curr_val[:,self.curr_day]) * (self.curr_val[:,self.curr_day] != self.target_mask))
        self.curr_day += 1
        done = trunc = self.curr_day >= self.batch_len
        return obs, reward, done, trunc, {}


    def process_data(self, data):
        """Process all of the initial data"""

        self.output_vars = self.config.ModelConfig.output_vars
        self.input_vars = self.config.ModelConfig.input_vars

        # Get normalized (weather) data 
        normalized_input_data, self.drange = util.embed_and_normalize([d.loc[:,self.input_vars] for d in data])
        input_lens = [len(d) for d in normalized_input_data]
        normalized_input_data = pad_sequence(normalized_input_data, batch_first=True, padding_value=0).to(self.device)
        
        self.drange = self.drange.to(self.device)

        # Get input data for use with model to avoid unnormalizing
        self.input_data = util.make_inputs([d.loc[:,self.input_vars] for d in data])
        
        # Get validation data
        normalized_output_data, self.output_range = util.embed_output([d.loc[:,self.output_vars] for d in data])
        normalized_output_data = pad_sequence(normalized_output_data, batch_first=True, padding_value=self.target_mask).to(self.device)
        self.output_range = self.output_range.to(self.device)

        # Get the dates
        dates = [d.loc[:,"DAY"].to_numpy().astype('datetime64[D]') for d in data]
        max_len = max(len(arr) for arr in dates)
        # Pad each array to the maximum length
        dates = [np.pad(arr, (0, max_len - len(arr)), mode='maximum') for arr in dates]

        # Shuffle to get train and test splits for data
        # 2:1 train/test split
        n = len(data)
        inds = np.arange(n)
        np.random.shuffle(inds)
        x = int(np.floor(n/3))
        
        self.data = {'train': torch.stack([normalized_input_data[i] for i in inds][x:]).to(torch.float32), 
                     'test': torch.stack([normalized_input_data[i] for i in inds][:x]).to(torch.float32)}
        self.val = {'train': torch.stack([normalized_output_data[i] for i in inds][x:]).to(torch.float32), 
                    'test': torch.stack([normalized_output_data[i] for i in inds][:x]).to(torch.float32)}
        self.dates = {'train': np.array([dates[i] for i in inds][x:]), 'test':np.array([dates[i] for i in inds][:x])}
