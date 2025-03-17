"""Base class for model env, handles processing data"""

import numpy as np 
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from model_engine import util

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
        self.config = config
        self.num_models = num_models
        self.target_mask = -1

    def reset(self):
        """Reset Model with corresponding data"""
        pass

    def step(self, action):
        """Take a step through the environment"""
        pass
    
    def param_cast(self, action):
        """Cast action to params"""
        params_predict = torch.tanh(action) + 1 # convert from tanh
        params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
        return params_predict

    def process_data(self, data, split:int=3):
        """Process all of the initial data"""

        self.output_vars = self.config.ModelConfig.output_vars
        self.input_vars = self.config.ModelConfig.input_vars

        # Get normalized (weather) data 
        normalized_input_data, self.drange = util.embed_and_normalize([d.loc[:,self.input_vars] for d in data])
        input_lens = [len(d) for d in normalized_input_data]
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
        n = len(data)
        inds = np.arange(n)
        np.random.shuffle(inds)
        x = int(np.floor(n/split))
        
        self.data = {'train': torch.stack([normalized_input_data[i] for i in inds][x:]).to(torch.float32), 
                     'test': torch.stack([normalized_input_data[i] for i in inds][:x]).to(torch.float32)}
        self.val = {'train': torch.stack([normalized_output_data[i] for i in inds][x:]).to(torch.float32), 
                    'test': torch.stack([normalized_output_data[i] for i in inds][:x]).to(torch.float32)}
        self.dates = {'train': np.array([dates[i] for i in inds][x:]), 'test':np.array([dates[i] for i in inds][:x])}
        # Get cultivar weather for use with embedding
        if "CULTIVAR" in data[0].columns:
            cultivar_data = torch.tensor([d.loc[0,"CULTIVAR"] for d in data]).to(torch.float32).to(self.device).unsqueeze(1)
            self.cultivars = {'train': torch.stack([cultivar_data[i] for i in inds][x:]).to(torch.float32), 
                    'test': torch.stack([cultivar_data[i] for i in inds][:x]).to(torch.float32)}