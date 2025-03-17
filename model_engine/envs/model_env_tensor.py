import numpy as np 
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from model_engine.engine import get_engine, MultiModelEngine
from model_engine import util
from model_engine.envs.base_env import Base_Env

class Model_Env_Tensor(Base_Env):
    """
    Environment wrapper around model
    """

    def __init__(self, config=None, data=None, num_models=1, **kwargs):
        """
        Initialize gym environment with model
        """
        super(Model_Env_Tensor, self).__init__(config, data)

        self.process_data(data)

        self.params = config.params
        self.params_range = torch.tensor(np.array(self.config.params_range,dtype=np.float32)).to(self.device)

        self.model = get_engine(self.config)(num_models=self.num_models, config=config['ModelConfig'], inputprovider=self.input_data, device=self.device)
        
        init_params = self.model.get_params()
        if isinstance(self.model, MultiModelEngine):
            self.init_params = torch.Tensor([[init_params[i][k] for k in self.params] for i in range(self.num_models)]).to(self.device)
        else: 
            self.init_params = torch.cat([init_params[k][:,None] for k in self.params], dim=-1).to(self.device).view(self.num_models, -1)

        self.observation_space = np.empty(shape=(1 + len(self.output_vars) + len(self.input_vars),))
        self.action_space = np.empty(shape=(len(self.params),))

    def reset(self, **kwargs):
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

        #obs = obs.detach().cpu().numpy().flatten()
        obs = obs.flatten()
        return obs, {}

    def step(self, action):
        """Take a step through the environment"""
        # Update model parameters and get weather
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).to(self.device)
        
        if action.ndim == 1:
            action = action.unsqueeze(0)

        params_predict = self.param_cast(action)
        self.model.set_model_params(params_predict, self.params)
        output = self.model.run(dates=self.curr_dates[:,self.curr_day])
        # Normalize output 
        normed_output = util.tensor_normalize(output, self.output_range).detach()
        normed_output = normed_output.view(normed_output.shape[0],-1)
        obs = torch.cat((normed_output, self.curr_data[:,self.curr_day]),dim=-1)
        
        reward = -torch.sum((normed_output != self.curr_val[:,self.curr_day]) * (self.curr_val[:,self.curr_day] != self.target_mask),axis=-1)
        self.curr_day += 1

        trunc = np.zeros(self.num_models)
        done = np.tile(self.curr_day >= self.batch_len, self.num_models)

        obs = obs.flatten()
        reward = reward.flatten()[0]
        return obs, reward, done, trunc, {}
    
    def param_cast(self, action):
        """Cast action to params"""
        params_predict = torch.tanh(action) + 1 # convert from tanh
        params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
        return params_predict
