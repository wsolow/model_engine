import numpy as np 
import random
import torch
from model_engine.envs.base_env import Base_Env
from model_engine.engine import get_engine, MultiModelEngine, SingleModelEngine
from model_engine import util

class Model_Env(Base_Env):
    """
    Environment wrapper around model
    """

    def __init__(self, config=None, data=None, num_models=1):
        """
        Initialize gym environment with model
        """
        super(Model_Env, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.config = config
        self.num_models = num_models
        self.target_mask = -1

        self.process_data(data)

        self.params = config.params
        self.params_range = np.array(config.params_range,dtype=np.float32)

        if self.config.ModelConfig.model_type == "Single":
            self.model = get_engine(self.config)(config=config['ModelConfig'], inputprovider=self.input_data, device=self.device)
            self.num_models = 1
        else: 
            self.model = get_engine(self.config)(num_models=self.num_models, config=config['ModelConfig'], inputprovider=self.input_data, device=self.device)
        
        init_params = self.model.get_params()
        if isinstance(self.model, MultiModelEngine):
            self.init_params = np.array([[init_params[i][k] for k in self.params] for i in range(self.num_models)])
        else: 
            self.init_params = np.concatenate([init_params[k][:,None] for k in self.params], axis=-1).reshape(self.num_models, -1)
        self.observation_space = np.empty(shape=(1 + len(self.output_vars) + len(self.input_vars),))
        self.action_space = np.empty(low=-np.inf, high=np.inf, shape=(len(self.params),))

    def reset(self, curr_data=None, curr_val=None, curr_dates=None):
        """Reset Model with corresponding data"""

        if curr_data is None: 
            # Shuffle data and record length
            inds = np.arange(len(self.data['train']))
            np.random.shuffle(inds)
            self.curr_data = self.data['train'][inds[:self.num_models]]
            self.curr_val = self.val['train'][inds[:self.num_models]]
            self.curr_dates = self.dates['train'][inds[:self.num_models]]
        else:
            assert curr_val is not None and curr_dates is not None, "All inputs must not be none"
            self.curr_data = np.expand_dims(curr_data,axis=0)
            self.curr_val = np.expand_dims(curr_val,axis=0)
            self.curr_dates = np.expand_dims(curr_dates,axis=0)
        self.curr_params = self.init_params.copy()
        # Get current batch and sequence length
        self.batch_len = self.curr_data.shape[1]
        self.curr_day = 1

        output = self.model.reset(self.num_models)
        # Cat weather onto obs
        normed_output = util.normalize(output, self.output_range)
        normed_output = normed_output.reshape(normed_output.shape[0],-1)
        obs = np.concatenate((normed_output, self.curr_data[:,0]),axis=-1)
        obs = obs.flatten() # TODO handle shape misamtch
        return obs, {}

    def step(self, action):
        """Take a step through the environment"""
        # Update model parameters and get weather
        if action.ndim == 1:
            action = np.expand_dims(action, axis=0)

        # Cast to range
        params_predict = self.param_cast(action)
        self.model.set_model_params(params_predict, self.params)
        # Run Model
        if isinstance(self.model, SingleModelEngine):
            output = self.model.run(dates=self.curr_dates[:,self.curr_day][0])
        else: 
            output = self.model.run(dates=self.curr_dates[:,self.curr_day])
        # Normalize output 
        normed_output = util.normalize(output, self.output_range)
        normed_output = normed_output.reshape(normed_output.shape[0],-1)
        obs = np.concatenate((normed_output, self.curr_data[:,self.curr_day]),axis=-1)
        
        reward = -np.sum((normed_output != self.curr_val[:,self.curr_day]) * (self.curr_val[:,self.curr_day] != self.target_mask),axis=-1)
        self.curr_day += 1
        trunc = np.zeros(self.num_models)
        done = np.tile(self.curr_day >= self.batch_len, self.num_models)

        obs = obs.flatten() # Handle shape mismatch
        reward = reward.flatten()[0] #TODO handle shape mismatch
        return obs, reward, done, trunc, {}

    def param_cast(self, action):
        """Cast action to params"""
        params_predict = np.tanh(action) + 1 # convert from tanh
        params_predict = self.params_range[:,0] + params_predict * (self.params_range[:,1]-self.params_range[:,0]) / 2
        return params_predict
    