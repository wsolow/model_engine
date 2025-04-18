"""Implementation of  models for phenological development in WOFOST

Classes defined here:
- DVS_Phenology: Implements the algorithms for phenologic development
- Vernalisation: 

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
import datetime
import pickle
import numpy as np
import torch

from traitlets_pcse import HasTraits, Instance, Dict

from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate,Tensor

class Model():

    def __init__(self):
        pass

    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        pass

    def calc_rates(self, day, drv):
        """
        Calculate the rates of change for the state variables
        """
        raise NotImplementedError
        
    def integrate(self, day, delt=1.0):
        """
        Integrate the state variables
        """
        raise NotImplementedError
    
    def get_output(self, vars:list=None):
        """
        Return the output of the model
        """
        raise NotImplementedError
    
    def set_model_params(self, args:dict):
        """
        Set the model phenology parameters from dictionary
        """
        raise NotImplementedError
    
    def get_params(self):
        """
        Return the model parameters as a dictionary
        """
        return {k:getattr(self.params, k) for k in self.params.trait_names()}
    
    def save_model(self, path:str):
        """
        Save the model to pickle
        """
        with open(path, "wb") as fp:
            pickle.dump(self.get_params(),fp)
        fp.close()

    def get_state_rates(self, var: list=None):
        """
        Return the states and rates
        """
        output_vars = []
        if var is None:
            for s in self.states._find_valid_variables():
                output_vars.append(getattr(self.states, s))
            for r in self.rates._find_valid_variables():
                output_vars.append(getattr(self.rates, r))
        else:
            for v in var:
                if v in self.states._find_valid_variables():
                    output_vars.append(getattr(self.states, v))
                elif v in self.rates._find_valid_variables():
                    output_vars.append(getattr(self.rates, v))
        return output_vars
    
    def set_state_rates(self, vars:list|dict):
        """
        Set all states and rates
        """
        if isinstance(vars, dict):
            for k,v in vars.items():
                if k in self.states._find_valid_variables():
                    setattr(self.states, k, v)
                elif j in self.rates._find_valid_variables():
                    setattr(self.rates, k, v)
        elif isinstance(vars, list) or isinstance(vars, np.ndarray) or isinstance(vars, torch.Tensor):
            if len(vars) != len(self.states._find_valid_variables()) + len(self.rates._find_valid_variables()):
                raise ValueError("Length of vars does not match states and rates")
            for i, s in enumerate(self.states._find_valid_variables()):
                setattr(self.states, s, vars[i])
            for j, r in enumerate(self.rates._find_valid_variables()):
                setattr(self.rates, r, vars[j + len(self.states._find_valid_variables())])

class TensorModel(HasTraits, Model):
    """
    Base class for model
    """

    states = Instance(StatesTemplate)
    rates = Instance(RatesTemplate)
    params = Instance(ParamTemplate)
             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device):
        """
        Initialize the model with parameters and states
        """
        self.device = device
        self.params = self.Parameters(parvalues)
        self.kiosk = kiosk

    def set_model_params(self, args:dict):
        """
        Set the model phenology parameters from dictionary
        """
        if isinstance(args, dict):
            for k, v in args.items():
                if k in self.params.trait_names():
                    setattr(self.params, k, v)

class BatchTensorModel(HasTraits, Model):

    """
    Base class for BatchTensorModel
    """

    states = Instance(StatesTemplate)
    rates = Instance(RatesTemplate)
    params = Instance(ParamTemplate)
             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        """
        Initialize the model with parameters and states
        """
        self.device = device
        self.num_models = num_models
        self.params = self.Parameters(parvalues, self.num_models)
        self.kiosk = kiosk
    
    def set_model_params(self, args:dict):
        """
        Set the model phenology parameters from dictionary
        """
        if isinstance(args, dict):
            for k, v in args.items():
                if k in self.params.trait_names():
                    setattr(self.params, k, v.squeeze(1))

class BatchTensorModelFast(HasTraits, Model):

    """
    Base class for BatchTensorModel
    """

    params = Instance(ParamTemplate)
             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        """
        Initialize the model with parameters and states
        """
        self.device = device
        self.num_models = num_models
        self.params = self.Parameters(parvalues, self.num_models)
        self.kiosk = kiosk
    
    def set_model_params(self, args:dict):
        """
        Set the model phenology parameters from dictionary
        """
        if isinstance(args, dict):
            for k, v in args.items():
                if k in self.params.trait_names():
                    setattr(self.params, k, v.squeeze(1))

