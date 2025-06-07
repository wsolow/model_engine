"""
engine.py

Contains multiple engines that wrap around a model class for a unified
API.

BaseEngine - base engine class that all others inherit
SingleModelEngine - Assumes that output from model is not batched
MultiModelEngine - Assumes that output from model is not batched but runs multiple
models simultaneously
BatchModelEngine - Assumes that output from model is batched, only compatible with 
batch models
BatchFastModelEngine - Not actually faster, legacy experiment code. Assumes 
output from model is batched and does not create WeatherDataContainer


Written by Will Solow, 2025
"""
import datetime
from datetime import date
import os
import numpy as np
import torch
import torch.nn.functional as F
from traitlets_pcse import Instance, HasTraits
import pandas as pd

from model_engine.util import param_loader, get_models
from model_engine.inputs.nasapower import NASAPowerWeatherDataProvider, WeatherDataProvider
from model_engine.models.base_model import BatchTensorModel, TensorModel
from model_engine.models.states_rates import VariableKiosk

class BaseEngine(HasTraits):
    """
    Base Wrapper class for models
    """
    inputdataprovider = Instance(WeatherDataProvider,allow_none=True)
    
    def __init__(self, config:dict=None, inputprovider=None, device='cpu'):
        self.device = device
        self.config = config
        self.start_date = np.datetime64(config['start_date'])
        self.day = self.start_date
        self.kiosk = VariableKiosk()

        # Output variables
        self.output_vars = self.config["output_vars"]
        self.input_vars = self.config["input_vars"]
        
        # Get the model constructor
        self.model_constr = get_models(f'{os.path.dirname(os.path.abspath(__file__))}/models')[config["model"]]

        if inputprovider is None:
            self.inputdataprovider = NASAPowerWeatherDataProvider(self.config["latitude"], self.config["longitude"])
        else:
            self.inputdataprovider = inputprovider

    def run(self, dates:datetime.date=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(date=dates)

        return self.get_output()
    
    def run_all(self, end_date=datetime.date(2000, 9, 7), same_yr:bool=True):
        """
        Run a simulation through termination
        Generally used to generate synthetic data
        """
        start_date = self.day.astype('datetime64[D]').astype(object)
        end_date = end_date.replace(year=start_date.year) if same_yr else end_date.replace(year=start_date.year+1)
        df = pd.DataFrame(index=range((end_date-start_date).days), columns=self.output_vars+self.input_vars)

        inp = self.get_input(self.day) # Do this first for correct ordering of input
        out = self.get_output().cpu().numpy().flatten()

        df.loc[0] = np.concatenate((out,inp))
        i=1
        while self.day < end_date:
            inp = self.get_input(np.datetime64(self.day.astype('datetime64[D]').tolist()+datetime.timedelta(days=1)))
            out = self.run().cpu().numpy().flatten()
            df.loc[i] = np.concatenate((out,inp))
            i+=1
        return df
    
    def get_state_rates_names(self):
        """
        Get names of valid states and ratse
        """
        return self.model.get_state_rates_names()
    
    def get_input(self, day):
        """
        Get the input to a model on the day
        """
        return np.array([getattr(self.inputdataprovider(day, type(self.model)), var) for var in self.input_vars],dtype=object)

    def get_output(self):
        """
        Get the output of a model
        """
        pass
     
class SingleModelEngine(BaseEngine):
    """
    Wrapper class for single engine model
    Assumes output of model is not batched
    """

    day = Instance(np.datetime64)
    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        super().__init__(config, inputprovider, device)

        self.model = self.model_constr(self.start_date, self.kiosk, param_loader(self.config), self.device)
        
        assert isinstance(self.model, TensorModel), "Model specified is not a Tensor Model, but we are using the SingleModelEngine as a wrapper!"
    def reset(self, year=None, day=None):
        """
        Resets mdoel to specific year and day if not none
        Otherwise resets to default configuration
        """
        if day is None:
            if year is not None:
                day = self.start_date.astype('M8[s]').astype(datetime.datetime).date()
                self.day = np.datetime64(day.replace(year=year))
            else:
                self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)

        return self.get_output()
    
    def run(self, dates:datetime.date=None, cultivar:int=-1, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(date=dates)

        return self.get_output()
    
    def _run(self, date:datetime.date=None, cultivar:int=-1, delt=1):
        """
        Make one time step of the simulation.
        """
        if date is None:
            self.day += np.timedelta64(1, 'D')
        else:
            self.day = date
        drv = self.inputdataprovider(self.day, self.model, cultivar=cultivar)

        self.calc_rates(self.day, drv)
        self.integrate(self.day, delt)
        
    def calc_rates(self, day:date, drv):
        """
        Calculate the rates for computing rate of state change
        """
        self.model.calc_rates(day, drv)

    def integrate(self, day:date, delt:float):
        """
        Integrate rates with states based on time change (delta)
        """
        self.model.integrate(day, delt)

    def set_model_params(self, new_params, param_list=None):
        self.model.set_model_params(dict(zip(param_list,torch.split(new_params,1,dim=-1))))

    def get_output(self):
        """
        Get the observable output of the model
        """
        return self.model.get_output(vars=self.output_vars)

    def get_params(self):
        """
        Get the parameters in the form of a dictionary
        """
        return self.model.get_params()
    
    def get_state(self):
        """
        Get the state of the model
        """
        state = self.model.get_state_rates()
        extra_vars = state[0]

        return [extra_vars, torch.stack(state[1:],dim=-1).to(self.device)]
    
    def set_state(self, state):
        """
        Set the state of the model
        """
        self.model.set_state_rates(state)
        return self.get_output()
    
class MultiModelEngine(BaseEngine):
    """
    Wrapper class for single engine model
    Assumes output of model is not batched
    But runs multiple engines in parallel
    LEGACY: Avoid using this wrapper, as the BatchModelEngine is much faster
    when you have a BatchModel available
    """
    days = Instance(np.ndarray)
    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        super().__init__(config, inputprovider, device)

        self.num_models = num_models
        self.models = [self.model_constr(self.start_date, self.kiosk, param_loader(self.config), self.device) for _ in range(self.num_models)]

        assert isinstance(self.models[0], TensorModel), "Use a TensorModel model with the MultiModelEngine, not a BatchTensorModel!"

    def reset(self, num_models=1, year=None, days=None):
        """
        Reset all models to specific year and day if passed
        """
        if days is None:
            
            if year is not None:
                day = self.start_date.astype('M8[s]').astype(datetime.datetime).date()
                self.days = np.tile(np.datetime64(day.replace(year=year)), self.num_models)
            else:
                self.days = np.tile(self.start_date, self.num_models)
        else:
            self.days = days

        [model.reset(self.days[i]) for i, model in enumerate(self.models)]

        return self.get_output(num_models=num_models)
    
    def run(self, dates:datetime.date=None, cultivars:list=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(dates=dates, cultivars=cultivars)

        return self.get_output(num_models=len(dates))
    
    def _run(self, dates:datetime.date=None, cultivars:list=None, delt=1):
        """
        Make one time step of the simulation.
        """
        if dates is None:
            for i in range(self.num_models):
                self.days[i] += np.timedelta64(1, 'D')
        else:
            self.days = dates
        if cultivars is None:
            drvs = [self.inputdataprovider(self.days[i],type(self.models[0])) for i in range(len(self.days))]
        else:
            drvs = [self.inputdataprovider(self.days[i],type(self.models[0]), cultivar=cultivars[i]) for i in range(len(self.days))]
        
        self.calc_rates(self.days, drvs)
        self.integrate(self.days, delt)

    def calc_rates(self, days:date, drvs):
        """
        Calculate the rates for computing rate of state change
        """
        [self.models[i].calc_rates(days[i], drvs[i]) for i in range(len(days))]

    def integrate(self, days:date, delt:float):
        """
        Integrate rates with states based on time change (delta)
        """
        [self.models[i].integrate(days[i], delt) for i in range(len(days))]

    def set_model_params(self, new_params, param_list:list):
        """
        Set the model parameters
        """
        [self.models[i].set_model_params(dict(zip(param_list,torch.split(new_params[i,:],1,dim=-1)))) for i in range(new_params.shape[0])]
    
    def get_output(self, num_models=1):
        """
        Get the observable output of the model
        """
        return torch.cat([self.models[i].get_output(vars=self.output_vars) for i in range(num_models)])

    def get_params(self):
        """
        Get the parameter dictionary 
        """
        return [self.models[i].get_params() for i in range(self.num_models)]
    
    def get_state(self, num_models=1, i=None):
        """
        Get the hidden state of the model
        """
        extra_vars = []
        states = []
        if i is None:
            for j in range(num_models):
                state = self.models[j].get_state_rates()
                extra_vars.append(state[0])
                states.append(torch.tensor(state[1:]))
            return [extra_vars, torch.stack(states,dim=-1).to(self.device)]
            
        else: 
            return torch.tensor(self.models[0].get_state_rates()).to(self.device)
    
    def set_state(self, state, num_models=1, i=None):
        """
        Set the state of the model
        """
        if i is None:
            [self.models[j].set_state_rates(state[j]) for j in range(num_models)]
        else: 
            self.models[0].set_state_rates(state)
        return self.get_output()
    
class BatchModelEngine(BaseEngine):
    """
    Wrapper class for the BatchModelEngine around Batch Models
    Model must be a tensormodel model. 
    This is the best option for wrapping models if a batch model is 
    available
    """

    days = Instance(np.ndarray)

    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):

        super().__init__(config, inputprovider, device)

        self.num_models = num_models
        self.model = self.model_constr(self.start_date, self.kiosk, param_loader(self.config), self.device, num_models=self.num_models)
        assert isinstance(self.model, BatchTensorModel), "Model specified is not a Batch Tensor Model, but we are using the BatchModelEngine as a wrapper!"
    
    def reset(self, num_models=1, year=None, day=None):
        """
        Reset the model
        """
        if day is None:
            if year is not None:
                day = self.start_date.astype('M8[s]').astype(datetime.datetime).date()
                self.day = np.datetime64(day.replace(year=year))
            else:
                self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)

        return self.get_output()[:num_models]
    
    def run(self, dates:datetime.date=None, cultivars:list=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(dates=dates, cultivars=cultivars)
        return self.get_output()[:len(dates)] if dates is not None else self.get_output()
    
    def _run(self, dates:datetime.date=None, cultivars:list=None, delt=1):
        """
        Make one time step of the simulation.
        """
        if dates is None:
            self.day += np.timedelta64(1, 'D')
            days = self.day
            drv = self.inputdataprovider(self.day, type(self.model), -1)
            drv.to_tensor(self.device)
        else:
            self.day = dates
            # Need to pad outputs to align with batch, we will ignore these in output
            if cultivars is None:
                days = np.pad(self.day, (0, self.num_models-len(self.day)), mode='constant', constant_values=self.day[-1]) \
                            if len(self.day) < self.num_models else self.day
                drv = self.inputdataprovider(days, type(self.model), np.tile(-1, len(days)))
            else:
                days = np.pad(self.day, (0, self.num_models-len(self.day)), mode='constant', constant_values=self.day[-1]) \
                            if len(self.day) < self.num_models else self.day
                cultivars = F.pad(cultivars, (0,0,0, self.num_models-len(cultivars)), mode='constant', value=float(cultivars[-1].cpu().numpy().flatten())) \
                            if len(cultivars) < self.num_models else cultivars
                drv = self.inputdataprovider(days, type(self.model), cultivars)
       
        self.calc_rates(days, drv)
        self.integrate(days, delt)
        
    def calc_rates(self, day:date, drv):
        """
        Calculate the rates for computing rate of state change
        """
        self.model.calc_rates(day, drv)

    def integrate(self, day:date, delt:float):
        """
        Integrate rates with states based on time change (delta)
        """
        self.model.integrate(day, delt)

    def set_model_params(self, new_params, param_list:list):
        """
        Set the model parameters
        """
        if new_params.shape[0] < self.num_models:
            bsize = new_params.shape[0]
            new_params = torch.nn.functional.pad(new_params, (0,0,0,self.num_models-new_params.shape[0]),value=0)
            new_params[bsize:] = new_params[0]
        self.model.set_model_params(dict(zip(param_list,torch.split(new_params,1,dim=-1))))
    
    def get_output(self):
        """
        Get the observable output of the model
        """
        return self.model.get_output(vars=self.output_vars)

    def get_params(self):
        """
        Get the parameter dictionary 
        """
        return self.model.get_params()
    
    def get_state(self, i=None):
        """
        Get the state of the model
        """
        state = self.model.get_state_rates()
        extra_vars = state[0]

        return [extra_vars, torch.stack(state[1:],dim=-1).to(self.device)]

    def set_state(self, state, i=None):
        """
        Set the state of the model
        """
        rep_factor = (self.num_models + state[1].size(0) - 1) // state[1].size(0)
        extra_vars = state[0]

        state_rates = state[1].repeat(rep_factor, 1)[:self.num_models].T
        self.model.set_state_rates([extra_vars, state_rates])

        return self.get_output()

class BatchFastModelEngine(BaseEngine):
    """
    Wrapper class for the BatchModelEngine around Batch Models
    Model must be a tensormodel model. 
    LEGACY: Currently kept, but generally the BatchModelEngine
    is a better option with more support
    """
    days = Instance(np.ndarray)

    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        super().__init__(config, inputprovider, device)
        
        self.drv_vars = self.config["input_vars"]
        if "DAY" in self.drv_vars:
            self.drv_vars.remove("DAY")
        self.drv_vars = dict(zip(self.drv_vars,range(len(self.drv_vars))))
        self.num_models = num_models
        self.model = self.model_constr(self.start_date, self.kiosk, param_loader(self.config), self.device, num_models=self.num_models)
        
        assert isinstance(self.model, BatchTensorModel), "Model must be a BatchTensorModel, as we are using the BatchFastModelEngine as a wrapper!"
    
    def reset(self, num_models=1, year=None, day=None):
        """
        Reset the model
        """
        if day is None:
            if year is not None:
                day = self.start_date.astype('M8[s]').astype(datetime.datetime).date()
                self.day = np.datetime64(day.replace(year=year))
            else:
                self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)
        return self.get_output()[:num_models]
    
    def run(self, dates:datetime.date=None, cultivars:list=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(dates=dates, cultivars=cultivars)
        return self.get_output()[:len(dates)]
    
    def _run(self, dates:datetime.date=None, cultivars:list=None, delt=1):
        """
        Make one time step of the simulation.
        """
        if dates is None:
            self.day += datetime.timedelta(days=delt)
        else:
            self.day = dates
        
        # Need to pad inputs to fit on batch
        if cultivars is None:
            days = np.pad(self.day, (0, self.num_models-len(self.day)), mode='constant', constant_values=self.day[-1]) \
                        if len(self.day) < self.num_models else self.day
            drv = self.inputdataprovider(days, type(self.model), np.tile(-1, len(days)))
        else:
            days = np.pad(self.day, (0, self.num_models-len(self.day)), mode='constant', constant_values=self.day[-1]) \
                        if len(self.day) < self.num_models else self.day
            cultivars = F.pad(cultivars, (0,0,0, self.num_models-len(cultivars)), mode='constant', value=float(cultivars[-1].cpu().numpy().flatten())) \
                        if len(cultivars) < self.num_models else cultivars
            drv = self.inputdataprovider(days, type(self.model), cultivars)

        self.calc_rates(self.day, drv, self.drv_vars)
        self.integrate(self.day, delt)
        
    def calc_rates(self, day:date, drv, drv_vars:dict=None):
        """
        Calculate the rates for computing rate of state change
        """
        self.model.calc_rates(day, drv, drv_vars)

    def integrate(self, day:date, delt:float):
        """
        Integrate rates with states based on time change (delta)
        """
        self.model.integrate(day, delt)

    def set_model_params(self, new_params, param_list:list):
        """
        Set the model parameters
        """
        if new_params.shape[0] < self.num_models:
            bsize = new_params.shape[0]
            new_params = torch.nn.functional.pad(new_params, (0,0,0,self.num_models-new_params.shape[0]),value=0)
            new_params[bsize:] = new_params[0]

        self.model.set_model_params(dict(zip(param_list,torch.split(new_params,1,dim=-1))))
    
    def get_output(self):
        """
        Get the observable output of the model
        """
        return self.model.get_output(vars=self.output_vars)

    def get_params(self):
        """
        Get the parameter dictionary 
        """
        return self.model.get_params()
    
    def get_state(self, i=None):
        """
        Get the state of the model
        """
        return torch.stack(self.model.get_state_rates(),dim=-1).to(self.device)

    def set_state(self, state, i=None):
        """
        Set the state of the model
        """
        rep_factor = (self.num_models + state.size(0) - 1) // state.size(0)

        state = state.repeat(rep_factor, 1)[:self.num_models]
        self.model.set_state_rates(state.T)
        return self.get_output()

def get_engine(config):
    """
    Get the engine constructor and validate that it is correct
    """
    
    if config.ModelConfig.model_type == "Batch":
        if "Fast" in config.ModelConfig.model:
            return BatchFastModelEngine
        if "Batch" in config.ModelConfig.model:
            return BatchModelEngine
        else:
            return MultiModelEngine
    elif config.ModelConfig.model_type == "Single":
        if "Batch" in config.ModelConfig.model or "Fast" in config.ModelConfig.model:
            raise Exception("Incorrect use of Batch Model with SingleEngine")
        else:
            return SingleModelEngine