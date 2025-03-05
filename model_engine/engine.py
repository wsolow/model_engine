"""
The Engine class in control of running the grape phenology model 
"""
import datetime
from datetime import date
import os
from traitlets_pcse import Instance, HasTraits
from .util import param_loader, get_models
from .weather.nasapower import NASAPowerWeatherDataProvider, WeatherDataProvider
import numpy as np
import torch
from model_engine.models.base_model import TensorModel, BaseModel

class BaseEngine(HasTraits):
    """Wrapper class for models"""
    inputdataprovider = Instance(WeatherDataProvider,allow_none=True)
    

    def __init__(self, config:dict=None, inputprovider=None, device='cpu'):
        self.device = device
        self.config = config
        
        self.start_date = np.datetime64(config['start_date'])
        self.day = self.start_date

        # Output variables
        self.output_vars = self.config["output_vars"]
        self.input_vars = self.config["input_vars"]

        # Driving variables
        if inputprovider is None:
            self.inputdataprovider = NASAPowerWeatherDataProvider(self.config["latitude"], self.config["longitude"])
        else:
            self.inputdataprovider = inputprovider

        # Initialize model
        self.model_constr = get_models(f'{os.path.dirname(os.path.abspath(__file__))}/models')[config["model"]]

    def run(self, date:datetime.date=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(date=date)

        return self.get_output()
       
class SingleModelEngine(BaseEngine):
    """Wrapper class for single engine model"""

    day = Instance(np.datetime64)
    def __init__(self, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize ModelEngine Class
        """
        super().__init__(config, inputprovider, device)

        self.model = self.model_constr(self.start_date, param_loader(self.config), self.device)
    
    def reset(self, i=0, day=None):
        """
        Reset the model
        """
        if day is None:
            self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)

        return self.get_output()
    
    def run(self, date:datetime.date=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(date=date)

        return self.get_output()
    
    def _run(self, date:datetime.date=None, delt=1):
        """
        Make one time step of the simulation.
        """
        # Update day
        if date is None:
            self.day += datetime.timedelta(days=delt)
        else:
            self.day = date
        # Get driving variables
        drv = self.inputdataprovider(self.day)
        # Rate calculation
        self.calc_rates(self.day, drv)

        # State integration
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

    def set_model_params(self, params:dict):
        """
        Set the model parameters
        """
        self.model.set_model_params(params)
    
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
    
class MultiModelEngine(BaseEngine):

    days = Instance(np.ndarray)
    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize MultiModelEngine Class
        """
        super().__init__(config, inputprovider, device)

        self.num_models = num_models
        self.models = [self.model_constr(self.start_date, param_loader(self.config), self.device) for _ in range(self.num_models)]

        assert not isinstance(self.models[0], TensorModel), "Do not use a TensorModel with the MultiEngineModel!"

    def reset(self, num_models=1, days=None):
        """
        Reset all models
        """
        if days is None:
            self.days = np.tile(self.start_date, self.num_models)
        else:
            self.days = days

        [model.reset(self.days[i]) for i, model in enumerate(self.models)]

        return self.get_output(num_models=num_models)
    
    def run(self, dates:datetime.date=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(dates=dates)

        return self.get_output(num_models=len(dates))
    
    def _run(self, dates:datetime.date=None, delt=1):
        """
        Make one time step of the simulation.
        """
        # Update day
        if dates is None:
            for i in range(self.num_models):
                self.days[i] += datetime.timedelta(days=delt)
        else:
            self.days = dates
        # Get driving variables
        drvs = [self.inputdataprovider(self.days[i]) for i in range(len(self.days))]
        # Rate calculation
        self.calc_rates(self.days, drvs)

        # State integration
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

    def set_model_params(self, new_params:torch.Tensor, param_list:list):
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
    
class TensorModelEngine(BaseEngine):
    """Wrapper class for single engine model"""

    days = Instance(np.ndarray)

    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize ModelEngine Class
        """
        super().__init__(config, inputprovider, device)

        self.num_models = num_models
        self.model = self.model_constr(self.start_date, param_loader(self.config), self.device, num_models=self.num_models)
        
        assert not isinstance(self.model, BaseModel), "Model specified is a BaseModel, but we are using the TensorModelEngine as a wrapper!"
    
    def reset(self, num_models=0, day=None):
        """
        Reset the model
        """
        if day is None:
            self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)

        return self.get_output()[:num_models]
    
    def run(self, dates:datetime.date=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(dates=dates)

        return self.get_output()[:len(dates)]
    
    def _run(self, dates:datetime.date=None, delt=1):
        """
        Make one time step of the simulation.
        """
        # Update day
        if dates is None:
            self.day += datetime.timedelta(days=delt)
        else:
            self.day = dates
        # Get driving variables
        drv = self.inputdataprovider(self.day)
        # Rate calculation
        self.calc_rates(self.day, drv)

        # State integration
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

    def set_model_params(self, new_params:torch.Tensor, param_list:list):
        """
        Set the model parameters
        """
        if new_params.shape[0] < self.num_models:
            new_params = torch.nn.functional.pad(new_params, (0,0,0,self.num_models-new_params.shape[0]),value=0)
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
