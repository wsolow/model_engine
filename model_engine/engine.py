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


class ModelEngine(HasTraits):
    """Wrapper class for models"""
    # sub components for simulation
    inputdataprovider = Instance(WeatherDataProvider,allow_none=True)
    drv = None
    day = Instance(np.datetime64)
    YEAR = [1984, 2022]

    def __init__(self, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize ModelEngine Class
        """
        self.device = device
        self.config = config

        #self.start_date = datetime.datetime.strptime(config['start_date'], "%Y-%m-%d")
        self.start_date = np.datetime64(config['start_date'])
        self.day = self.start_date

        # Driving variables
        if inputprovider is None:
            self.inputdataprovider = NASAPowerWeatherDataProvider(self.config["latitude"], self.config["longitude"])
        else:
            self.inputdataprovider = inputprovider

        # Initialize model
        model_constr = get_models(f'{os.path.dirname(os.path.abspath(__file__))}/models')[config["model"]]
        self.model = model_constr(self.start_date, param_loader(self.config), device)

        # Output variables
        self.output_vars = self.config["output_vars"]
        self.input_vars = self.config["input_vars"]

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

        # Set all rate variables to zero
        #self.zerofy()

    def reset(self, day=None):
        """
        Reset the model
        """
        if day is None:
            self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)

        return self.get_output()
    
    def _run(self, drv:None, date:datetime.date=None, delt=1):
        """
        Make one time step of the simulation.
        """
        # Update day
        if date is None:
            self.day += datetime.timedelta(days=delt)
        else:
            self.day = date
        # Get driving variables
        if drv is None:
            drv = self.inputdataprovider(self.day)
        # Rate calculation
        self.calc_rates(self.day, drv)

        # State integration
        self.integrate(self.day, delt)

    def run(self, date:datetime.date=None, drv=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        if drv is not None:
            if not isinstance(drv, torch.Tensor):
                drv = torch.tensor(drv)
        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(drv=drv, date=date)

        return self.get_output()
    
    def set_model_params(self, params:dict):
        """
        Set the model parameters
        """
        self.model.set_model_params(params)
    
    def get_crop_output(self):
        """
        Return the output of the model
        """
        return [self.day] + self.model.get_state_rates(var=self.output_vars) 

    def get_output(self):
        """
        Get the observable output of the model
        """
        # Vars=None
        return self.model.get_output(vars=self.output_vars)
    
    def zerofy(self):
        """
        Zero out all the rates
        """
        self.model.rates.zerofy()

    def get_params(self):
        """
        Get the parameter dictionary 
        """
        return self.model.get_params()
    
    def save_model(self, path:str):
        """
        Save the model as a dictionary
        """
        self.model.save_model(path)

