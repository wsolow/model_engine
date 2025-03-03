"""
The Engine class in control of running the grape phenology model 
"""
import datetime
from datetime import date
from traitlets_pcse import Instance, HasTraits
from .util import data_loader, config_loader
from .grape_phenology import Grape_Phenology
from .nasapower import NASAPowerWeatherDataProvider, WeatherDataProvider, WeatherDataContainer, FileWeatherDataContainer
import pandas as pd
import numpy as np
from .grape_phenology_differentiable import Differentiable_Grape_Phenology
import torch

class GrapePhenologyEngine(HasTraits):

    """Convenience class for running WOFOST8.0 nutrient and water-limited production

    :param parameterprovider: A ParameterProvider instance providing all parameter values
    :param weatherdataprovider: A WeatherDataProvider object
    :param agromanagement: Agromanagement data
    """
    # sub components for simulation
    weatherdataprovider = Instance(WeatherDataProvider,allow_none=True)
    drv = None
    day = Instance(date)
    YEAR = [1984, 2022]

    def __init__(self, config:dict=None, drv=None, config_fpath:str=None, params:dict=None):
        """
        Initialize GrapePhenologyEngine Class Class
        """
        # Get model configuration
        if config is None:
            self.config, self.model_differentiable = data_loader(config_fpath)
        else:
            if "CropConfig" in config.keys():
                self.config, self.model_differentiable = config_loader(config)
            else:
                self.config, self.model_differentiable = config, False
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.day = self.start_date

        # Driving variables
        if drv is None:
            self.weatherdataprovider = NASAPowerWeatherDataProvider(self.config["latitude"], self.config["longitude"])
            self.drv = self.weatherdataprovider(self.day)
        else:
            self.weatherdataprovider = None
            self.drv = drv

        # initialize model and set params as needed
        if self.model_differentiable:
            self.model = Differentiable_Grape_Phenology(self.start_date, self.config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.output_func = self.get_tensor_output
        else:
            self.model = Grape_Phenology(self.start_date, self.config)
            self.output_func = self.get_output
        if params is not None:
            self.set_model_params(params)

        # Output variables
        self.output_vars = self.config["output_vars"]
        if self.output_vars == None:
            self.output_vars = self.model.get_output_vars()
        self.weather_vars = self.config["weather_vars"]

    def calc_rates(self, day:date, drv:WeatherDataContainer):
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
            self.drv = self.weatherdataprovider(self.day)
        else: 
            self.drv = drv

        # Rate calculation
        self.calc_rates(self.day, self.drv)

        # State integration
        self.integrate(self.day, delt)

    def run(self, date:datetime.date=None, drv=None, days:int=1):
        """
        Advances the system state with given number of days
        """
        if self.model_differentiable:
            if not isinstance(drv, torch.Tensor):
                drv = torch.tensor(drv)
        else: 
            if isinstance(drv, list) or isinstance(drv, np.ndarray):
                drv = FileWeatherDataContainer(**dict(zip(self.weather_vars, drv)))

        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(drv=drv, date=date)

        return self.output_func()

    def run_all(self):
        """
        Run a simulation through termination
        """

        output = [self.get_crop_output()+self.get_weather_output()]
        states = [self.get_state()]
        while self.day < self.end_date:
            daily_output = self.run()
            states.append(self.get_state())
            output.append(daily_output)

        return pd.DataFrame(output, columns=["DAY"]+self.output_vars+self.weather_vars), states

    def get_crop_output(self):
        """
        Return the output of the model
        """
        return [self.day] + self.model.get_output(var=self.output_vars) 

    def get_weather_output(self):
        """
        Get the weather output for the day
        """
        weather = []
        for v in self.weather_vars:
            weather.append(getattr(self.drv, v))
        return weather

    def get_output(self):
        """
        Get all crop and weather output
        """
        return self.get_crop_output()+self.get_weather_output()

    def get_tensor_output(self):
        """
        Get tensor output of model
        """
        return self.model.get_tensor_output()
    
    def zerofy(self):
        """
        Zero out all the rates
        """
        self.model.rates.zerofy()

    def get_param_dict(self):
        """
        Get the parameter dictionary 
        """
        return self.model.get_param_dict()
    
    def set_model_params(self, params:dict|list):
        """
        Set the model parameters
        """
        if params is not None:
            if len(params) == 1 and not isinstance(params, dict):
                params = {"TBASEM":float(params[0])}
            elif not isinstance(params, dict):
                keys = ["TBASEM","TEFFMX","TSUMEM","TSUM1","TSUM2","TSUM3", "TSUM4", "MLDORM","Q10C","CSUMDB"]
                params = dict(zip(keys, params))
            
            for k,v in params.items():
                params[k] = v
            self.model.set_model_params(params)
    
    def save_model(self, path:str):
        """
        Save the model as a dictionary
        """
        self.model.save_model(path)

    def get_state(self):
        """
        Get all the model output and weather output
        """
        return [self.day] + self.model.get_state_rates()

    def set_state(self, s):
        """
        Set the model state
        """
        self.day = s[0]
        self.model.set_state_rates(s[1:])

    def reset(self, params=None, year: int=None, random_year:bool=False):
        """
        Reset the model
        """
        if year is not None:
            self.start_date = self.start_date.replace(year=year)
            self.end_date = self.end_date.replace(year=year)
        elif random_year:
            year = np.random.randint(low=GrapePhenologyEngine.YEAR[0], high=GrapePhenologyEngine.YEAR[1])
            self.start_date = self.start_date.replace(year=year)
            self.end_date = self.end_date.replace(year=year)

        self.day = self.start_date
        self.model.reset(self.day)
        self.set_model_params(params)

        return self.output_func()