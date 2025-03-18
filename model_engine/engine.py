"""
The Engine class in control of running the grape phenology model 
"""
import datetime
from datetime import date
import os
import numpy as np
import torch
from traitlets_pcse import Instance, HasTraits
import pandas as pd

from .util import param_loader, get_models
from .inputs.nasapower import NASAPowerWeatherDataProvider, WeatherDataProvider
from model_engine.models.base_model import BatchTensorModel, TensorModel, NumpyModel, BatchNumpyModel

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

        # Initialize model
        self.model_constr = get_models(f'{os.path.dirname(os.path.abspath(__file__))}/models')[config["model"]]

        # Driving variables
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
    
    def run_all(self):
        """
        Run a simulation through termination
        """
        start_date = self.day.astype('datetime64[D]').astype(object)
        end_date = datetime.date(2000, 9, 7).replace(year=start_date.year)
        df = pd.DataFrame(index=range((end_date-start_date).days), columns=self.output_vars+self.input_vars)

        inp = self.get_input(self.day) # Do this first for correct odering
        out = self.get_output().cpu().numpy().flatten()

        df.loc[0] = np.concatenate((out,inp))
        i=1
        while self.day < end_date:
            inp = self.get_input(np.datetime64(self.day.astype('datetime64[D]').tolist()+datetime.timedelta(days=1)))
            out = self.run().cpu().numpy().flatten()
            df.loc[i] = np.concatenate((out,inp))
            i+=1
        return df
    
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
    """Wrapper class for single engine model"""

    day = Instance(np.datetime64)
    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize ModelEngine Class
        """
        super().__init__(config, inputprovider, device)

        self.model = self.model_constr(self.start_date, param_loader(self.config), self.device)
    
    def reset(self, year=None, day=None):
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
        # Update day
        if date is None:
            self.day += np.timedelta64(1, 'D')
        else:
            self.day = date
        # Get driving variables
        drv = self.inputdataprovider(self.day, self.model, cultivar=cultivar)
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

    def set_model_params(self, new_params, param_list=None):
        """
        Set the model parameters
        """
        if isinstance(self.model, NumpyModel):
            self.model.set_model_params(dict(zip(param_list,np.split(new_params,len(param_list),axis=-1))))
        else:
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
    
class MultiModelEngine(BaseEngine):

    days = Instance(np.ndarray)
    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize MultiModelEngine Class
        """
        super().__init__(config, inputprovider, device)

        self.num_models = num_models
        self.models = [self.model_constr(self.start_date, param_loader(self.config), self.device) for _ in range(self.num_models)]

        assert not isinstance(self.models[0], BatchTensorModel), "Do not use a BatchTensorModel with the MultiEngineModel!"

    def reset(self, num_models=1, year=None, days=None):
        """
        Reset all models
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
        # Update day
        if dates is None:
            for i in range(self.num_models):
                self.days[i] += np.timedelta64(1, 'D')
        else:
            self.days = dates
        # Get driving variables
        if cultivars is None:
            drvs = [self.inputdataprovider(self.days[i],type(self.models[0])) for i in range(len(self.days))]
        else:
            drvs = [self.inputdataprovider(self.days[i],type(self.models[0]), cultivar=cultivars[i]) for i in range(len(self.days))]
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

    def set_model_params(self, new_params, param_list:list):
        """
        Set the model parameters
        """
        if isinstance(self.models[0], NumpyModel):
            [self.models[i].set_model_params(dict(zip(param_list,np.split(new_params[i,:],len(param_list),axis=-1)))) for i in range(new_params.shape[0])]
        else:
            [self.models[i].set_model_params(dict(zip(param_list,torch.split(new_params[i,:],1,dim=-1)))) for i in range(new_params.shape[0])]
    
    def get_output(self, num_models=1):
        """
        Get the observable output of the model
        """
        if isinstance(self.models[0], NumpyModel):
            return np.concatenate([self.models[i].get_output(vars=self.output_vars) for i in range(num_models)])
        else:
            return torch.cat([self.models[i].get_output(vars=self.output_vars) for i in range(num_models)])

    def get_params(self):
        """
        Get the parameter dictionary 
        """
        return [self.models[i].get_params() for i in range(self.num_models)]
    
class BatchModelEngine(BaseEngine):
    """Wrapper class for single engine model"""

    days = Instance(np.ndarray)

    def __init__(self, num_models:int=1, config:dict=None, inputprovider=None, device='cpu'):
        """
        Initialize ModelEngine Class
        """
        super().__init__(config, inputprovider, device)

        self.num_models = num_models
        self.model = self.model_constr(self.start_date, param_loader(self.config), self.device, num_models=self.num_models)
        
        assert not (isinstance(self.model, TensorModel) or isinstance(self.model, NumpyModel)), "Model specified is a Tensor or Numpy Model, but we are using the BatchModelEngine as a wrapper!"
    
    def reset(self, num_models=0, year=None, day=None):
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
        # Update day
        if dates is None:
            self.day += datetime.timedelta(days=delt)
        else:
            self.day = dates
        # Get driving variables
        if cultivars is None:
            drv = self.inputdataprovider(self.day, type(self.model), np.tile(-1, len(self.day)))
        else:
            drv = self.inputdataprovider(self.day, type(self.model), cultivars)
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

    def set_model_params(self, new_params, param_list:list):
        """
        Set the model parameters
        """
        if new_params.shape[0] < self.num_models:
            new_params = torch.nn.functional.pad(new_params, (0,0,0,self.num_models-new_params.shape[0]),value=0)
        if isinstance(self.model, BatchNumpyModel):
            if isinstance(new_params, torch.Tensor):
                new_params = new_params.detach().cpu().numpy()
            self.model.set_model_params(dict(zip(param_list,np.split(new_params,len(param_list),axis=-1))))
        else:
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

def get_engine(config):
    """
    Get the engine constructor and validate that it is correct
    """
    
    if config.ModelConfig.model_type == "Batch":
        if "TensorBatch" in config.ModelConfig.model:
            return BatchModelEngine
        elif "NumpyBatch" in config.ModelConfig.model:
            return BatchModelEngine
        else:
            return MultiModelEngine
    elif config.ModelConfig.model_type == "Single":
        if "TensorBatch" in config.ModelConfig.model or "NumpyBatch" in config.ModelConfig.model:
            raise Exception("Incorrect use of Batch Model with SingleEngine")
        else:
            return SingleModelEngine