"""
Miscellaneous utilities for PCSE

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
import yaml
import os
import pandas as pd
from inspect import getmembers, isclass
import importlib.util 
from model_engine.models.base_model import BaseModel, TensorModel
from model_engine.weather.nasapower import DFWeatherDataProvider
    
def param_loader(config:dict):
    """
    Load the configuration of a model from dictionary
    """

    model = yaml.safe_load(open(f"{os.getcwd()}/{config['config_fpath']}{config['model']}.yaml"))

    cv = model["ModelParameters"]["Sets"][config["model_parameters"]]  

    for c in cv.keys():
        cv[c] = cv[c][0]

    for k,v in model.items():
        cv[k] = v

    return cv

def get_models(folder_path):
    """Get all the models in the /models/ folder"""
    constructors = {}
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.py'):
            file_path = os.path.join(folder_path, filename)
            
            # Remove the .py extension
            module_name = filename[:-3]  
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for name, obj in getmembers(module):
                if isclass(obj) and (issubclass(obj, BaseModel) or issubclass(obj,TensorModel)):
                    constructors[f'{name}'] = obj
    return constructors         
    
def make_inputs(dfs):
    """
    Make input providers based on the given data frames
    """
    return DFWeatherDataProvider(pd.concat(dfs, ignore_index=True)) 