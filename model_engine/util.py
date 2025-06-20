"""
util.py 

Utility functions for the model_egnine class

Modified by Will Solow, 2024
"""
import yaml
import os
import pandas as pd
import numpy as np
import datetime
import torch
import pickle
from inspect import getmembers, isclass
import importlib.util 

from model_engine.models.base_model import Model
from model_engine.inputs.input_providers import MultiTensorWeatherDataProvider, MultiTensorProvider

EPS = 1e-12
PHENOLOGY_INT = {"Ecodorm":0, "Budbreak":1, "Flowering":2, "Veraison":3, "Ripe":4}

# Available cultivars for simulation
CROP_NAMES = {'grape_phenology':["Aligote", "Alvarinho", "Auxerrois", "Barbera", "Cabernet_Franc", 
                   "Cabernet_Sauvignon", "Chardonnay", "Chenin_Blanc", "Concord",
                    "Durif", "Gewurztraminer", "Green_Veltliner", "Grenache",  # Dolcetto is also absent as no valid years
                   "Lemberger", "Malbec", "Melon", "Merlot", "Mourvedre", "Muscat_Blanc", "Nebbiolo", 
                   "Petit_Verdot", "Pinot_Blanc", "Pinot_Gris", "Pinot_Noir", "Riesling", 
                   "Sangiovese", "Sauvignon_Blanc", "Semillon", "Tempranillo", # NOTE: Syrah is removed currently
                   "Viognier", "Zinfandel"], 
                'grape_coldhardiness': 
                ["Barbera", "Cabernet_Franc", # Removed Alvarinho, Auxerrois, Melon, Aligote, 
                   "Cabernet_Sauvignon", "Chardonnay", "Chenin_Blanc", "Concord",
                    "Gewurztraminer", "Grenache",  # Green_Veltliner Dolcetto is also absent as no valid years
                   "Lemberger", "Malbec", "Merlot", "Mourvedre", "Nebbiolo", # Muscat_Blanc
                   "Pinot_Gris", "Riesling", # Petit Verdot Pinot_Blanc Pinot_Noir
                   "Sangiovese", "Sauvignon_Blanc", "Semillon", "Syrah", # Tempranillo
                   "Viognier", "Zinfandel"],
                'wofost':
                ["Winter_Wheat_101", "Winter_Wheat_102", "Winter_Wheat_103", "Winter_Wheat_104", 
                 "Winter_Wheat_105", "Winter_Wheat_106", "Winter_Wheat_107", "Bermude",
                 "Apache"]}

def param_loader(config:dict):
    """
    Load the configuration of a model from dictionary
    """
    try:
        model_name, model_num = config['model_parameters'].split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config['model_parameters']}`")
    
    fname = f"{os.getcwd()}/{config['config_fpath']}{model_name}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")

    try:
        cv = model["ModelParameters"]["Sets"][model_num] 
    except:
        raise Exception(f"Incorrectly specified parameter file {fname}. Ensure that `{model_name}` contains parameter set `{model_num}`")

    for c in cv.keys():
        cv[c] = cv[c][0]

    return cv

def per_task_param_loader(config:dict, params):
    """
    Load the available configurations of a model from dictionary and put them on tensor
    """
    try:
        model_name, model_num = config['model_parameters'].split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config['model_parameters']}`")
    
    fname = f"{os.getcwd()}/{config['config_fpath']}{model_name}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")
    
    init_params = []
    for n in CROP_NAMES[model_name]:
        try:
            cv = model["ModelParameters"]["Sets"][n] 
        except:
            raise Exception(f"Incorrectly specified parameter file {fname}. Ensure that `{model_name}` contains parameter set `{model_num}`")

        task_params = []
        for c in cv.keys():
            if c in params:
                task_params.append(cv[c][0])
        init_params.append(task_params)

    return torch.tensor(init_params)

def get_models(folder_path):
    """
    Get all the models in the /models/ folder
    """
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
                if isclass(obj) and (issubclass(obj, Model)):
                    constructors[f'{name}'] = obj
        elif os.path.isdir(f"{folder_path}/{filename}"): # is directory
            constr = get_models(f"{folder_path}/{filename}")
            constructors = constructors | constr

    return constructors   

def load_data(path):
    """
    Load data from a pickle file
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    for d in data:
        d.rename(columns={'DATE': 'DAY'}, inplace=True)
    return data

def load_data_multi(path, cultivars):
    """
    Load pickle files for cultivar data
    given the passed cultivars
    """
    data = []
    for i,c in enumerate(cultivars):

        with open(f"{path}{c}.pkl", "rb") as f:
            cult_data = pickle.load(f)
        for cult in cult_data:
            cult["CULTIVAR"] = i
            data.append(cult) 
    for d in data:
        d.rename(columns={'DATE': 'DAY'}, inplace=True)
    return data

def int_to_day_of_year(day_number):
    """
    Converts an integer representing the day of the year to a date.
    """
    return datetime.datetime(1900, 1, 1) + datetime.timedelta(days=day_number - 1)

def tensor_appendleft(tensor: torch.Tensor, new_values: torch.Tensor) -> torch.Tensor:
    """
    Insert new_values at the left (index 0), shift tensor right, and drop last elements.
    Supports 1D or 2D tensors.
    """
    # Make sure new_values can broadcast to tensor shape except last dim = 1
    if not isinstance(new_values, torch.Tensor):
        new_values = torch.tensor(new_values).to(tensor.device)
    new_values = new_values.unsqueeze(-1) if new_values.ndim == 0 else new_values
    new_values = new_values.unsqueeze(-1) if new_values.dim() == tensor.dim() - 1 else new_values
    
    # Shift right by slicing all except last element on last dim
    shifted = torch.cat([new_values, tensor[..., :-1]], dim=-1)
    return shifted

def tensor_pop(tensor: torch.Tensor, fill_value=0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove and return the last element from tensor along last dimension,
    shift everything left, fill last position with fill_value.
    Returns (shifted_tensor, popped_values).
    Supports 1D or 2D tensors.
    """
    popped = tensor[..., -1].clone()
    shifted = torch.cat([tensor[..., 1:], torch.full_like(tensor[..., -1:], fill_value)], dim=-1)

    return shifted, popped
