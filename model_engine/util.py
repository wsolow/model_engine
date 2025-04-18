"""
Miscellaneous utilities for PCSE

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
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
from model_engine.inputs.input_providers import MultiTensorWeatherDataProvider, WeatherDataProvider, MultiTensorProvider
    
EPS = 1e-12

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
                if isclass(obj) and (issubclass(obj, Model)):
                    constructors[f'{name}'] = obj
        elif os.path.isdir(f"{folder_path}/{filename}"): # is directory
            constr = get_models(f"{folder_path}/{filename}")
            constructors = constructors | constr
    return constructors   

def make_tensor_inputs(config, dfs):
    """
    Make input providers based on the given data frames
    """
    try:
        model_name, model_num = config.ModelConfig.model_parameters.split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config.ModelConfig.model_parameters}`")
    
    if config.reduced_years:
        prefix = f"{model_name}_reduced"
    else:
        prefix = f"{model_name}_extra"
    
    if "Fast" in config.ModelConfig.model:
        fname = f"data/weather_providers/{prefix}_fast_{config.cultivar}.pkl"
    else:
        fname = f"data/weather_providers/{prefix}_{config.cultivar}.pkl"
        
    '''if os.path.exists(fname): # Removed this code as it led to too many errors
        if "Fast" in config.ModelConfig.model:
            wp = MultiTensorProvider()
        else:
            wp = MultiTensorWeatherDataProvider()
        wp._load(fname)
    else:'''
    if "Fast" in config.ModelConfig.model:
        wp = MultiTensorProvider(pd.concat(dfs, ignore_index=True))
    else:
        wp = MultiTensorWeatherDataProvider(pd.concat(dfs, ignore_index=True)) 
    wp._dump(fname)
    return wp

def embed_and_normalize_dvs(data):
    """
    Embed datetime and normalize all data
    """
    tens = []
    data = [d.drop("DVS",axis=1) for d in data]
    data_max = np.max([np.max(d,axis=0) for d in data],axis=0)
    data_max = np.concatenate(([1,1], np.delete(data_max, 0,0))).astype(np.float32)
    data_min = np.min([np.min(d,axis=0) for d in data], axis=0)
    data_min = np.concatenate(([-1,-1], np.delete(data_min, 0,0))).astype(np.float32)

    for d in data:
        d = d.to_numpy()
        dt = np.reshape([ date_to_cyclic(d[i,0]) for i in range(len(d[:,0]))], (-1,2))
        # Concatenate after deleting original date column
        d = np.concatenate((dt, np.delete(d, 0, 1)),axis=1).astype(np.float64)
        
        # Min max normalization
        d = (d - data_min) / (data_max - data_min + EPS)
        tens.append(torch.tensor(d,dtype=torch.float32))
    return tens, torch.tensor(np.stack((data_min,data_max),axis=-1))

def embed_and_normalize(data):
    """
    Embed datetime and normalize all data
    """
    tens = []
    data_max = np.max([np.max(d,axis=0) for d in data],axis=0)
    data_min = np.min([np.min(d,axis=0) for d in data], axis=0)
    
    #data_max = np.concatenate(([1], np.delete(data_max, 0,0))).astype(np.float32)
    #data_min = np.concatenate(([0], np.delete(data_min, 0,0))).astype(np.float32)

    data_max = np.concatenate(([1,1], np.delete(data_max, 0,0))).astype(np.float32)
    data_min = np.concatenate(([-1,-1], np.delete(data_min, 0,0))).astype(np.float32)

    for d in data:
        d = d.to_numpy()
        dt = np.reshape([ date_to_cyclic(d[i,0]) for i in range(len(d[:,0]))], (-1,2))

        # Concatenate after deleting original date column
        d = np.concatenate((dt, np.delete(d, 0, 1)),axis=1).astype(np.float64)
        #dt = np.array([ date_to_frac(d[i,0]) for i in range(len(d[:,0]))])[:,np.newaxis]
        #d = np.concatenate((dt, np.delete(d, 0, 1)),axis=1).astype(np.float64)
        
        # Min max normalization
        d = (d - data_min) / (data_max - data_min + EPS)
        tens.append(torch.tensor(d,dtype=torch.float32))    
    return tens, torch.tensor(np.stack((data_min,data_max),axis=-1))

def embed_output(data):
    """
    Embed datetime and normalize output data
    """
    tens = []
    data_max = np.max([np.max(d,axis=0) for d in data],axis=0)
    data_min = np.min([np.min(d,axis=0) for d in data], axis=0)
    
    for d in data:
        d = d.to_numpy()

        # Concatenate after deleting original date column
        # Min max normalization
        d = (d - data_min) / (data_max - data_min + EPS)
        tens.append(torch.tensor(d,dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_min,data_max),axis=-1))

def embed_cultivar(data):
    """
    Embed datetime and normalize output data
    """
    tens = []
    data_max = np.max([np.max(d,axis=0) for d in data],axis=0)
    data_min = np.min([np.min(d,axis=0) for d in data], axis=0)
    
    for d in data:

        # Concatenate after deleting original date column
        # Min max normalization
        d = (d - data_min) / (data_max - data_min + EPS)
        tens.append(torch.tensor(d,dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_min,data_max),axis=-1))


def date_to_cyclic(date_str):
    """
    Convert datetime to cyclic embedding
    """
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    elif isinstance(date_str, datetime.date):
        date_obj = date_str
    else:
        msg = "Invalid type to convert to date"
        raise Exception(msg)
    day_of_year = date_obj.timetuple().tm_yday
    year_sin = np.sin(2 * np.pi * day_of_year / 365)
    year_cos = np.cos(2 * np.pi * day_of_year / 365)
    return [year_sin, year_cos]

def date_to_frac(date_str):
    """
    Convert datetime to fraction embedding
    """
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    elif isinstance(date_str, datetime.date):
        date_obj = date_str
    else:
        msg = "Invalid type to convert to date"
        raise Exception(msg)
    day_of_year = date_obj.timetuple().tm_yday

    return day_of_year / 365

def normalize(data, drange):
    """
    Normalize data given a range
    """
    return (data - drange[:,0]) / (drange[:,1] - drange[:,0] + EPS)

def unnormalize(data, drange):
    """
    Unnormalize data given a range
    """
    return data * (drange[:,1] - drange[:,0] + EPS) + drange[:,0]

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
    """Converts an integer representing the day of the year to a date."""
    #return np.datetime64(datetime.datetime(1900, 1, 1) + datetime.timedelta(days=day_number - 1))
    return datetime.datetime(1900, 1, 1) + datetime.timedelta(days=day_number - 1)


