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
PHENOLOGY_INT = {"Ecodorm":0, "Budbreak":1, "Flowering":2, "Veraison":3, "Ripe":4, "Endodorm":5}

# Available cultivars for simulation
GRAPE_NAMES = {'grape_phenology':["Aligote", "Alvarinho", "Auxerrois", "Barbera", "Cabernet_Franc", 
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
                   "Viognier", "Zinfandel"]}

CROP_NAMES = {'wofost': ["wheat"]}

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
    for n in GRAPE_NAMES[model_name]:
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

def make_tensor_inputs(config, dfs):
    """
    Make input providers based on the given data frames
    Converts data frames to tensor table 
    """
        
    if "Fast" in config.ModelConfig.model:
        wp = MultiTensorProvider(pd.concat(dfs, ignore_index=True))
    else:
        wp = MultiTensorWeatherDataProvider(pd.concat(dfs, ignore_index=True)) 

    return wp

def embed_and_normalize_minmax_dvs(data):
    """
    Embed datetime and normalize all data based on min/max normalization
    Embeds date as cyclic so it takes up two features
    """
    tens = []
    # Find min and max ranges and concatenate on min/max range for sin/consine embedding
    # of date
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

def embed_and_normalize_minmax(data):
    """
    Embed datetime and normalize all data using min/max normalization
    """
    tens = []
    # Find min and max ranges and concatenate on min/max range for sin/consine embedding
    # of date
    stacked_data = np.vstack([d.to_numpy()[:,1:] for d in data]).astype(np.float32)
    data_max = np.nanmax(stacked_data,axis=0)
    data_min = np.nanmin(stacked_data, axis=0)
    data_max = np.concatenate(([1,1], data_max)).astype(np.float32)
    data_min = np.concatenate(([-1,-1], data_min)).astype(np.float32)

    for d in data:
        d = d.to_numpy()
        dt = np.reshape([ date_to_cyclic(d[i,0]) for i in range(len(d[:,0]))], (-1,2))
        # Concatenate after deleting original date column
        d = np.concatenate((dt, d[:,1:]),axis=1).astype(np.float32)
    
        # Min max normalization
        d = (d - data_min) / (data_max - data_min + EPS)
        tens.append(torch.tensor(d,dtype=torch.float32))    
    return tens, torch.tensor(np.stack((data_min,data_max),axis=-1))

def embed_and_normalize_zscore(data):
    """
    Embed and normalize all data using z-score normalization
    """
    tens = []
    stacked_data = np.vstack([d.to_numpy()[:,1:] for d in data]).astype(np.float32)
    data_mean = np.nanmean(stacked_data,axis=0).astype(np.float32)
    data_std = np.std(stacked_data,axis=0).astype(np.float32)
    data_mean = np.concatenate(([0,0], data_mean)).astype(np.float32)
    data_std = np.concatenate(([1/np.sqrt(2),1/np.sqrt(2)], data_std)).astype(np.float32)
    for d in data:
        d = d.to_numpy()
        # Z-score normalization
        dt = np.reshape([ date_to_cyclic(d[i,0]) for i in range(len(d[:,0]))], (-1,2))
        d = np.concatenate((dt, d[:,1:]),axis=1).astype(np.float32)
        d = (d - data_mean) / (data_std + EPS)

        tens.append(torch.tensor(d.astype(np.float32),dtype=torch.float32))    
    return tens, torch.tensor(np.stack((data_mean,data_std),axis=-1))

def embed_output_minmax(data):
    """
    Normalize output data and return ranges
    """
    tens = []
    
    stacked_data = np.vstack([d.to_numpy() for d in data]).astype(np.float32)
    data_max = np.nanmax(stacked_data,axis=0)
    data_min = np.nanmin(stacked_data, axis=0)

    for d in data:
        d = d.to_numpy()

        # Concatenate after deleting original date column
        # Min max normalization
        d = (d - data_min) / (data_max - data_min + EPS)
        tens.append(torch.tensor(d,dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_min,data_max),axis=-1))

def embed_output(data):
    """
    Normalize output data and return ranges
    """
    tens = []
    
    for d in data:
        d = d.to_numpy()

        # Concatenate after deleting original date column
        # Min max normalization
        tens.append(torch.tensor(d,dtype=torch.float32))

    return tens

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


