"""Handles Respiration of crop 

Written by: Will Solow, 2025
"""
from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WOFOST_Maintenance_Respiration_TensorBatch(BatchTensorModel):
    """Maintenance respiration in WOFOST
    """
    
    class Parameters(ParamTemplate):
        Q10 = Tensor(-99.)
        RMR = Tensor(-99.)
        RML = Tensor(-99.)
        RMS = Tensor(-99.)
        RMO = Tensor(-99.)
        RFSETB = TensorBatchAfgenTrait()

    class RateVariables(RatesTemplate):
        PMRES = Tensor(-99.)

    def __init__(self, day:date, kiosk, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["PMRES"])
        
    def __call__(self, day:date, drv):
        """Calculate the maintenence respiration of the crop
        """
        p = self.params
        k = self.kiosk
        
        RMRES = (p.RMR * k.WRT +
                 p.RML * k.WLV +
                 p.RMS * k.WST +
                 p.RMO * k.WSO)
        
        RMRES = RMRES * p.RFSETB(k.DVS)
        TEFF = p.Q10 ** ((drv.TEMP - 25.) / 10.)
        self.rates.PMRES = RMRES * TEFF

        self.rates._update_kiosk()
        
        return self.rates.PMRES
    
    def reset(self, day:date):
        """Reset states and rates
        """
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["PMRES"])

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.rates.PMRES
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.rates,v)
            return output_vars
        
    def get_extra_states(self):
        """
        Get extra states
        """
        return {}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)