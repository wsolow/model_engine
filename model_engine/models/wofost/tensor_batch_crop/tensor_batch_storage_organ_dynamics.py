"""Handles storage organ dynamics for crop. Modified from original WOFOST
to include the death of storage organs

Written by Will Solow, 2025
"""

from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WOFOST_Storage_Organ_Dynamics_TensorBatch(BatchTensorModel):

    """Implementation of storage organ dynamics.
    """

    class Parameters(ParamTemplate):      
        SPA    = TensorBatchAfgenTrait()
        TDWI   = Tensor(-99.)
        RDRSOB = TensorBatchAfgenTrait()
        RDRSOF = TensorBatchAfgenTrait()

    class StateVariables(StatesTemplate):
        WSO  = Tensor(-99.) 
        DWSO = Tensor(-99.) 
        TWSO = Tensor(-99.) 
        PAI  = Tensor(-99.) 

    class RateVariables(RatesTemplate):
        GRSO = Tensor(-99.)
        DRSO = Tensor(-99.)
        GWSO = Tensor(-99.)
        
    def __init__(self, day: date, kiosk, parvalues: dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)
        
        p = self.params
        
        FO = self.kiosk.FO
        FR = self.kiosk.FR
        WSO  = (p.TDWI * (1 - FR)) * FO
        DWSO = 0.
        TWSO = WSO + DWSO
        
        PAI = WSO * p.SPA(self.kiosk.DVS)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["TWSO", "WSO"],
                                          WSO=WSO, DWSO=DWSO, TWSO=TWSO,
                                          PAI=PAI)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=[])

        self.zero_tensor = torch.tensor([0.]).to(self.device)
        self.one_tensor = torch.tensor([1.]).to(self.device)

    def calc_rates(self, day:date, drv):
        """Compute rates for integration
        """
        r  = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        r.GRSO = k.ADMI * k.FO
        r.DRSO = s.WSO * torch.clamp(p.RDRSOB(k.DVS) + p.RDRSOF(drv.TEMP), self.zero_tensor, self.one_tensor)
        r.GWSO = r.GRSO - r.DRSO

        self.rates._update_kiosk()

    def integrate(self, day:date, delt:float=1.0):
        """Integrate rates
        """
        p = self.params
        r = self.rates
        s = self.states

        s.WSO = s.WSO + r.GWSO
        s.DWSO = s.DWSO + r.DRSO
        s.TWSO = s.WSO + s.DWSO
        s.PAI = s.WSO * p.SPA(self.kiosk.DVS)

        self.states._update_kiosk()

    def reset(self, day:date):
        """Reset states and rates
        """
        p = self.params
        FO = self.kiosk.FO
        FR = self.kiosk.FR
        WSO  = (p.TDWI * (1 - FR)) * FO
        DWSO = 0.
        TWSO = WSO + DWSO
        
        PAI = WSO * p.SPA(self.kiosk.DVS)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["TWSO", "WSO"],
                                          WSO=WSO, DWSO=DWSO, TWSO=TWSO,
                                          PAI=PAI)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=[])

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.WSO
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
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