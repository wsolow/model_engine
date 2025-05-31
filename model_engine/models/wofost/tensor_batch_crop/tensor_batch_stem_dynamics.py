"""Handles stem biomass dynamics for crop model

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date 
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate


class WOFOST_Stem_Dynamics_TensorBatch(BatchTensorModel):
    """Implementation of stem biomass dynamics.
    
    """

    class Parameters(ParamTemplate):      
        RDRSTB = TensorBatchAfgenTrait()
        SSATB  = TensorBatchAfgenTrait()
        TDWI   = Tensor(-99.)

    class StateVariables(StatesTemplate):
        WST  = Tensor(-99.)
        DWST = Tensor(-99.)
        TWST = Tensor(-99.)
        SAI  = Tensor(-99.) 

    class RateVariables(RatesTemplate):
        GRST = Tensor(-99.)
        DRST = Tensor(-99.)
        GWST = Tensor(-99.)
        
    def __init__(self, day:date, kiosk, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)
        
        FS   = self.kiosk.FS
        FR   = self.kiosk.FR
        WST  = (self.params.TDWI * (1-FR)) * FS
        DWST = 0.
        TWST = WST + DWST
        
        DVS = self.kiosk.DVS
        SAI = WST * self.params.SSATB(DVS)

        self.states = self.StateVariables(num_models=self.num_models,kiosk=self.kiosk, publish=["WST", "TWST"],
                                          WST=WST, DWST=DWST, TWST=TWST, SAI=SAI)
        self.rates  = self.RateVariables(num_models=self.num_models,kiosk=self.kiosk, publish=["GRST", "DRST"])
    
    def calc_rates(self, day:date, drv, _emerging):
        """Compute state rates before integration
        """
        r  = self.rates
        s = self.states
        p = self.params
        
        r.GRST = self.kiosk.ADMI * self.kiosk.FS
        r.DRST = p.RDRSTB(self.kiosk.DVS) * s.WST
        r.GWST = r.GRST - r.DRST

        r.GRST = torch.where(_emerging, 0.0, r.GRST)
        r.DRST = torch.where(_emerging, 0.0, r.DRST)
        r.GWST = torch.where(_emerging, 0.0, r.GWST)

        self.rates._update_kiosk()

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        p = self.params
        r = self.rates
        s = self.states

        s.WST = s.WST + r.GWST
        s.DWST = s.DWST + r.DRST
        s.TWST = s.WST + s.DWST

        s.SAI = s.WST * p.SSATB(self.kiosk.DVS)

        self.states._update_kiosk()
        
    def reset(self, day:date):
        """Reset states and rates
        """
        
        FS   = self.kiosk.FS
        FR   = self.kiosk.FR
        WST  = (self.params.TDWI * (1-FR)) * FS
        DWST = 0.
        TWST = WST + DWST
        
        DVS = self.kiosk.DVS
        SAI = WST * self.params.SSATB(DVS)

        self.states = self.StateVariables(num_models=self.num_models,kiosk=self.kiosk, publish=["WST", "TWST"],
                                          WST=WST, DWST=DWST, TWST=TWST, SAI=SAI)
        self.rates  = self.RateVariables(num_models=self.num_models,kiosk=self.kiosk, publish=["GRST", "DRST"])

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.WST
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