"""Handles stem biomass dynamics for crop model

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date 
import torch

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate


class WOFOST_Stem_Dynamics(TensorModel):
    """Implementation of stem biomass dynamics.
    
    """

    class Parameters(ParamTemplate):      
        RDRSTB = TensorAfgenTrait()
        SSATB  = TensorAfgenTrait()
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
        
    def __init__(self, day:date, kiosk, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)
        
        FS   = self.kiosk.FS
        FR   = self.kiosk.FR
        WST  = (self.params.TDWI * (1-FR)) * FS
        DWST = 0.
        TWST = WST + DWST
        
        DVS = self.kiosk.DVS
        SAI = WST * self.params.SSATB(DVS)

        self.states = self.StateVariables(kiosk=self.kiosk, publish=["WST", "TWST"],
                                          WST=WST, DWST=DWST, TWST=TWST, SAI=SAI)
        self.rates  = self.RateVariables(kiosk=self.kiosk, publish=["GRST", "DRST"])
    
    def calc_rates(self, day:date, drv):
        """Compute state rates before integration
        """
        r  = self.rates
        s = self.states
        p = self.params
        
        r.GRST = self.kiosk.ADMI * self.kiosk.FS
        r.DRST = p.RDRSTB(self.kiosk.DVS) * s.WST
        r.GWST = r.GRST - r.DRST

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

        self.states = self.StateVariables(kiosk=self.kiosk, publish=["WST", "TWST"],
                                          WST=WST, DWST=DWST, TWST=TWST, SAI=SAI)
        self.rates  = self.RateVariables(kiosk=self.kiosk, publish=["GRST", "DRST"])

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.WST
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.rates,v)
            return output_vars