"""Handles storage organ dynamics for crop. Modified from original WOFOST
to include the death of storage organs

Written by Will Solow, 2025
"""

from datetime import date
import torch

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WOFOST_Storage_Organ_Dynamics(TensorModel):

    """Implementation of storage organ dynamics.
    """

    class Parameters(ParamTemplate):      
        SPA    = TensorAfgenTrait()
        TDWI   = Tensor(-99.)
        RDRSOB = TensorAfgenTrait()
        RDRSOF = TensorAfgenTrait()

    class StateVariables(StatesTemplate):
        WSO  = Tensor(-99.) 
        DWSO = Tensor(-99.) 
        TWSO = Tensor(-99.) 
        PAI  = Tensor(-99.) 

    class RateVariables(RatesTemplate):
        GRSO = Tensor(-99.)
        DRSO = Tensor(-99.)
        GWSO = Tensor(-99.)
        
    def __init__(self, day: date, kiosk, parvalues: dict, device):

        super().__init__(day, kiosk, parvalues, device)
        
        p = self.params
        
        FO = self.kiosk.FO
        FR = self.kiosk.FR
        WSO  = (p.TDWI * (1 - FR)) * FO
        DWSO = 0.
        TWSO = WSO + DWSO
        
        PAI = WSO * p.SPA(self.kiosk.DVS)

        self.states = self.StateVariables(kiosk=self.kiosk, publish=[],
                                          WSO=WSO, DWSO=DWSO, TWSO=TWSO,
                                          PAI=PAI)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=[])

        self.zero_tensor = torch.Tensor([0.]).to(self.device)
        self.one_tensor = torch.Tensor([1.]).to(self.device)

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

        self.states = self.StateVariables(kiosk=self.kiosk, publish=[],
                                          WSO=WSO, DWSO=DWSO, TWSO=TWSO,
                                          PAI=PAI)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=[])
