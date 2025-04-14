"""Handles storage organ dynamics for crop. Modified from original WOFOST
to include the death of storage organs

Written by Will Solow, 2025
"""

from datetime import date

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
        HWSO = Tensor(-99.) 
        PAI  = Tensor(-99.) 
        LHW  = Tensor(-99.) 

    class RateVariables(RatesTemplate):
        GRSO = Tensor(-99.)
        DRSO = Tensor(-99.)
        GWSO = Tensor(-99.)
        DHSO = Tensor(-99.)
        
    def __init__(self, day: date, kiosk, parvalues: dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.params = self.Parameters(parvalues)
        self.kiosk = kiosk
        
        params = self.params
        
        FO = self.kiosk["FO"]
        FR = self.kiosk["FR"]
        WSO  = (params.TDWI * (1-FR)) * FO
        DWSO = 0.
        HWSO = 0.
        LHW = HWSO
        TWSO = WSO + DWSO
        
        PAI = WSO * params.SPA(self.kiosk.DVS)

        self.states = self.StateVariables(kiosk, publish=["WSO", "DWSO", "TWSO", 
                                                          "HWSO", "PAI", "LHW"],
                                          WSO=WSO, DWSO=DWSO, TWSO=TWSO, HWSO=HWSO,
                                          PAI=PAI, LHW=LHW)
        
        self.rates = self.RateVariables(kiosk, publish=[ "GRSO", "DRSO", "GWSO", "DHSO"])

    
    def calc_rates(self, day:date, drv):
        """Compute rates for integration
        """
        rates  = self.rates
        states = self.states
        params = self.params
        k = self.kiosk
        
        FO = self.kiosk["FO"]
        ADMI = self.kiosk["ADMI"]

        
        rates.GRSO = ADMI * FO

        rates.DRSO = states.WSO * limit(0, 1, params.RDRSOB(k.DVS)+params.RDRSOF(drv.TEMP))
        rates.DHSO = states.HWSO * limit(0, 1, params.RDRSOB(k.DVS)+params.RDRSOF(drv.TEMP))
        rates.GWSO = rates.GRSO - rates.DRSO

    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate rates
        """
        params = self.params
        rates = self.rates
        states = self.states

        
        states.WSO += rates.GWSO
        states.HWSO += rates.GRSO - rates.DHSO
        states.DWSO += rates.DRSO
        states.TWSO = states.WSO + states.DWSO
        
        states.HWSO = limit(0, states.WSO, states.HWSO)
        
        states.PAI = states.WSO * params.SPA(self.kiosk.DVS)

    def reset(self):
        """Reset states and rates
        """
        
        params = self.params
        s = self.states
        r = self.rates
        
        FO = self.kiosk["FO"]
        FR = self.kiosk["FR"]
        
        WSO  = (params.TDWI * (1-FR)) * FO
        DWSO = 0.
        HWSO = 0.
        LHW = HWSO
        TWSO = WSO + DWSO
        
        PAI = WSO * params.SPA(self.kiosk.DVS)

        s.WSO=WSO
        s.DWSO=DWSO
        s.TWSO=TWSO
        s.HWSO=HWSO
        s.PAI=PAI
        s.LHW=LHW

        r.GRSO = r.DRSO = r.GWSO = r.DHSO = 0
