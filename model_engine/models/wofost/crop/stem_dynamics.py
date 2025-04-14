"""Handles stem biomass dynamics for crop model

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date 

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
        
    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        
        self.params = self.Parameters(parvalues)
        self.kiosk  = kiosk

        
        params = self.params
        
        FS = self.kiosk["FS"]
        FR = self.kiosk["FR"]
        WST  = (params.TDWI * (1-FR)) * FS
        DWST = 0.
        TWST = WST + DWST
        
        DVS = self.kiosk["DVS"]
        SAI = WST * params.SSATB(DVS)

        self.states = self.StateVariables(kiosk, publish=["WST", "DWST", "TWST", "SAI"],
                                          WST=WST, DWST=DWST, TWST=TWST, SAI=SAI)
        self.rates  = self.RateVariables(kiosk, publish=["GRST", "DRST", "GWST"])
    
    def calc_rates(self, day:date, drv):
        """Compute state rates before integration
        """
        rates  = self.rates
        states = self.states
        params = self.params
        
        DVS = self.kiosk["DVS"]
        FS = self.kiosk["FS"]
        ADMI = self.kiosk["ADMI"]

        
        rates.GRST = ADMI * FS
        rates.DRST = params.RDRSTB(DVS) * states.WST
        rates.GWST = rates.GRST - rates.DRST

    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        params = self.params
        rates = self.rates
        states = self.states

        
        states.WST += rates.GWST
        states.DWST += rates.DRST
        states.TWST = states.WST + states.DWST

        
        DVS = self.kiosk["DVS"]
        states.SAI = states.WST * params.SSATB(DVS)

    def publish_states(self):
        params = self.params
        rates = self.rates
        states = self.states

        
        states.WST += rates.GWST
        states.DWST += rates.DRST
        states.TWST = states.WST + states.DWST

        
        DVS = self.kiosk["DVS"]
        states.SAI = states.WST * params.SSATB(DVS)

    def reset(self):
        """Reset states and rates
        """
        
        params = self.params
        s = self.states
        r = self.rates
        
        FS = self.kiosk["FS"]
        FR = self.kiosk["FR"]
        WST  = (params.TDWI * (1-FR)) * FS
        DWST = 0.
        TWST = WST + DWST
        
        DVS = self.kiosk["DVS"]
        SAI = WST * params.SSATB(DVS)

        s.WST=WST
        s.DWST=DWST
        s.TWST=TWST
        s.SAI=SAI

        r.GRST = r.DRST = r.GWST = 0
