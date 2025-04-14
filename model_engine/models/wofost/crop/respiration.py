"""Handles Respiration of crop 

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
from datetime import date

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WOFOST_Maintenance_Respiration(TensorModel):
    """Maintenance respiration in WOFOST

    """
    
    class Parameters(ParamTemplate):
        Q10 = Tensor(-99.)
        RMR = Tensor(-99.)
        RML = Tensor(-99.)
        RMS = Tensor(-99.)
        RMO = Tensor(-99.)
        RFSETB = TensorAfgenTrait()

    class RateVariables(RatesTemplate):
        PMRES = Tensor(-99.)

    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish="PMRES")
        self.kiosk = kiosk
        
    def __call__(self, day:date, drv):
        """Calculate the maintenence respiration of the crop
        """
        p = self.params
        kk = self.kiosk
        
        RMRES = (p.RMR * kk["WRT"] +
                 p.RML * kk["WLV"] +
                 p.RMS * kk["WST"] +
                 p.RMO * kk["WSO"])
        RMRES *= p.RFSETB(kk["DVS"])
        TEFF = p.Q10**((drv.TEMP-25.)/10.)
        self.rates.PMRES = RMRES * TEFF
        return self.rates.PMRES
    
    def reset(self):
        """Reset states and rates
        """
        self.rates.PMRES = 0
