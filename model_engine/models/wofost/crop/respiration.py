"""Handles Respiration of crop 

Written by: Will Solow, 2025
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

    def __init__(self, day:date, kiosk, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        self.rates = self.RateVariables(kiosk=self.kiosk, publish=["PMRES"])
        
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

        return self.rates.PMRES
    
    def reset(self):
        """Reset states and rates
        """
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=["PMRES"])
