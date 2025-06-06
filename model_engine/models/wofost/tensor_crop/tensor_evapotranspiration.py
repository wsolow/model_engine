"""Simulation object for computing evaporation and transpiration based on CO2 effects

Written by Will Solow, 2025
"""
from datetime import date
import torch

from traitlets_pcse import Bool
from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate


def SWEAF(ET0, DEPNR):
    """Calculates the Soil Water Easily Available Fraction (SWEAF).

    :param ET0: The evapotranpiration from a reference crop.
    :param DEPNR: The crop dependency number.
    
    The fraction of easily available soil water between field capacity and
    wilting point is a function of the potential evapotranspiration rate
    (for a closed canopy) in cm/day, ET0, and the crop group number, DEPNR
    (from 1 (=drought-sensitive) to 5 (=drought-resistent)). The function
    SWEAF describes this relationship given in tabular form by Doorenbos &
    Kassam (1979) and by Van Keulen & Wolf (1986; p.108, table 20)
    http://edepot.wur.nl/168025.

    Modified by Will Solow, 2025
    To support Pytorch Tensors
    """
    A = 0.76
    B = 1.5
    
    sweaf = 1. / (A + B * ET0) - (5. - DEPNR) * 0.10

    
    if (DEPNR < 3.):
        sweaf = sweaf + (ET0 - 0.6) / (DEPNR * (DEPNR + 3.))

    return torch.clamp(torch.tensor([0.10]).to(sweaf.device), torch.tensor([0.95]).to(sweaf.device), sweaf)

class EvapotranspirationCO2_Tensor(TensorModel):
    """Calculation of evaporation (water and soil) and transpiration rates
    taking into account the CO2 effect on crop transpiration.
    """

    _IDWST = Tensor(0)
    _IDOST = Tensor(0)

    class Parameters(ParamTemplate):
        CFET    = Tensor(-99.)
        DEPNR   = Tensor(-99.)
        KDIFTB  = TensorAfgenTrait()
        IAIRDU  = Tensor(-99.)
        IOX     = Tensor(-99.)
        CRAIRC  = Tensor(-99.)
        SM0     = Tensor(-99.)
        SMW     = Tensor(-99.)
        SMFCF   = Tensor(-99.)
        CO2     = Tensor(-99.)
        CO2TRATB = TensorAfgenTrait()

    class RateVariables(RatesTemplate):
        EVWMX = Tensor(-99.)
        EVSMX = Tensor(-99.)
        TRAMX = Tensor(-99.)
        TRA   = Tensor(-99.)
        IDOS  = Bool(False)
        IDWS  = Bool(False)
        RFWS = Tensor(-99.)
        RFOS = Tensor(-99.)
        RFTRA = Tensor(-99.)

    def __init__(self, day:date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        self.rates = self.RateVariables(kiosk=self.kiosk, 
                    publish=["TRA", "EVWMX", "EVSMX", "RFTRA", "RFOS"])
        
        self.zero_tensor = torch.tensor([0.]).to(self.device)
        self.one_tensor = torch.tensor([1.0]).to(self.device)

    def __call__(self, day:date, drv):
        """Calls the Evapotranspiration object to compute value to be returned to 
        model
        """
        p = self.params
        r = self.rates
        k = self.kiosk
        LAI = k.LAI
        SM = k.SM
        DSOS = k.DSOS

        RF_TRAMX_CO2 = p.CO2TRATB(p.CO2)

        ET0_CROP = torch.max(self.zero_tensor, p.CFET * drv.ET0)

        KGLOB = 0.75 * p.KDIFTB(k.DVS)
        EKL = torch.exp(-KGLOB * LAI)
        r.EVWMX = drv.E0 * EKL
        r.EVSMX = torch.max(self.zero_tensor, drv.ES0 * EKL)
        r.TRAMX = ET0_CROP * (1.-EKL) * RF_TRAMX_CO2

        SWDEP = SWEAF(ET0_CROP, p.DEPNR)

        SMCR = (1. - SWDEP) * (p.SMFCF - p.SMW) + p.SMW

        r.RFWS = torch.clamp((SM - p.SMW) / (SMCR - p.SMW), self.zero_tensor, self.one_tensor)

        r.RFOS = 1.
        if p.IAIRDU == 0 and p.IOX == 1:
            RFOSMX = torch.clamp((p.SM0 - SM)/p.CRAIRC, self.zero_tensor, self.one_tensor)
            
            r.RFOS = RFOSMX + (1. - torch.min(DSOS, torch.tensor([4]).to(self.device)) / 4.) * (1. - RFOSMX)

        r.RFTRA = r.RFOS * r.RFWS
        r.TRA = r.TRAMX * r.RFTRA

        if r.RFWS < 1.:
            r.IDWS = True
            self._IDWST = self._IDWST + 1
        if r.RFOS < 1.:
            r.IDOS = True
            self._IDOST = self._IDOST + 1

        self.rates._update_kiosk()

        return r.TRA, r.TRAMX

    def reset(self, day:date):
        """Reset states and rates
        """

        self.rates = self.RateVariables(kiosk=self.kiosk, 
                    publish=["TRA", "EVWMX", "EVSMX", "RFTRA", "RFOS"])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.rates.TRA
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
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
        return {"_IDWST": self._IDWST,
                "_IDOST": self._IDOST}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)