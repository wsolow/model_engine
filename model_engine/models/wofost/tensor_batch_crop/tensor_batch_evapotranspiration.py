"""Simulation object for computing evaporation and transpiration based on CO2 effects

Written by Will Solow, 2025
"""
from datetime import date
import torch

from traitlets_pcse import Bool
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
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

    sweaf = torch.where(DEPNR < 3., sweaf + (ET0 - 0.6) / (DEPNR * (DEPNR + 3.)), sweaf)

    return torch.clamp(torch.tensor([0.10]).to(sweaf.device), torch.tensor([0.95]).to(sweaf.device), sweaf)

class EvapotranspirationCO2_TensorBatch(BatchTensorModel):
    """Calculation of evaporation (water and soil) and transpiration rates
    taking into account the CO2 effect on crop transpiration.
    """

    _IDWST = Tensor(0)
    _IDOST = Tensor(0)

    class Parameters(ParamTemplate):
        CFET    = Tensor(-99.)
        DEPNR   = Tensor(-99.)
        KDIFTB  = TensorBatchAfgenTrait()
        IAIRDU  = Tensor(-99.)
        IOX     = Tensor(-99.)
        CRAIRC  = Tensor(-99.)
        SM0     = Tensor(-99.)
        SMW     = Tensor(-99.)
        SMFCF   = Tensor(-99.)
        CO2     = Tensor(-99.)
        CO2TRATB = TensorBatchAfgenTrait()

    class RateVariables(RatesTemplate):
        EVWMX = Tensor(-99.)
        EVSMX = Tensor(-99.)
        TRAMX = Tensor(-99.)
        TRA   = Tensor(-99.)
        IDOS  = Tensor(0.0)
        IDWS  = Tensor(0.0)
        RFWS = Tensor(-99.)
        RFOS = Tensor(-99.)
        RFTRA = Tensor(-99.)

    def __init__(self, day:date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, 
                    publish=["TRA", "EVWMX", "EVSMX", "RFTRA", "RFOS"])
        
        self.zero_tensor = torch.tensor([0.]).to(self.device)
        self.one_tensor = torch.tensor([1.0]).to(self.device)

    def __call__(self, day:date, drv, _emerging):
        """Calls the Evapotranspiration object to compute value to be returned to 
        model
        """
        p = self.params
        r = self.rates
        k = self.kiosk

        RF_TRAMX_CO2 = p.CO2TRATB(p.CO2)

        ET0_CROP = torch.max(self.zero_tensor, p.CFET * drv.ET0)

        KGLOB = 0.75 * p.KDIFTB(k.DVS)
        EKL = torch.exp(-KGLOB * k.LAI)
        r.EVWMX = drv.E0 * EKL
        r.EVSMX = torch.max(self.zero_tensor, drv.ES0 * EKL)
        r.TRAMX = ET0_CROP * (1.-EKL) * RF_TRAMX_CO2

        SWDEP = SWEAF(ET0_CROP, p.DEPNR)

        SMCR = (1. - SWDEP) * (p.SMFCF - p.SMW) + p.SMW

        r.RFWS = torch.clamp((k.SM - p.SMW) / (SMCR - p.SMW), self.zero_tensor, self.one_tensor)

        RFOSMX = torch.clamp((p.SM0 - k.SM)/p.CRAIRC, self.zero_tensor, self.one_tensor)
        r.RFOS = torch.where((p.IAIRDU == 0) & (p.IOX == 1), RFOSMX + (1. - torch.min(k.DSOS, torch.tensor([4]).to(self.device)) / 4.) * (1. - RFOSMX), \
                             torch.ones((self.num_models,)).to(self.device) )
        
        r.RFTRA = r.RFOS * r.RFWS
        r.TRA = r.TRAMX * r.RFTRA

        r.IDWS = torch.where(r.RFWS < 1., 1.0, r.IDWS)
        self._IDWST = torch.where(r.RFWS < 1., self._IDWST + 1, self._IDWST)

        r.IDOS = torch.where(r.RFOS < 1., 1.0, r.IDOS)
        self._IDOST = torch.where(r.RFOS < 1., self._IDOST + 1, self._IDOST)

        # Only store values when emerging is false
        r.EVWMX = torch.where(_emerging, 0.0, r.EVWMX)
        r.EVSMX = torch.where(_emerging, 0.0, r.EVSMX)
        r.TRAMX = torch.where(_emerging, 0.0, r.TRAMX)
        r.TRA   = torch.where(_emerging, 0.0, r.TRA)
        r.IDOS  = torch.where(_emerging, 0.0, r.IDOS)
        r.IDWS  = torch.where(_emerging, 0.0, r.IDWS)
        r.RFWS = torch.where(_emerging, 0.0, r.RFWS)
        r.RFOS = torch.where(_emerging, 0.0, r.RFOS)
        r.RFTRA = torch.where(_emerging, 0.0, r.RFTRA)

        self._IDWST = torch.where(_emerging, 0.0, self._IDWST)
        self._IDOST = torch.where(_emerging, 0.0, self._IDOST)

        self.rates._update_kiosk()

        return r.TRA, r.TRAMX

    def reset(self, day:date):
        """Reset states and rates
        """

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, 
                    publish=["TRA", "EVWMX", "EVSMX", "RFTRA", "RFOS"])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.rates.TRA
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
        return {"_IDWST": self._IDWST,
                "_IDOST": self._IDOST}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)