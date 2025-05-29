"""Class for computing root biomass dynamics and rooting depth

Written by: Will Solow, 2024
"""

from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
 
class WOFOST_Root_Dynamics_TensorBatch(BatchTensorModel):
    """Root biomass dynamics and rooting depth.
    """

    class Parameters(ParamTemplate):
        RDI    = Tensor(-99.)
        RRI    = Tensor(-99.)
        RDMCR  = Tensor(-99.)
        RDMSOL = Tensor(-99.)
        TDWI   = Tensor(-99.)
        IAIRDU = Tensor(-99)
        RDRRTB = TensorBatchAfgenTrait()
        RDRROS = TensorBatchAfgenTrait()
        NTHRESH = Tensor(-99.) 
        PTHRESH = Tensor(-99.) 
        KTHRESH = Tensor(-99.) 
        RDRRNPK = TensorBatchAfgenTrait()
                    
    class RateVariables(RatesTemplate):
        RR   = Tensor(-99.)
        GRRT = Tensor(-99.)
        DRRT1 = Tensor(-99.) 
        DRRT2 = Tensor(-99.) 
        DRRT3 = Tensor(-99.) 
        DRRT = Tensor(-99.)
        GWRT = Tensor(-99.)

    class StateVariables(StatesTemplate):
        RD   = Tensor(-99.)
        RDM  = Tensor(-99.)
        WRT  = Tensor(-99.)
        DWRT = Tensor(-99.)
        TWRT = Tensor(-99.)
        
    def __init__(self, day:date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        p = self.params
        
        rdmax = torch.max(p.RDI, torch.min(p.RDMCR, p.RDMSOL))
        RDM = rdmax
        RD = p.RDI
        
        WRT  = p.TDWI * self.kiosk.FR
        DWRT = 0.
        TWRT = WRT + DWRT

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["WRT", "TWRT", "RD"],
                                          RD=RD, RDM=RDM, WRT=WRT, DWRT=DWRT,
                                          TWRT=TWRT)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["GRRT", "DRRT"])

    def calc_rates(self, day:date, drv):
        """Calculate state rates for integration
        """
        p = self.params
        r = self.rates
        s = self.states
        k = self.kiosk

        r.GRRT = k.FR * k.DMI

        RDRNPK = torch.max(torch.max(k.SURFACE_N / p.NTHRESH, k.SURFACE_P / p.PTHRESH), k.SURFACE_K / p.KTHRESH)
        r.DRRT1 = p.RDRRTB(k.DVS)
        r.DRRT2 = p.RDRROS(k.RFOS)
        r.DRRT3 = p.RDRRNPK(RDRNPK)

        r.DRRT = s.WRT * torch.clamp(torch.max(r.DRRT1, r.DRRT2+r.DRRT3), \
                                     torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device) )
        r.GWRT = r.GRRT - r.DRRT
        
        r.RR = torch.min((s.RDM - s.RD), p.RRI)
        
        if k.FR == 0.:
            r.RR = 0.

        self.rates._update_kiosk()
    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate rates for new states
        """
        r = self.rates
        s = self.states

        s.WRT = s.WRT + r.GWRT
        s.DWRT = s.DWRT + r.DRRT
        s.TWRT = s.WRT + s.DWRT
        s.RD = s.RD + r.RR

        self.states._update_kiosk()

    def reset(self, day:date):
        """Reset all states and rates to initial values
        """

        p = self.params
        
        rdmax = torch.max(p.RDI, torch.min(p.RDMCR, p.RDMSOL))
        RDM = rdmax
        RD = p.RDI
        
        WRT  = p.TDWI * self.kiosk.FR
        DWRT = 0.
        TWRT = WRT + DWRT

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["WRT", "TWRT", "RD"],
                                          RD=RD, RDM=RDM, WRT=WRT, DWRT=DWRT,
                                          TWRT=TWRT)
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["GRRT", "DRRT"])

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.WRT
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
        return {}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)