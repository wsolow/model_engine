"""Class for computing root biomass dynamics and rooting depth

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
 
class WOFOST_Root_Dynamics(TensorModel):
    """Root biomass dynamics and rooting depth.
    """

    class Parameters(ParamTemplate):
        RDI    = Tensor(-99.)
        RRI    = Tensor(-99.)
        RDMCR  = Tensor(-99.)
        RDMSOL = Tensor(-99.)
        TDWI   = Tensor(-99.)
        IAIRDU = Tensor(-99)
        RDRRTB = TensorAfgenTrait()
        RDRROS = TensorAfgenTrait()
        NTHRESH = Tensor(-99.) 
        PTHRESH = Tensor(-99.) 
        KTHRESH = Tensor(-99.) 
        RDRRNPK = TensorAfgenTrait()
                    
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
        
    def __init__(self, day:date , kiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.kiosk = kiosk
        
        
        params = self.params
        
        rdmax = max(params.RDI, min(params.RDMCR, params.RDMSOL))
        RDM = rdmax
        RD = params.RDI
        
        WRT  = params.TDWI * self.kiosk.FR
        DWRT = 0.
        TWRT = WRT + DWRT

        self.states = self.StateVariables(kiosk, publish=["RD", "RDM", "WRT", 
                                                          "DWRT", "TWRT"],
                                          RD=RD, RDM=RDM, WRT=WRT, DWRT=DWRT,
                                          TWRT=TWRT)
        
        self.rates = self.RateVariables(kiosk, publish=["RR", "GRRT", "DRRT1",
                                                        "DRRT2", "DRRT3", "DRRT", "GWRT"])

    
    def calc_rates(self, day:date, drv):
        """Calculate state rates for integration
        """
        p = self.params
        r = self.rates
        s = self.states
        k = self.kiosk

        
        r.GRRT = k.FR * k.DMI

        
        RDRNPK = max(k.SURFACE_N / p.NTHRESH, k.SURFACE_P / p.PTHRESH, k.SURFACE_K / p.KTHRESH)
        r.DRRT1 = p.RDRRTB(k.DVS)
        r.DRRT2 = p.RDRROS(k.RFOS)
        r.DRRT3 = p.RDRRNPK(RDRNPK)

        
        r.DRRT = s.WRT * limit(0, 1, max(r.DRRT1, r.DRRT2+r.DRRT3))
        r.GWRT = r.GRRT - r.DRRT
        
        
        r.RR = min((s.RDM - s.RD), p.RRI)
        
        
        if k.FR == 0.:
            r.RR = 0.
    
    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate rates for new states
        """
        rates = self.rates
        states = self.states

        
        states.WRT += rates.GWRT
        
        states.DWRT += rates.DRRT
        
        states.TWRT = states.WRT + states.DWRT
        
        states.RD += rates.RR

    def publish_states(self):
        states = self.states

        
        states.WRT = states.WRT
        
        states.DWRT = states.DWRT
        
        states.TWRT = states.TWRT
        
        states.RD = states.RD

    def reset(self):
        """Reset all states and rates to initial values
        """
        
        params = self.params
        s = self.states
        r = self.rates
        
        rdmax = max(params.RDI, min(params.RDMCR, params.RDMSOL))
        RDM = rdmax
        RD = params.RDI
        
        WRT  = params.TDWI * self.kiosk.FR
        DWRT = 0.
        TWRT = WRT + DWRT
        
        s.RD = RD,
        s.RDM = RDM
        s.WRT = WRT
        s.DWRT = DWRT
        s.TWRT = TWRT

        r.RR = r.GRRT = r.DRRT = r.GWRT = 0
