"""
Performs bookkeeping for how NPK is translocated around roots, leaves, and stems

Written by: Allard de Wit and Iwan Supi (allard.dewit@wur.nl), July 2015
Approach based on: LINTUL N/P/K made by Joost Wolf
Modified by Will Solow, 2024
"""

from datetime import date

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class NPK_Translocation(TensorModel):
    """Does the bookkeeping for translocation of N/P/K from the roots, leaves
    and stems towards the storage organs of the crop.

    """

    class Parameters(ParamTemplate):
        NRESIDLV = Tensor(-99.)  
        NRESIDST = Tensor(-99.)  
        NRESIDRT = Tensor(-99.)  

        PRESIDLV = Tensor(-99.)  
        PRESIDST = Tensor(-99.)  
        PRESIDRT = Tensor(-99.)  

        KRESIDLV = Tensor(-99.)  
        KRESIDST = Tensor(-99.)  
        KRESIDRT = Tensor(-99.)  

        NPK_TRANSLRT_FR = Tensor(-99.)                      
        DVS_NPK_TRANSL = Tensor(-99.) 
        

    class RateVariables(RatesTemplate):
        RNTRANSLOCATIONLV = Tensor(-99.)  
        RNTRANSLOCATIONST = Tensor(-99.)  
        RNTRANSLOCATIONRT = Tensor(-99.)  

        RPTRANSLOCATIONLV = Tensor(-99.)  
        RPTRANSLOCATIONST = Tensor(-99.)  
        RPTRANSLOCATIONRT = Tensor(-99.)  

        RKTRANSLOCATIONLV = Tensor(-99.)  
        RKTRANSLOCATIONST = Tensor(-99.)  
        RKTRANSLOCATIONRT = Tensor(-99.)  

    class StateVariables(StatesTemplate):
        NTRANSLOCATABLELV = Tensor(-99.)  
        NTRANSLOCATABLEST = Tensor(-99.)  
        NTRANSLOCATABLERT = Tensor(-99.)  
        
        PTRANSLOCATABLELV = Tensor(-99.)  
        PTRANSLOCATABLEST = Tensor(-99.)  
        PTRANSLOCATABLERT = Tensor(-99.)  
        
        KTRANSLOCATABLELV = Tensor(-99.)  
        KTRANSLOCATABLEST = Tensor(-99.)  
        KTRANSLOCATABLERT = Tensor(-99.)  

        NTRANSLOCATABLE = Tensor(-99.)  
        PTRANSLOCATABLE = Tensor(-99.)  
        KTRANSLOCATABLE = Tensor(-99.)  

    def __init__(self, day:date, kiosk, parvalues:dict):

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["RNTRANSLOCATIONLV", "RNTRANSLOCATIONST", "RNTRANSLOCATIONRT",
                                                        "RPTRANSLOCATIONLV", "RPTRANSLOCATIONST", "RPTRANSLOCATIONRT",
                                                        "RKTRANSLOCATIONLV", "RKTRANSLOCATIONST", "RKTRANSLOCATIONRT"])

        self.states = self.StateVariables(kiosk,
            NTRANSLOCATABLELV=0., NTRANSLOCATABLEST=0., NTRANSLOCATABLERT=0., PTRANSLOCATABLELV=0., PTRANSLOCATABLEST=0.,
            PTRANSLOCATABLERT=0., KTRANSLOCATABLELV=0., KTRANSLOCATABLEST=0. ,KTRANSLOCATABLERT=0.,
            NTRANSLOCATABLE=0., PTRANSLOCATABLE=0., KTRANSLOCATABLE=0.,
            publish=["NTRANSLOCATABLE", "PTRANSLOCATABLE", "KTRANSLOCATABLE", "NTRANSLOCATABLELV", 
                     "NTRANSLOCATABLEST", "NTRANSLOCATABLERT", "PTRANSLOCATABLELV", 
                     "PTRANSLOCATABLEST", "PTRANSLOCATABLERT", "KTRANSLOCATABLELV", 
                     "KTRANSLOCATABLEST", "KTRANSLOCATABLERT",])
        self.kiosk = kiosk
        
    
    def calc_rates(self, day:date, drv):
        """Calculate rates for integration
        """
        r = self.rates
        s = self.states
        k = self.kiosk

        if s.NTRANSLOCATABLE > 0.:
            r.RNTRANSLOCATIONLV = k.RNUPTAKESO * s.NTRANSLOCATABLELV / s.NTRANSLOCATABLE
            r.RNTRANSLOCATIONST = k.RNUPTAKESO * s.NTRANSLOCATABLEST / s.NTRANSLOCATABLE
            r.RNTRANSLOCATIONRT = k.RNUPTAKESO * s.NTRANSLOCATABLERT / s.NTRANSLOCATABLE
        else:
            r.RNTRANSLOCATIONLV = r.RNTRANSLOCATIONST = r.RNTRANSLOCATIONRT = 0.

        if s.PTRANSLOCATABLE > 0:
            r.RPTRANSLOCATIONLV = k.RPUPTAKESO * s.PTRANSLOCATABLELV / s.PTRANSLOCATABLE
            r.RPTRANSLOCATIONST = k.RPUPTAKESO * s.PTRANSLOCATABLEST / s.PTRANSLOCATABLE
            r.RPTRANSLOCATIONRT = k.RPUPTAKESO * s.PTRANSLOCATABLERT / s.PTRANSLOCATABLE
        else:
            r.RPTRANSLOCATIONLV = r.RPTRANSLOCATIONST = r.RPTRANSLOCATIONRT = 0.

        if s.KTRANSLOCATABLE > 0:
            r.RKTRANSLOCATIONLV = k.RKUPTAKESO * s.KTRANSLOCATABLELV / s.KTRANSLOCATABLE
            r.RKTRANSLOCATIONST = k.RKUPTAKESO * s.KTRANSLOCATABLEST / s.KTRANSLOCATABLE
            r.RKTRANSLOCATIONRT = k.RKUPTAKESO * s.KTRANSLOCATABLERT / s.KTRANSLOCATABLE
        else:
            r.RKTRANSLOCATIONLV = r.RKTRANSLOCATIONST = r.RKTRANSLOCATIONRT = 0.

    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        p = self.params
        s = self.states
        k = self.kiosk
        
        
        s.NTRANSLOCATABLELV = max(0., k.NAMOUNTLV - k.WLV * p.NRESIDLV)
        s.NTRANSLOCATABLEST = max(0., k.NAMOUNTST - k.WST * p.NRESIDST)
        s.NTRANSLOCATABLERT = max(0., k.NAMOUNTRT - k.WRT * p.NRESIDRT)

        
        s.PTRANSLOCATABLELV = max(0., k.PAMOUNTLV - k.WLV * p.PRESIDLV)
        s.PTRANSLOCATABLEST = max(0., k.PAMOUNTST - k.WST * p.PRESIDST)
        s.PTRANSLOCATABLERT = max(0., k.PAMOUNTRT - k.WRT * p.PRESIDRT)

        
        s.KTRANSLOCATABLELV = max(0., k.KAMOUNTLV - k.WLV * p.KRESIDLV)
        s.KTRANSLOCATABLEST = max(0., k.KAMOUNTST - k.WST * p.KRESIDST)
        s.KTRANSLOCATABLERT = max(0., k.KAMOUNTRT - k.WRT * p.KRESIDRT)

        
        if k.DVS > p.DVS_NPK_TRANSL:
            s.NTRANSLOCATABLE = s.NTRANSLOCATABLELV + s.NTRANSLOCATABLEST + s.NTRANSLOCATABLERT
            s.PTRANSLOCATABLE = s.PTRANSLOCATABLELV + s.PTRANSLOCATABLEST + s.PTRANSLOCATABLERT
            s.KTRANSLOCATABLE = s.KTRANSLOCATABLELV + s.KTRANSLOCATABLEST + s.KTRANSLOCATABLERT
        else:
            s.NTRANSLOCATABLE = s.PTRANSLOCATABLE = s.KTRANSLOCATABLE = 0

    def reset(self):
        """Reset states and rates
        """ 
        s = self.states
        r = self.rates

        r.RNTRANSLOCATIONLV = r.RNTRANSLOCATIONST = r.RNTRANSLOCATIONRT = r.RPTRANSLOCATIONLV \
            = r.RPTRANSLOCATIONST = r.RPTRANSLOCATIONRT = r.RKTRANSLOCATIONLV \
            = r.RKTRANSLOCATIONST = r.RKTRANSLOCATIONRT = 0

        s.NTRANSLOCATABLELV = s.NTRANSLOCATABLEST = s.NTRANSLOCATABLERT = s.PTRANSLOCATABLELV \
            = s.PTRANSLOCATABLEST = s.PTRANSLOCATABLERT = s.KTRANSLOCATABLELV \
            = s.KTRANSLOCATABLEST = s.KTRANSLOCATABLERT = s.NTRANSLOCATABLE \
            = s.PTRANSLOCATABLE = s.KTRANSLOCATABLE = 0