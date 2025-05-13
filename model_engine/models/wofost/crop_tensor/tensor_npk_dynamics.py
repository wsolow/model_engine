"""Overall implementation for the NPK dynamics of the crop including
subclasses to 
    * NPK Demand Uptake
    * NPK Stress
    * NPK Translocation
    
Written by Will Solow, 2024
"""

from datetime import date
import torch

from traitlets_pcse import Instance

from model_engine.models.wofost.crop_tensor.nutrients.tensor_npk_demand_uptake import NPK_Demand_Uptake_Tensor as NPK_Demand_Uptake
from model_engine.models.wofost.crop_tensor.nutrients.tensor_npk_translocation import NPK_Translocation_Tensor as NPK_Translocation


from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class NPK_Crop_Dynamics_Tensor(TensorModel):
    """Implementation of overall NPK crop dynamics.
    """

    translocation = Instance(TensorModel)
    demand_uptake = Instance(TensorModel)

    NAMOUNTLVI = Tensor(-99.)  
    NAMOUNTSTI = Tensor(-99.)  
    NAMOUNTRTI = Tensor(-99.)  
    NAMOUNTSOI = Tensor(-99.)  
    
    PAMOUNTLVI = Tensor(-99.)  
    PAMOUNTSTI = Tensor(-99.)  
    PAMOUNTRTI = Tensor(-99.)  
    PAMOUNTSOI = Tensor(-99.)  

    KAMOUNTLVI = Tensor(-99.)  
    KAMOUNTSTI = Tensor(-99.)  
    KAMOUNTRTI = Tensor(-99.)  
    KAMOUNTSOI = Tensor(-99.)  

    class Parameters(ParamTemplate):
        NMAXLV_TB = TensorAfgenTrait()
        PMAXLV_TB = TensorAfgenTrait()
        KMAXLV_TB = TensorAfgenTrait()
        NMAXST_FR = Tensor(-99.)
        NMAXRT_FR = Tensor(-99.)
        PMAXST_FR = Tensor(-99.)
        PMAXRT_FR = Tensor(-99.)
        KMAXST_FR = Tensor(-99.)
        KMAXRT_FR = Tensor(-99.)
        NRESIDLV = Tensor(-99.)  
        NRESIDST = Tensor(-99.)  
        NRESIDRT = Tensor(-99.)  
        PRESIDLV = Tensor(-99.)  
        PRESIDST = Tensor(-99.)  
        PRESIDRT = Tensor(-99.)  
        KRESIDLV = Tensor(-99.)  
        KRESIDST = Tensor(-99.)  
        KRESIDRT = Tensor(-99.)  

    class StateVariables(StatesTemplate):
        NAMOUNTLV = Tensor(-99.) 
        PAMOUNTLV = Tensor(-99.) 
        KAMOUNTLV = Tensor(-99.) 
        
        NAMOUNTST = Tensor(-99.) 
        PAMOUNTST = Tensor(-99.) 
        KAMOUNTST = Tensor(-99.) 
      
        NAMOUNTSO = Tensor(-99.) 
        PAMOUNTSO = Tensor(-99.) 
        KAMOUNTSO = Tensor(-99.) 
        
        NAMOUNTRT = Tensor(-99.) 
        PAMOUNTRT = Tensor(-99.) 
        KAMOUNTRT = Tensor(-99.) 
        
        NUPTAKETOTAL = Tensor(-99.) 
        PUPTAKETOTAL = Tensor(-99.) 
        KUPTAKETOTAL = Tensor(-99.) 
        NFIXTOTAL = Tensor(-99.) 
        
        NlossesTotal = Tensor(-99.)
        PlossesTotal = Tensor(-99.)
        KlossesTotal = Tensor(-99.)

    class RateVariables(RatesTemplate):
        RNAMOUNTLV = Tensor(-99.)  
        RPAMOUNTLV = Tensor(-99.)
        RKAMOUNTLV = Tensor(-99.)
        
        RNAMOUNTST = Tensor(-99.)
        RPAMOUNTST = Tensor(-99.)
        RKAMOUNTST = Tensor(-99.)
               
        RNAMOUNTRT = Tensor(-99.)
        RPAMOUNTRT = Tensor(-99.)
        RKAMOUNTRT = Tensor(-99.)
        
        RNAMOUNTSO = Tensor(-99.)
        RPAMOUNTSO = Tensor(-99.)
        RKAMOUNTSO = Tensor(-99.)
               
        RNDEATHLV = Tensor(-99.)  
        RNDEATHST = Tensor(-99.)  
        RNDEATHRT = Tensor(-99.)  
        
        RPDEATHLV = Tensor(-99.)  
        RPDEATHST = Tensor(-99.)  
        RPDEATHRT = Tensor(-99.)  
        
        RKDEATHLV = Tensor(-99.)  
        RKDEATHST = Tensor(-99.)  
        RKDEATHRT = Tensor(-99.)  

        RNLOSS = Tensor(-99.)
        RPLOSS = Tensor(-99.)
        RKLOSS = Tensor(-99.)
        
    def __init__(self, day:date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)
        
        self.translocation = NPK_Translocation(day, self.kiosk, parvalues, self.device)
        self.demand_uptake = NPK_Demand_Uptake(day, self.kiosk, parvalues, self.device)

        p = self.params
        k = self.kiosk
        
        self.NAMOUNTLVI = NAMOUNTLV = k.WLV * p.NMAXLV_TB(k.DVS)
        self.NAMOUNTSTI = NAMOUNTST = k.WST * p.NMAXLV_TB(k.DVS) * p.NMAXST_FR
        self.NAMOUNTRTI = NAMOUNTRT = k.WRT * p.NMAXLV_TB(k.DVS) * p.NMAXRT_FR
        self.NAMOUNTSOI = NAMOUNTSO = 0.
        
        self.PAMOUNTLVI = PAMOUNTLV = k.WLV * p.PMAXLV_TB(k.DVS)
        self.PAMOUNTSTI = PAMOUNTST = k.WST * p.PMAXLV_TB(k.DVS) * p.PMAXST_FR
        self.PAMOUNTRTI = PAMOUNTRT = k.WRT * p.PMAXLV_TB(k.DVS) * p.PMAXRT_FR
        self.PAMOUNTSOI = PAMOUNTSO = 0.

        self.KAMOUNTLVI = KAMOUNTLV = k.WLV * p.KMAXLV_TB(k.DVS)
        self.KAMOUNTSTI = KAMOUNTST = k.WST * p.KMAXLV_TB(k.DVS) * p.KMAXST_FR
        self.KAMOUNTRTI = KAMOUNTRT = k.WRT * p.KMAXLV_TB(k.DVS) * p.KMAXRT_FR
        self.KAMOUNTSOI = KAMOUNTSO = 0.

        self.states = self.StateVariables(kiosk=self.kiosk,
            publish=["NAMOUNTLV", "NAMOUNTST", "NAMOUNTRT", "PAMOUNTLV", "PAMOUNTST", "PAMOUNTRT",
                     "KAMOUNTLV", "KAMOUNTST", "KAMOUNTRT", "NAMOUNTSO", "PAMOUNTSO", "KAMOUNTSO"],
                        NAMOUNTLV=NAMOUNTLV, NAMOUNTST=NAMOUNTST, NAMOUNTRT=NAMOUNTRT, NAMOUNTSO=NAMOUNTSO,
                        PAMOUNTLV=PAMOUNTLV, PAMOUNTST=PAMOUNTST, PAMOUNTRT=PAMOUNTRT, PAMOUNTSO=PAMOUNTSO,
                        KAMOUNTLV=KAMOUNTLV, KAMOUNTST=KAMOUNTST, KAMOUNTRT=KAMOUNTRT, KAMOUNTSO=KAMOUNTSO,
                        NUPTAKETOTAL=0, PUPTAKETOTAL=0., KUPTAKETOTAL=0., NFIXTOTAL=0.,
                        NlossesTotal=0, PlossesTotal=0., KlossesTotal=0.)
        
        self.rates = self.RateVariables(kiosk=self.kiosk,
            publish=[])

    def calc_rates(self, day:date, drv):
        """Calculate state rates
        """
        r = self.rates
        p = self.params
        k = self.kiosk
        
        self.demand_uptake.calc_rates(day, drv)
        self.translocation.calc_rates(day, drv)

        r.RNDEATHLV = p.NRESIDLV * k.DRLV
        r.RNDEATHST = p.NRESIDST * k.DRST
        r.RNDEATHRT = p.NRESIDRT * k.DRRT

        r.RPDEATHLV = p.PRESIDLV * k.DRLV
        r.RPDEATHST = p.PRESIDST * k.DRST
        r.RPDEATHRT = p.PRESIDRT * k.DRRT

        r.RKDEATHLV = p.KRESIDLV * k.DRLV
        r.RKDEATHST = p.KRESIDST * k.DRST
        r.RKDEATHRT = p.KRESIDRT * k.DRRT

        r.RNAMOUNTLV = k.RNUPTAKELV - k.RNTRANSLOCATIONLV - r.RNDEATHLV
        r.RNAMOUNTST = k.RNUPTAKEST - k.RNTRANSLOCATIONST - r.RNDEATHST
        r.RNAMOUNTRT = k.RNUPTAKERT - k.RNTRANSLOCATIONRT - r.RNDEATHRT
        r.RNAMOUNTSO = k.RNUPTAKESO
        
        r.RPAMOUNTLV = k.RPUPTAKELV - k.RPTRANSLOCATIONLV - r.RPDEATHLV
        r.RPAMOUNTST = k.RPUPTAKEST - k.RPTRANSLOCATIONST - r.RPDEATHST
        r.RPAMOUNTRT = k.RPUPTAKERT - k.RPTRANSLOCATIONRT - r.RPDEATHRT
        r.RPAMOUNTSO = k.RPUPTAKESO

        r.RKAMOUNTLV = k.RKUPTAKELV - k.RKTRANSLOCATIONLV - r.RKDEATHLV
        r.RKAMOUNTST = k.RKUPTAKEST - k.RKTRANSLOCATIONST - r.RKDEATHST
        r.RKAMOUNTRT = k.RKUPTAKERT - k.RKTRANSLOCATIONRT - r.RKDEATHRT
        r.RKAMOUNTSO = k.RKUPTAKESO
        
        r.RNLOSS = r.RNDEATHLV + r.RNDEATHST + r.RNDEATHRT
        r.RPLOSS = r.RPDEATHLV + r.RPDEATHST + r.RPDEATHRT
        r.RKLOSS = r.RKDEATHLV + r.RKDEATHST + r.RKDEATHRT

        self.rates._update_kiosk()

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        r = self.rates
        s = self.states
        k = self.kiosk

        s.NAMOUNTLV = s.NAMOUNTLV + r.RNAMOUNTLV
        s.NAMOUNTST = s.NAMOUNTST + r.RNAMOUNTST
        s.NAMOUNTRT = s.NAMOUNTRT + r.RNAMOUNTRT
        s.NAMOUNTSO = s.NAMOUNTSO + r.RNAMOUNTSO
        
        s.PAMOUNTLV = s.PAMOUNTLV + r.RPAMOUNTLV
        s.PAMOUNTST = s.PAMOUNTST + r.RPAMOUNTST
        s.PAMOUNTRT = s.PAMOUNTRT + r.RPAMOUNTRT
        s.PAMOUNTSO = s.PAMOUNTSO + r.RPAMOUNTSO

        s.KAMOUNTLV = s.KAMOUNTLV + r.RKAMOUNTLV
        s.KAMOUNTST = s.KAMOUNTST + r.RKAMOUNTST
        s.KAMOUNTRT = s.KAMOUNTRT + r.RKAMOUNTRT
        s.KAMOUNTSO = s.KAMOUNTSO + r.RKAMOUNTSO
        
        self.translocation.integrate(day, delt)
        self.demand_uptake.integrate(day, delt)

        s.NUPTAKETOTAL = s.NUPTAKETOTAL + k.RNUPTAKE
        s.PUPTAKETOTAL = s.PUPTAKETOTAL + k.RPUPTAKE
        s.KUPTAKETOTAL = s.KUPTAKETOTAL + k.RKUPTAKE
        s.NFIXTOTAL    = s.NFIXTOTAL + k.RNFIXATION
        
        s.NlossesTotal = s.NlossesTotal + r.RNLOSS
        s.PlossesTotal = s.PlossesTotal + r.RPLOSS
        s.KlossesTotal = s.KlossesTotal + r.RKLOSS

        self.states._update_kiosk()

    def reset(self, day:date):
        """Reset states and rates
        """
        
        self.translocation.reset(day)
        self.demand_uptake.reset(day)

        p = self.params
        k = self.kiosk
        
        self.NAMOUNTLVI = NAMOUNTLV = k.WLV * p.NMAXLV_TB(k.DVS)
        self.NAMOUNTSTI = NAMOUNTST = k.WST * p.NMAXLV_TB(k.DVS) * p.NMAXST_FR
        self.NAMOUNTRTI = NAMOUNTRT = k.WRT * p.NMAXLV_TB(k.DVS) * p.NMAXRT_FR
        self.NAMOUNTSOI = NAMOUNTSO = 0.
        
        self.PAMOUNTLVI = PAMOUNTLV = k.WLV * p.PMAXLV_TB(k.DVS)
        self.PAMOUNTSTI = PAMOUNTST = k.WST * p.PMAXLV_TB(k.DVS) * p.PMAXST_FR
        self.PAMOUNTRTI = PAMOUNTRT = k.WRT * p.PMAXLV_TB(k.DVS) * p.PMAXRT_FR
        self.PAMOUNTSOI = PAMOUNTSO = 0.

        self.KAMOUNTLVI = KAMOUNTLV = k.WLV * p.KMAXLV_TB(k.DVS)
        self.KAMOUNTSTI = KAMOUNTST = k.WST * p.KMAXLV_TB(k.DVS) * p.KMAXST_FR
        self.KAMOUNTRTI = KAMOUNTRT = k.WRT * p.KMAXLV_TB(k.DVS) * p.KMAXRT_FR
        self.KAMOUNTSOI = KAMOUNTSO = 0.

        self.states = self.StateVariables(kiosk=self.kiosk,
            publish=["NAMOUNTLV", "NAMOUNTST", "NAMOUNTRT", "PAMOUNTLV", "PAMOUNTST", "PAMOUNTRT",
                     "KAMOUNTLV", "KAMOUNTST", "KAMOUNTRT", "NAMOUNTSO", "PAMOUNTSO", "KAMOUNTSO"],
                        NAMOUNTLV=NAMOUNTLV, NAMOUNTST=NAMOUNTST, NAMOUNTRT=NAMOUNTRT, NAMOUNTSO=NAMOUNTSO,
                        PAMOUNTLV=PAMOUNTLV, PAMOUNTST=PAMOUNTST, PAMOUNTRT=PAMOUNTRT, PAMOUNTSO=PAMOUNTSO,
                        KAMOUNTLV=KAMOUNTLV, KAMOUNTST=KAMOUNTST, KAMOUNTRT=KAMOUNTRT, KAMOUNTSO=KAMOUNTSO,
                        NUPTAKETOTAL=0, PUPTAKETOTAL=0., KUPTAKETOTAL=0., NFIXTOTAL=0.,
                        NlossesTotal=0, PlossesTotal=0., KlossesTotal=0.)
        
        self.rates = self.RateVariables(kiosk=self.kiosk,
            publish=[])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.NUPTAKETOTAL
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