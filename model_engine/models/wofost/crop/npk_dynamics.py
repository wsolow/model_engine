"""Overall implementation for the NPK dynamics of the crop including
subclasses to 
    * NPK Demand Uptake
    * NPK Stress
    * NPK Translocation
    
Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date

from traitlets_pcse import Instance

from model_engine.models.wofost.crop.nutrients.npk_demand_uptake import NPK_Demand_Uptake
from model_engine.models.wofost.crop.nutrients.npk_translocation import NPK_Translocation


from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class NPK_Crop_Dynamics(TensorModel):
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
        
    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: current day
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameters as key/value pairs
        """  
        
        self.params = self.Parameters(parvalues)
        self.kiosk = kiosk
        
        
        self.translocation = NPK_Translocation(day, kiosk, parvalues)
        self.demand_uptake = NPK_Demand_Uptake(day, kiosk, parvalues)

        
        params = self.params
        k = kiosk

        
        self.NAMOUNTLVI = NAMOUNTLV = k.WLV * params.NMAXLV_TB(k.DVS)
        self.NAMOUNTSTI = NAMOUNTST = k.WST * params.NMAXLV_TB(k.DVS) * params.NMAXST_FR
        self.NAMOUNTRTI = NAMOUNTRT = k.WRT * params.NMAXLV_TB(k.DVS) * params.NMAXRT_FR
        self.NAMOUNTSOI = NAMOUNTSO = 0.
        
        self.PAMOUNTLVI = PAMOUNTLV = k.WLV * params.PMAXLV_TB(k.DVS)
        self.PAMOUNTSTI = PAMOUNTST = k.WST * params.PMAXLV_TB(k.DVS) * params.PMAXST_FR
        self.PAMOUNTRTI = PAMOUNTRT = k.WRT * params.PMAXLV_TB(k.DVS) * params.PMAXRT_FR
        self.PAMOUNTSOI = PAMOUNTSO = 0.

        self.KAMOUNTLVI = KAMOUNTLV = k.WLV * params.KMAXLV_TB(k.DVS)
        self.KAMOUNTSTI = KAMOUNTST = k.WST * params.KMAXLV_TB(k.DVS) * params.KMAXST_FR
        self.KAMOUNTRTI = KAMOUNTRT = k.WRT * params.KMAXLV_TB(k.DVS) * params.KMAXRT_FR
        self.KAMOUNTSOI = KAMOUNTSO = 0.

        self.states = self.StateVariables(kiosk,
            publish=["NAMOUNTLV", "PAMOUNTLV", "KAMOUNTLV", "NAMOUNTST", "PAMOUNTST", 
                     "KAMOUNTST", "NAMOUNTSO", "PAMOUNTSO", "KAMOUNTSO", "NAMOUNTRT", 
                     "PAMOUNTRT", "KAMOUNTRT","NUPTAKETOTAL", "PUPTAKETOTAL", "KUPTAKETOTAL", 
                     "NFIXTOTAL", "NlossesTotal", "PlossesTotal", "KlossesTotal"],
                        NAMOUNTLV=NAMOUNTLV, NAMOUNTST=NAMOUNTST, NAMOUNTRT=NAMOUNTRT, NAMOUNTSO=NAMOUNTSO,
                        PAMOUNTLV=PAMOUNTLV, PAMOUNTST=PAMOUNTST, PAMOUNTRT=PAMOUNTRT, PAMOUNTSO=PAMOUNTSO,
                        KAMOUNTLV=KAMOUNTLV, KAMOUNTST=KAMOUNTST, KAMOUNTRT=KAMOUNTRT, KAMOUNTSO=KAMOUNTSO,
                        NUPTAKETOTAL=0, PUPTAKETOTAL=0., KUPTAKETOTAL=0., NFIXTOTAL=0.,
                        NlossesTotal=0, PlossesTotal=0., KlossesTotal=0.)
        
        self.rates = self.RateVariables(kiosk,
            publish=["RNAMOUNTLV", "RPAMOUNTLV", "RKAMOUNTLV", "RNAMOUNTST", 
                     "RPAMOUNTST", "RKAMOUNTST", "RNAMOUNTRT", "RPAMOUNTRT",  
                     "RKAMOUNTRT", "RNAMOUNTSO", "RPAMOUNTSO", "RKAMOUNTSO", 
                     "RNDEATHLV", "RNDEATHST", "RNDEATHRT", "RPDEATHLV", "RPDEATHST", 
                     "RPDEATHRT", "RKDEATHLV","RKDEATHST", "RKDEATHRT", "RNLOSS", 
                     "RPLOSS", "RKLOSS"])

    
    def calc_rates(self, day:date, drv):
        """Calculate state rates
        """
        rates = self.rates
        params = self.params
        k = self.kiosk
        
        self.demand_uptake.calc_rates(day, drv)
        self.translocation.calc_rates(day, drv)

        
        rates.RNDEATHLV = params.NRESIDLV * k.DRLV
        rates.RNDEATHST = params.NRESIDST * k.DRST
        rates.RNDEATHRT = params.NRESIDRT * k.DRRT

        rates.RPDEATHLV = params.PRESIDLV * k.DRLV
        rates.RPDEATHST = params.PRESIDST * k.DRST
        rates.RPDEATHRT = params.PRESIDRT * k.DRRT

        rates.RKDEATHLV = params.KRESIDLV * k.DRLV
        rates.RKDEATHST = params.KRESIDST * k.DRST
        rates.RKDEATHRT = params.KRESIDRT * k.DRRT

        
        
        
        rates.RNAMOUNTLV = k.RNUPTAKELV - k.RNTRANSLOCATIONLV - rates.RNDEATHLV
        rates.RNAMOUNTST = k.RNUPTAKEST - k.RNTRANSLOCATIONST - rates.RNDEATHST
        rates.RNAMOUNTRT = k.RNUPTAKERT - k.RNTRANSLOCATIONRT - rates.RNDEATHRT
        rates.RNAMOUNTSO = k.RNUPTAKESO
        
        
        rates.RPAMOUNTLV = k.RPUPTAKELV - k.RPTRANSLOCATIONLV - rates.RPDEATHLV
        rates.RPAMOUNTST = k.RPUPTAKEST - k.RPTRANSLOCATIONST - rates.RPDEATHST
        rates.RPAMOUNTRT = k.RPUPTAKERT - k.RPTRANSLOCATIONRT - rates.RPDEATHRT
        rates.RPAMOUNTSO = k.RPUPTAKESO

        
        rates.RKAMOUNTLV = k.RKUPTAKELV - k.RKTRANSLOCATIONLV - rates.RKDEATHLV
        rates.RKAMOUNTST = k.RKUPTAKEST - k.RKTRANSLOCATIONST - rates.RKDEATHST
        rates.RKAMOUNTRT = k.RKUPTAKERT - k.RKTRANSLOCATIONRT - rates.RKDEATHRT
        rates.RKAMOUNTSO = k.RKUPTAKESO
        
        rates.RNLOSS = rates.RNDEATHLV + rates.RNDEATHST + rates.RNDEATHRT
        rates.RPLOSS = rates.RPDEATHLV + rates.RPDEATHST + rates.RPDEATHRT
        rates.RKLOSS = rates.RKDEATHLV + rates.RKDEATHST + rates.RKDEATHRT

        self._check_N_balance(day)
        self._check_P_balance(day)
        self._check_K_balance(day)
        
    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        rates = self.rates
        states = self.states
        k = self.kiosk

        
        states.NAMOUNTLV += rates.RNAMOUNTLV
        states.NAMOUNTST += rates.RNAMOUNTST
        states.NAMOUNTRT += rates.RNAMOUNTRT
        states.NAMOUNTSO += rates.RNAMOUNTSO
        
        
        states.PAMOUNTLV += rates.RPAMOUNTLV
        states.PAMOUNTST += rates.RPAMOUNTST
        states.PAMOUNTRT += rates.RPAMOUNTRT
        states.PAMOUNTSO += rates.RPAMOUNTSO

        
        states.KAMOUNTLV += rates.RKAMOUNTLV
        states.KAMOUNTST += rates.RKAMOUNTST
        states.KAMOUNTRT += rates.RKAMOUNTRT
        states.KAMOUNTSO += rates.RKAMOUNTSO
        
        self.translocation.integrate(day, delt)
        self.demand_uptake.integrate(day, delt)

        
        states.NUPTAKETOTAL += k.RNUPTAKE
        states.PUPTAKETOTAL += k.RPUPTAKE
        states.KUPTAKETOTAL += k.RKUPTAKE
        states.NFIXTOTAL += k.RNFIXATION
        
        states.NlossesTotal += rates.RNLOSS
        states.PlossesTotal += rates.RPLOSS
        states.KlossesTotal += rates.RKLOSS

    def _check_N_balance(self, day:date):
        """Check the Nitrogen balance is valid"""
        s = self.states

    def _check_P_balance(self, day:date):
        """Check that the Phosphorous balance is valid"""
        s = self.states

    def _check_K_balance(self, day:date):
        """Check that the Potassium balance is valid"""
        s = self.states

    def reset(self):
        """Reset states and rates
        """
        
        self.translocation.reset()
        self.demand_uptake.reset()

        params = self.params
        k = self.kiosk
        s = self.states
        r = self.rates

        self.NAMOUNTLVI = NAMOUNTLV = k.WLV * params.NMAXLV_TB(k.DVS)
        self.NAMOUNTSTI = NAMOUNTST = k.WST * params.NMAXLV_TB(k.DVS) * params.NMAXST_FR
        self.NAMOUNTRTI = NAMOUNTRT = k.WRT * params.NMAXLV_TB(k.DVS) * params.NMAXRT_FR
        self.NAMOUNTSOI = NAMOUNTSO = 0.
        
        self.PAMOUNTLVI = PAMOUNTLV = k.WLV * params.PMAXLV_TB(k.DVS)
        self.PAMOUNTSTI = PAMOUNTST = k.WST * params.PMAXLV_TB(k.DVS) * params.PMAXST_FR
        self.PAMOUNTRTI = PAMOUNTRT = k.WRT * params.PMAXLV_TB(k.DVS) * params.PMAXRT_FR
        self.PAMOUNTSOI = PAMOUNTSO = 0.

        self.KAMOUNTLVI = KAMOUNTLV = k.WLV * params.KMAXLV_TB(k.DVS)
        self.KAMOUNTSTI = KAMOUNTST = k.WST * params.KMAXLV_TB(k.DVS) * params.KMAXST_FR
        self.KAMOUNTRTI = KAMOUNTRT = k.WRT * params.KMAXLV_TB(k.DVS) * params.KMAXRT_FR
        self.KAMOUNTSOI = KAMOUNTSO = 0.

        s.NAMOUNTLV=NAMOUNTLV
        s.NAMOUNTST=NAMOUNTST
        s.NAMOUNTRT=NAMOUNTRT
        s.NAMOUNTSO=NAMOUNTSO
        s.PAMOUNTLV=PAMOUNTLV
        s.PAMOUNTST=PAMOUNTST
        s.PAMOUNTRT=PAMOUNTRT
        s.PAMOUNTSO=PAMOUNTSO
        s.KAMOUNTLV=KAMOUNTLV
        s.KAMOUNTST=KAMOUNTST
        s.KAMOUNTRT=KAMOUNTRT
        s.KAMOUNTSO=KAMOUNTSO
        s.NUPTAKETOTAL=0
        s.PUPTAKETOTAL=0.
        s.KUPTAKETOTAL=0.
        s.NFIXTOTAL=0.
        s.NlossesTotal=0
        s.PlossesTotal=0.
        s.KlossesTotal=0.

        r.RNAMOUNTLV = r.RPAMOUNTLV = r.RKAMOUNTLV = r.RNAMOUNTST = r.RPAMOUNTST \
            = r.RKAMOUNTST = r.RNAMOUNTRT = r.RPAMOUNTRT = r.RKAMOUNTRT = r.RNAMOUNTSO \
            = r.RPAMOUNTSO = r.RKAMOUNTSO = r.RNDEATHLV = r.RNDEATHST = r.RNDEATHRT \
            = r.RPDEATHLV = r.RPDEATHST = r.RPDEATHRT = r.RKDEATHLV = r.RKDEATHST \
            = r.RKDEATHRT = r.RNLOSS = r.RPLOSS = r.RKLOSS = 0
        