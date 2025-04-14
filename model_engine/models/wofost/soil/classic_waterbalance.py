"""Python implementations of the WOFOST waterbalance modules for simulation
of potential production (`WaterbalancePP`) and water-limited production
(`WaterbalanceFD`) under freely draining conditions.

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
from datetime import date
from math import sqrt

from traitlets_pcse import Instance, List

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WaterbalanceFD(TensorModel):
    """Waterbalance for freely draining soils under water-limited production.

    """
    
    RDold = Tensor(-99.)
    RDM = Tensor(-99.)
    
    DSLR = Tensor(-99.)
    
    RINold = Tensor(-99)
    
    NINFTB = Instance(TensorAfgenTrait)
    
    _RIRR = Tensor(0.)
    
    DEFAULT_RD = Tensor(10.)
    
    _increments_W = List()

    class Parameters(ParamTemplate):
        
        SMFCF  = Tensor(-99.)
        SM0    = Tensor(-99.)
        SMW    = Tensor(-99.)
        CRAIRC = Tensor(-99.)
        SOPE   = Tensor(-99.)
        KSUB   = Tensor(-99.)
        RDMSOL = Tensor(-99.)
        SMLIM  = Tensor(-99.)
        
        IFUNRN = Tensor(-99.)
        SSMAX  = Tensor(-99.)
        SSI    = Tensor(-99.)
        WAV    = Tensor(-99.)
        NOTINF = Tensor(-99.)

    class StateVariables(StatesTemplate):
        SM = Tensor(-99.)
        SS = Tensor(-99.)
        SSI = Tensor(-99.)
        WC  = Tensor(-99.)
        WI = Tensor(-99.)
        WLOW  = Tensor(-99.)
        WLOWI = Tensor(-99.)
        WWLOW = Tensor(-99.)
        
        WTRAT    = Tensor(-99.)
        EVST     = Tensor(-99.)
        EVWT     = Tensor(-99.)
        TSR      = Tensor(-99.)
        RAINT    = Tensor(-99.)
        WART     = Tensor(-99.)
        TOTINF   = Tensor(-99.)
        TOTIRR   = Tensor(-99.)
        TOTIRRIG = Tensor(-99.)
        PERCT    = Tensor(-99.)
        LOSST    = Tensor(-99.)
        
        WBALRT = Tensor(-99.)
        WBALTT = Tensor(-99.)
        DSOS = Tensor(-99)

    class RateVariables(RatesTemplate):
        EVS   = Tensor(-99.)
        EVW   = Tensor(-99.)
        WTRA  = Tensor(-99.)
        RIN   = Tensor(-99.)
        RIRR  = Tensor(-99.)
        PERC  = Tensor(-99.)
        LOSS  = Tensor(-99.)
        DW    = Tensor(-99.)
        DWLOW = Tensor(-99.)
        DTSR = Tensor(-99.)
        DSS = Tensor(-99.)
        DRAINT = Tensor(-99.)

    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: ParameterProvider containing all parameters
        """

        
        if parvalues["SM0"] < parvalues["SMW"]:
            parvalues["SM0"] = parvalues["SMW"] + .000001
        SMLIM = limit(parvalues["SMW"], parvalues["SM0"],  parvalues["SMLIM"])

        if SMLIM != parvalues["SMLIM"]:
            pass
            
            

        
        self.params = self.Parameters(parvalues)
        p = self.params
        
        RD = self.DEFAULT_RD
        RDM = max(RD, p.RDMSOL)
        self.RDold = RD
        self.RDM = RDM
        
        
        SS = p.SSI
        
        
        
        SM = limit(p.SMW, SMLIM, (p.SMW + p.WAV/RD))
        WC = SM * RD
        WI = WC
        
        
        
        WLOW  = limit(0., p.SM0*(RDM - RD), (p.WAV + RDM*p.SMW - WC))
        WLOWI = WLOW
        
        
        WWLOW = WC + WLOW

        
        
        self.DSLR = 1. if (SM >= (p.SMW + 0.5*(p.SMFCF-p.SMW))) else 5.

        
        self.RINold = 0.
        self.NINFTB = Afgen([0.0,0.0, 0.5,0.0, 1.5,1.0])

        
        self.states = self.StateVariables(kiosk, 
                                          publish=["SM", "SS", "SSI", "WC", "WI", 
                                                   "WLOW", "WLOWI", "WWLOW", "WTRAT", 
                                                   "EVST", "EVWT", "TSR", "RAINT", 
                                                   "WART", "TOTINF", "TOTIRR", "PERCT", 
                                                   "LOSST", "WBALRT", "WBALTT", "DSOS",
                                                   "TOTIRRIG"], 
                           SM=SM, SS=SS,
                           SSI=p.SSI, WC=WC, WI=WI, WLOW=WLOW, WLOWI=WLOWI,
                           WWLOW=WWLOW, WTRAT=0., EVST=0., EVWT=0., TSR=0.,
                           RAINT=0., WART=0., TOTINF=0., TOTIRR=0., DSOS=0,
                           PERCT=0., LOSST=0., WBALRT=-999., WBALTT=-999., 
                           TOTIRRIG=0.)
        self.rates = self.RateVariables(kiosk, 
                                        publish=["EVS", "EVW", "WTRA", "RIN", 
                                                 "RIRR", "PERC", "LOSS", "DW",
                                                 "DWLOW", "DTSR", "DSS", "DRAINT"])
        
        
        
        
        
        
        

        self._increments_W = []

    
    def calc_rates(self, day:date, drv):
        """Calculate state rates for integration
        """
        s = self.states
        p = self.params
        r = self.rates
        k = self.kiosk

        
        r.RIRR = self._RIRR
        self._RIRR = 0.

        
        
        
        
        
        if "TRA" not in self.kiosk:
            r.WTRA = 0.
            EVWMX = drv.E0
            EVSMX = drv.ES0
        else:
            r.WTRA = k.TRA
            EVWMX = k.EVWMX
            EVSMX = k.EVSMX

        
        r.EVW = 0.
        r.EVS = 0.
        if s.SS > 1.:
            
            
            r.EVW = EVWMX
        else:
            
            if self.RINold >= 1:
                
                
                r.EVS = EVSMX
                self.DSLR = 1.
            else:
                
                EVSMXT = EVSMX * (sqrt(self.DSLR + 1) - sqrt(self.DSLR))
                r.EVS = min(EVSMX, EVSMXT + self.RINold)
                self.DSLR += 1

        
        if p.IFUNRN == 0:
            RINPRE = (1. - p.NOTINF) * drv.RAIN
        else:
            
            RINPRE = (1. - p.NOTINF * self.NINFTB(drv.RAIN)) * drv.RAIN


        
        
        RINPRE = RINPRE + r.RIRR + s.SS
        if s.SS > 0.1:
            
            AVAIL = RINPRE + r.RIRR - r.EVW
            RINPRE = min(p.SOPE, AVAIL)
            
        RD = self._determine_rooting_depth()
        
        
        WE = p.SMFCF * RD
        
        
        
        PERC1 = limit(0., p.SOPE, (s.WC - WE) - r.WTRA - r.EVS)

        
        
        WELOW = p.SMFCF * (self.RDM - RD)
        r.LOSS = limit(0., p.KSUB, (s.WLOW - WELOW + PERC1))

        
        PERC2 = ((self.RDM - RD) * p.SM0 - s.WLOW) + r.LOSS
        r.PERC = min(PERC1, PERC2)

        
        r.RIN = min(RINPRE, (p.SM0 - s.SM)*RD + r.WTRA + r.EVS + r.PERC)
        self.RINold = r.RIN

        
        r.DW = r.RIN - r.WTRA - r.EVS - r.PERC
        r.DWLOW = r.PERC - r.LOSS

        
        
        Wtmp = s.WC + r.DW
        if Wtmp < 0.0:
            r.EVS += Wtmp
            
            r.DW = -s.WC

        
        
        
        
        SStmp = drv.RAIN + r.RIRR - r.EVW - r.RIN
        
        r.DSS = min(SStmp, (p.SSMAX - s.SS))
        
        r.DTSR = SStmp - r.DSS
        
        r.DRAINT = drv.RAIN

    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate states from rates
        """
        s = self.states
        p = self.params
        r = self.rates
        
        

        
        s.WTRAT += r.WTRA * delt

        
        s.EVWT += r.EVW * delt
        s.EVST += r.EVS * delt

        
        s.RAINT += r.DRAINT * delt
        s.TOTINF += r.RIN * delt
        s.TOTIRR += r.RIRR * delt

        
        s.SS += r.DSS * delt
        s.TSR += r.DTSR * delt

        
        s.WC += r.DW * delt
        assert s.WC >= 0., "Negative amount of water in root zone on day %s: %s" % (day, s.WC)

        
        s.PERCT += r.PERC * delt
        s.LOSST += r.LOSS * delt

        
        s.WLOW += r.DWLOW * delt
        
        s.WWLOW = s.WC + s.WLOW * delt

        

        
        RD = self._determine_rooting_depth()
        RDchange = RD - self.RDold
        self._redistribute_water(RDchange)

        
        s.SM = s.WC/RD

        
        if s.SM >= (p.SM0 - p.CRAIRC):
            s.DSOS += 1

        
        self.RDold = RD

    
    def finalize(self, day:date):
        """Finalize states
        """
        s = self.states
        p = self.params

        s.WBALRT = s.TOTINF + s.WI + s.WART - s.EVST - s.WTRAT - s.PERCT - s.WC + sum(self._increments_W)
        s.WBALTT = (s.SSI + s.RAINT + s.TOTIRR + s.WI - s.WC + sum(self._increments_W) +
                    s.WLOWI - s.WLOW - s.WTRAT - s.EVWT - s.EVST - s.TSR - s.LOSST - s.SS)


    def _determine_rooting_depth(self):
        """Determines appropriate use of the rooting depth (RD)

        This function includes the logic to determine the depth of the upper (rooted)
        layer of the water balance. See the comment in the code for a detailed description.
        """
        if "RD" in self.kiosk:
            return self.kiosk["RD"]
        else:
            
            return self.DEFAULT_RD

    def _redistribute_water(self, RDchange:float):
        """Redistributes the water between the root zone and the lower zone.

        :param RDchange: Change in root depth [cm] positive for downward growth,
                         negative for upward growth

        Redistribution of water is needed when roots grow during the growing season
        and when the crop is finished and the root zone shifts back from the crop rooted
        depth to the default depth of the upper (rooted) layer of the water balance.
        Or when the initial rooting depth of a crop is different from the default one used
        by the water balance module (10 cm)
        """
        s = self.states
        p = self.params
        
        WDR = 0.
        if RDchange > 0.001:
            
            
            WDR = s.WLOW * RDchange/(p.RDMSOL - self.RDold)
            
            WDR = min(s.WLOW, WDR)
        else:
            
            
            WDR = s.WC * RDchange/self.RDold

        if WDR != 0.:
            
            s.WLOW -= WDR
            
            s.WC += WDR
            
            s.WART += WDR
