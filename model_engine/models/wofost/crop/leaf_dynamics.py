"""Handles growth of leaf dynamics in the crop

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
from math import exp
from collections import deque
from array import array
from datetime import date

from traitlets_pcse import Instance

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WOFOST_Leaf_Dynamics_NPK(TensorModel):
    """Leaf dynamics for the WOFOST crop model including leaf response to
    NPK stress.
    """

    class Parameters(ParamTemplate):
        RGRLAI = Tensor(-99.)
        SPAN = Tensor(-99.)
        TBASE = Tensor(-99.)
        PERDL = Tensor(-99.)
        TDWI = Tensor(-99.)
        SLATB = TensorAfgenTrait()
        KDIFTB = TensorAfgenTrait()
        RDRLV_NPK = Tensor(-99.)  
        NSLA_NPK = Tensor(-99.)  
        NLAI_NPK = Tensor(-99.)  
                                  

    class StateVariables(StatesTemplate):
        LV = Instance(deque)
        SLA = Instance(deque)
        LVAGE = Instance(deque)
        LAIEM = Tensor(-99.)
        LASUM = Tensor(-99.)
        LAIEXP = Tensor(-99.)
        LAIMAX = Tensor(-99.)
        LAI = Tensor(-99.)
        WLV = Tensor(-99.)
        DWLV = Tensor(-99.)
        TWLV = Tensor(-99.)

    class RateVariables(RatesTemplate):
        GRLV = Tensor(-99.)
        DSLV1 = Tensor(-99.)
        DSLV2 = Tensor(-99.)
        DSLV3 = Tensor(-99.)
        DSLV4 = Tensor(-99.)
        DSLV = Tensor(-99.)
        DALV = Tensor(-99.)
        DRLV = Tensor(-99.)
        SLAT = Tensor(-99.)
        FYSAGE = Tensor(-99.)
        GLAIEX = Tensor(-99.)
        GLASOL = Tensor(-99.)

    def __init__(self, day:date, kiosk, cropdata:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param cropdata: dictionary with WOFOST cropdata key/value pairs
        """

        self.kiosk = kiosk
        self.params = self.Parameters(cropdata)

        
        p = self.params
        k = self.kiosk
        
        WLV = (p.TDWI * (1-k.FR)) * k.FL
        DWLV = 0.
        TWLV = WLV + DWLV

        
        SLA = deque([p.SLATB(k.DVS)])
        LVAGE = deque([0.])
        LV = deque([WLV])

        
        LAIEM = LV[0] * SLA[0]
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        SAI = PAI = 0
        if "SAI" in self.kiosk:
            SAI = k.SAI
        if "PAI" in self.kiosk:
            PAI = k.PAI
        LAI = LASUM + SAI + PAI

        
        self.states = self.StateVariables(kiosk, 
                publish=["LV", "SLA", "LVAGE", "LAIEM", "LASUM", "LAIEXP", 
                         "LAIMAX", "LAI", "WLV", "DWLV", "TWLV"], 
                LV=LV, SLA=SLA, LVAGE=LVAGE, LAIEM=LAIEM, LASUM=LASUM, LAIEXP=LAIEXP, 
                LAIMAX=LAIMAX, LAI=LAI, WLV=WLV, DWLV=DWLV, TWLV=TWLV)
        
        self.rates = self.RateVariables(kiosk,
                publish=["GRLV", "DSLV1", "DSLV2", "DSLV3", "DSLV4", "DSLV", 
                         "DALV", "DRLV", "SLAT", "FYSAGE", "GLAIEX", "GLASOL"])
    
    def _calc_LAI(self):
        """Compute LAI as Total leaf area Index as sum of leaf, pod and stem area
        """
        k = self.kiosk
        SAI = PAI = 0
        if "SAI" in self.kiosk:
            SAI = k.SAI
        if "PAI" in self.kiosk:
            PAI = k.PAI
        return self.states.LASUM + SAI + PAI

    def calc_rates(self, day:date, drv):
        """Calculate state rates
        """
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        
        
        r.GRLV = k.ADMI * k.FL

        
        r.DSLV1 = s.WLV * (1.-k.RFTRA) * p.PERDL

        
        LAICR = 3.2/p.KDIFTB(k.DVS)
        r.DSLV2 = s.WLV * limit(0., 0.03, 0.03*(s.LAI-LAICR)/LAICR)

        
        
        if "RF_FROST" in self.kiosk:
            r.DSLV3 = s.WLV * k.RF_FROST
        else:
            r.DSLV3 = 0.

        
        
        
        r.DSLV4 = s.WLV * p.RDRLV_NPK * (1.0 - self.kiosk["NPKI"])

        
        
        r.DSLV = max(r.DSLV1, r.DSLV2, r.DSLV3) + r.DSLV4

        
        
        
        
        
        DALV = 0.0
        for lv, lvage in zip(s.LV, s.LVAGE):
            if lvage > p.SPAN:
                DALV += lv
        r.DALV = DALV

        
        r.DRLV = max(r.DSLV, r.DALV)

        
        r.FYSAGE = max(0., (drv.TEMP - p.TBASE)/(35. - p.TBASE))

        
        
        sla_npk_factor = exp(-p.NSLA_NPK * (1.0 - k.NPKI))

        
        r.SLAT = p.SLATB(k.DVS) * sla_npk_factor

        
        if s.LAIEXP < 6.:
            DTEFF = max(0., drv.TEMP-p.TBASE)

            
            
            if k.DVS < 0.2 and s.LAI < 0.75:
                factor = k.RFTRA * exp(-p.NLAI_NPK * (1.0 - k.NPKI))
            else:
                factor = 1.

            r.GLAIEX = s.LAIEXP * p.RGRLAI * DTEFF * factor
            
            r.GLASOL = r.GRLV * r.SLAT
            
            GLA = min(r.GLAIEX, r.GLASOL)
            
            if r.GRLV > 0.:
                r.SLAT = GLA/r.GRLV

    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates to new state
        """
        p = self.params
        r = self.rates
        s = self.states

        
        tLV = array('d', s.LV)
        tSLA = array('d', s.SLA)
        tLVAGE = array('d', s.LVAGE)
        tDRLV = r.DRLV

        
        
        for LVweigth in reversed(s.LV):
            if tDRLV > 0.:
                if tDRLV >= LVweigth: 
                    tDRLV -= LVweigth
                    tLV.pop()
                    tLVAGE.pop()
                    tSLA.pop()
                else: 
                    tLV[-1] -= tDRLV
                    tDRLV = 0.
            else:
                break

        
        tLVAGE = deque([age + r.FYSAGE for age in tLVAGE])
        tLV = deque(tLV)
        tSLA = deque(tSLA)

        
        
        tLV.appendleft(r.GRLV)
        tSLA.appendleft(r.SLAT)
        tLVAGE.appendleft(0.)

        
        s.LASUM = sum([lv*sla for lv, sla in zip(tLV, tSLA)])
        s.LAI = self._calc_LAI()
        s.LAIMAX = max(s.LAI, s.LAIMAX)

        
        s.LAIEXP += r.GLAIEX

        
        s.WLV  = sum(tLV)
        s.DWLV += r.DRLV
        s.TWLV = s.WLV + s.DWLV

        
        self.states.LV = tLV
        self.states.SLA = tSLA
        self.states.LVAGE = tLVAGE

    def reset(self):
        """Reset states and rates
        """
        
        p = self.params
        k = self.kiosk
        s = self.states
        r = self.rates
        
        WLV = (p.TDWI * (1-k.FR)) * k.FL
        DWLV = 0.
        TWLV = WLV + DWLV

        
        SLA = deque([p.SLATB(k.DVS)])
        LVAGE = deque([0.])
        LV = deque([WLV])

        
        LAIEM = LV[0] * SLA[0]
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        SAI = PAI = 0
        if "SAI" in self.kiosk:
            SAI = k.SAI
        if "PAI" in self.kiosk:
            PAI = k.PAI
        LAI = LASUM + SAI + PAI

        s.LV=LV
        s.SLA=SLA
        s.LVAGE=LVAGE
        s.LAIEM=LAIEM
        s.LASUM=LASUM
        s.LAIEXP=LAIEXP
        s.LAIMAX=LAIMAX
        s.LAI=LAI
        s.WLV=WLV
        s.DWLV=DWLV
        s.TWLV=TWLV

        r.GRLV = r.DSLV1 = r.DSLV2 = r.DSLV3 = r.DSLV4 = r.DSLV = r.DALV = r.DRLV \
            = r.SLAT = r.FYSAGE = r.GLAIEX = r.GLASOL = 0