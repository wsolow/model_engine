"""Handles growth of leaf dynamics in the crop

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
from math import exp
from collections import deque
from array import array
from datetime import date
import torch

from traitlets_pcse import Instance

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class WOFOST_Leaf_Dynamics_NPK_Tensor(TensorModel):
    """Leaf dynamics for the WOFOST crop model including leaf response to
    NPK stress.
    """

    LV = Instance(deque)
    SLA = Instance(deque)
    LVAGE = Instance(deque)

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

    def __init__(self, day:date, kiosk, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        p = self.params
        k = self.kiosk
        
        WLV = (p.TDWI * (1 - k.FR)) * k.FL
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

        self.LV = LV
        self.SLA = SLA
        self.LVAGE = LVAGE

        self.states = self.StateVariables(kiosk=self.kiosk, 
                publish=["LAI", "WLV", "TWLV"], 
                LAIEM=LAIEM, LASUM=LASUM, LAIEXP=LAIEXP, 
                LAIMAX=LAIMAX, LAI=LAI, WLV=WLV, DWLV=DWLV, TWLV=TWLV)
        
        self.rates = self.RateVariables(kiosk=self.kiosk,
                publish=["GRLV", "DRLV"])
    
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

        r.GRLV =  torch.tensor([0.75]).to(self.device) # k.ADMI * k.FL
        r.DSLV1 = s.WLV * (1. - k.RFTRA) * p.PERDL

        LAICR = 3.2 / p.KDIFTB(k.DVS)

        r.DSLV2 = s.WLV * torch.clamp(0.03*(s.LAI-LAICR)/LAICR, \
                                      torch.tensor([0.]).to(self.device), torch.tensor([0.03]).to(self.device))

        if "RF_FROST" in self.kiosk:
            r.DSLV3 = s.WLV * k.RF_FROST
        else:
            r.DSLV3 = 0.

        r.DSLV4 = s.WLV * p.RDRLV_NPK * (1.0 - k.NPKI)
        r.DSLV = torch.max(torch.max(r.DSLV1, r.DSLV2), r.DSLV3) + r.DSLV4

        DALV = 0.0
        for lv, lvage in zip(self.LV, self.LVAGE):
            if lvage > p.SPAN:
                DALV = DALV + lv
        r.DALV = DALV
        r.DRLV = torch.tensor([0.0001]).to(self.device) # torch.max(r.DSLV, r.DALV)

        r.FYSAGE = torch.max(torch.tensor([0.]).to(self.device), (drv.TEMP - p.TBASE) / (35. - p.TBASE))
        sla_npk_factor = torch.exp(-p.NSLA_NPK * (1.0 - k.NPKI))
        r.SLAT = p.SLATB(k.DVS) * sla_npk_factor

        
        if s.LAIEXP < 6.:
            DTEFF = torch.max(torch.tensor([0.]).to(self.device), drv.TEMP-p.TBASE)

            if k.DVS < 0.2 and s.LAI < 0.75:
                factor = k.NPKI * torch.exp(-p.NLAI_NPK * (1.0 - k.RFTRA))
            else:
                factor = 1.

            r.GLAIEX = s.LAIEXP * p.RGRLAI * DTEFF * factor
            
            r.GLASOL = r.GRLV * r.SLAT
            
            GLA = torch.min(r.GLAIEX, r.GLASOL)
            
            if r.GRLV > 0.:
                r.SLAT = GLA/r.GRLV
        
        self.rates._update_kiosk()

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates to new state
        """
        p = self.params
        r = self.rates
        s = self.states

        tLV = array('d', self.LV)
        tSLA = array('d', self.SLA)
        tLVAGE = array('d', self.LVAGE)
        tDRLV = r.DRLV

        for LVweigth in reversed(self.LV):
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

        s.LASUM = torch.tensor([0.05]).to(self.device) #torch.sum(torch.tensor([lv * sla for lv, sla in zip(tLV, tSLA)]).to(self.device)).unsqueeze(0)
        s.LAI = self._calc_LAI()
        s.LAIMAX = torch.max(s.LAI, s.LAIMAX)

        s.LAIEXP = s.LAIEXP + r.GLAIEX

        s.WLV = torch.tensor([15.00]).to(self.device) #torch.sum(torch.tensor(tLV).to(self.device))
        s.DWLV = s.DWLV + r.DRLV
        s.TWLV = s.WLV + s.DWLV

        self.LV = tLV
        self.SLA = tSLA
        self.LVAGE = tLVAGE

        self.states._update_kiosk()

    def reset(self, day:date):
        """Reset states and rates
        """

        p = self.params
        k = self.kiosk
        
        WLV = (p.TDWI * (1 - k.FR)) * k.FL
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

        self.LV = LV
        self.SLA = SLA
        self.LVAGE = LVAGE

        self.states = self.StateVariables(kiosk=self.kiosk, 
                publish=["LAI", "WLV", "TWLV"], 
                LAIEM=LAIEM, LASUM=LASUM, LAIEXP=LAIEXP, 
                LAIMAX=LAIMAX, LAI=LAI, WLV=WLV, DWLV=DWLV, TWLV=TWLV)
        
        self.rates = self.RateVariables(kiosk=self.kiosk,
                publish=["GRLV", "DRLV"])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.LAI
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
        return {"LV": self.LV,
                "SLA": self.SLA,
                "LVAGE": self.LVAGE}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)