"""Handles growth of leaf dynamics in the crop

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
from array import array
from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from model_engine.util import tensor_pop, tensor_appendleft
from model_engine.util import EPS

class WOFOST_Leaf_Dynamics_NPK_TensorBatch(BatchTensorModel):
    """Leaf dynamics for the WOFOST crop model including leaf response to
    NPK stress.
    """

    LV = Tensor(-99.)
    SLA = Tensor(-99.)
    LVAGE = Tensor(-99.)
    LVPOINTER = Tensor(-99.)

    class Parameters(ParamTemplate):
        RGRLAI = Tensor(-99.)
        SPAN = Tensor(-99.)
        TBASE = Tensor(-99.)
        PERDL = Tensor(-99.)
        TDWI = Tensor(-99.)
        SLATB = TensorBatchAfgenTrait()
        KDIFTB = TensorBatchAfgenTrait()
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

    def __init__(self, day:date, kiosk, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        p = self.params
        k = self.kiosk
        
        WLV = (p.TDWI * (1 - k.FR)) * k.FL
        DWLV = 0.
        TWLV = WLV + DWLV

        SLA = torch.zeros((self.num_models, 365)).to(self.device)
        LVAGE = torch.zeros((self.num_models, 365)).to(self.device)
        LV = torch.zeros((self.num_models, 365)).to(self.device)
        SLA[:,0] = p.SLATB(k.DVS)
        LV[:,0] = WLV

        LAIEM = LV[:,0] * SLA[:,0]
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
        self.LVPOINTER = torch.ones((self.num_models,)).to(self.device)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, 
                publish=["LAI", "WLV", "TWLV"], 
                LAIEM=LAIEM, LASUM=LASUM, LAIEXP=LAIEXP, 
                LAIMAX=LAIMAX, LAI=LAI, WLV=WLV, DWLV=DWLV, TWLV=TWLV)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk,
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

    def calc_rates(self, day:date, drv, _emerging:torch.tensor):
        """Calculate state rates
        """
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        r.GRLV = k.ADMI * k.FL
        r.DSLV1 = s.WLV * (1. - k.RFTRA) * p.PERDL

        LAICR = 3.2 / p.KDIFTB(k.DVS)

        r.DSLV2 = s.WLV * torch.clamp(0.03*(s.LAI-LAICR)/LAICR, \
                                      torch.tensor([0.]).to(self.device), torch.tensor([0.03]).to(self.device))

        if "RF_FROST" in self.kiosk:
            r.DSLV3 = s.WLV * k.RF_FROST
        else:
            r.DSLV3 = torch.zeros((self.num_models,)).to(self.device)

        r.DSLV4 = s.WLV * p.RDRLV_NPK * (1.0 - k.NPKI)
        r.DSLV = torch.max(torch.max(r.DSLV1, r.DSLV2), r.DSLV3) + r.DSLV4

        DALV = torch.where(self.LVAGE > p.SPAN.unsqueeze(1), self.LV, 0.0)
        r.DALV = torch.sum(DALV, dim=1)
        r.DRLV = torch.max(r.DSLV, r.DALV)

        r.FYSAGE = torch.max(torch.tensor([0.]).to(self.device), (drv.TEMP - p.TBASE) / (35. - p.TBASE))
        sla_npk_factor = torch.exp(-p.NSLA_NPK * (1.0 - k.NPKI))
        r.SLAT = p.SLATB(k.DVS) * sla_npk_factor


        factor = torch.where((k.DVS < 0.2) & (s.LAI < 0.75), k.RFTRA * torch.exp(-p.NLAI_NPK * (1.0 - k.NPKI)), 1.)
        DTEFF = torch.max(torch.tensor([0.]).to(self.device), drv.TEMP-p.TBASE)

        r.GLAIEX = torch.where(s.LAIEXP < 6., s.LAIEXP * p.RGRLAI * DTEFF * factor, 0.)
        r.GLASOL = torch.where(s.LAIEXP < 6., r.GRLV * r.SLAT, 0.)

        r.SLAT = torch.where((s.LAIEXP < 6.) & (r.GRLV > 0.), torch.min(r.GLAIEX, r.GLASOL) / (r.GRLV), r.SLAT) #TODO might need to add + EPS back
        # Evaluate to 0 when _emerging
        r.GRLV = torch.where(_emerging, 0.0, r.GRLV)
        r.DSLV1 = torch.where(_emerging, 0.0, r.DSLV1)
        r.DSLV2 = torch.where(_emerging, 0.0, r.DSLV2)
        r.DSLV3 = torch.where(_emerging, 0.0, r.DSLV3)
        r.DSLV4 = torch.where(_emerging, 0.0, r.DSLV4)
        r.DSLV = torch.where(_emerging, 0.0, r.DSLV)
        r.DALV = torch.where(_emerging, 0.0, r.DALV)
        r.DRLV = torch.where(_emerging, 0.0, r.DRLV)
        r.SLAT = torch.where(_emerging, 0.0, r.SLAT)
        r.FYSAGE = torch.where(_emerging, 0.0, r.FYSAGE)
        r.GLAIEX = torch.where(_emerging, 0.0, r.GLAIEX)
        r.GLASOL = torch.where(_emerging, 0.0, r.GLASOL)

        self.rates._update_kiosk()

    def process_LV(self, tLV, tLVAGE, tSLA, tDRLV, tLVPOINTER):
        """
        Process tLV, tLVAGE, tSLA tensors based on demand tDRLV (shape: batch_size,).
        tLV, tLVAGE, tSLA: tensors of shape (batch_size, history_length).
        Returns updated tensors and tDRLV.
        """
        batch_size, history_length = tLV.shape

        tLV_new = tLV.clone()
        tLVAGE_new = tLVAGE.clone()
        tSLA_new = tSLA.clone()
        tDRLV_new = tDRLV.clone()

        # Process from end to start (right to left)
        for i in reversed(range(history_length)):
            remaining = tDRLV_new > 0

            if not remaining.any():
                break
            LVweight = tLV_new[:, i]

            # Case 1: Full removal (demand >= LVweight)
            full_remove = (tDRLV_new >= LVweight) & remaining
            tDRLV_new = torch.where(full_remove, tDRLV_new - LVweight, tDRLV_new) # Not sure about this line 
            tLV_new[:, i] = torch.where(full_remove, torch.tensor(0., device=tLV.device), tLV_new[:, i])
            tLVAGE_new[:, i] = torch.where(full_remove, torch.tensor(0., device=tLVAGE.device), tLVAGE_new[:, i])
            tSLA_new[:, i] = torch.where(full_remove, torch.tensor(0., device=tSLA_new.device), tSLA_new[:, i])
            tLVPOINTER = torch.where(full_remove & (i < tLVPOINTER), tLVPOINTER - 1, tLVPOINTER)

            # Case 2: Partial removal (demand < LVweight)
            partial_remove = (tDRLV_new < LVweight) & remaining
            tLV_new[:, i] = torch.where(partial_remove, LVweight - tDRLV_new, tLV_new[:, i])
            tDRLV_new = torch.where(partial_remove, torch.tensor(0., device=tDRLV.device), tDRLV_new)

        return tLV_new, tLVAGE_new, tSLA_new, tLVPOINTER

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates to new state
        """

        p = self.params
        r = self.rates
        s = self.states

        tLV = self.LV
        tSLA = self.SLA
        tLVAGE = self.LVAGE
        tDRLV = r.DRLV
        tLVPOINTER = self.LVPOINTER

        tLV, tLVAGE, tSLA, tLVPOINTER = self.process_LV(tLV, tLVAGE, tSLA, tDRLV, tLVPOINTER)

        updates = torch.zeros_like(tLVAGE).to(self.device)
        #updates[torch.arange(self.num_models, device=self.device), :tLVPOINTER.long()+1] = r.FYSAGE
        #tLVAGE = tLVAGE + updates

        mask = torch.arange(tLV.shape[1], device=self.device).unsqueeze(0) < (tLVPOINTER.unsqueeze(1) + 1)
        updates[mask] = r.FYSAGE.unsqueeze(1).expand_as(updates)[mask]
        tLVAGE = tLVAGE + updates

        tLV = tensor_appendleft(tLV, r.GRLV)
        tSLA = tensor_appendleft(tSLA, r.SLAT)
        tLVAGE = tensor_appendleft(tLVAGE, torch.zeros((self.num_models,)).to(self.device))
        tLVPOINTER = tLVPOINTER + 1

        s.LASUM = torch.sum(tLV * tSLA, dim=1)
        s.LAI = self._calc_LAI()
        s.LAIMAX = torch.max(s.LAI, s.LAIMAX)
        s.LAIEXP = s.LAIEXP + r.GLAIEX

        s.WLV = torch.sum(tLV, dim=1)
        s.DWLV = s.DWLV + r.DRLV
        s.TWLV = s.WLV + s.DWLV

        self.LV = tLV
        self.SLA = tSLA
        self.LVAGE = tLVAGE
        self.LVPOINTER = tLVPOINTER

        self.states._update_kiosk()

    def reset(self, day:date):
        """Reset states and rates
        """

        p = self.params
        k = self.kiosk
        
        WLV = (p.TDWI * (1 - k.FR)) * k.FL
        DWLV = 0.
        TWLV = WLV + DWLV

        SLA = torch.zeros((self.num_models, 365)).to(self.device)
        LVAGE = torch.zeros((self.num_models, 365)).to(self.device)
        LV = torch.zeros((self.num_models, 365)).to(self.device)
        SLA[:,0] = p.SLATB(k.DVS)
        LV[:,0] = WLV

        LAIEM = LV[:,0] * SLA[:,0]
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
        self.LVPOINTER = torch.ones((self.num_models,)).to(self.device)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, 
                publish=["LAI", "WLV", "TWLV"], 
                LAIEM=LAIEM, LASUM=LASUM, LAIEXP=LAIEXP, 
                LAIMAX=LAIMAX, LAI=LAI, WLV=WLV, DWLV=DWLV, TWLV=TWLV)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk,
                publish=["GRLV", "DRLV"])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.LAI
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
        return {"LV": self.LV,
                "SLA": self.SLA,
                "LVAGE": self.LVAGE}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)