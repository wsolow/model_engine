"""Calculates NPK Demand for the crop and corresponding uptake from soil

Written by: Will Solow, 2025
"""
from datetime import date
import torch

from collections import namedtuple
from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

MaxNutrientConcentrations = namedtuple("MaxNutrientConcentrations",
                                       ["NMAXLV", "PMAXLV", "KMAXLV",
                                        "NMAXST", "PMAXST", "KMAXST",
                                        "NMAXRT", "PMAXRT", "KMAXRT",
                                        "NMAXSO", "PMAXSO", "KMAXSO"])

class NPK_Demand_Uptake_Tensor(TensorModel):
    """Calculates the crop N/P/K demand and its uptake from the soil.
    """
    class Parameters(ParamTemplate):
        NMAXLV_TB = TensorAfgenTrait()  
        PMAXLV_TB = TensorAfgenTrait()  
        KMAXLV_TB = TensorAfgenTrait()  
        
        NMAXRT_FR = Tensor(-99.)  
        PMAXRT_FR = Tensor(-99.)  
        KMAXRT_FR = Tensor(-99.)  

        NMAXST_FR = Tensor(-99.)  
        PMAXST_FR = Tensor(-99.)  
        KMAXST_FR = Tensor(-99.)  
        
        NMAXSO = Tensor(-99.)  
        PMAXSO = Tensor(-99.)  
        KMAXSO = Tensor(-99.)  
        
        TCNT = Tensor(-99.)  
        TCPT = Tensor(-99.)  
        TCKT = Tensor(-99.)  

        NFIX_FR = Tensor(-99.)  
        RNUPTAKEMAX = Tensor()  
        RPUPTAKEMAX = Tensor()  
        RKUPTAKEMAX = Tensor()  

        DVS_NPK_STOP = Tensor(-99.)

    class RateVariables(RatesTemplate):
        RNUPTAKELV = Tensor(-99.)  
        RNUPTAKEST = Tensor(-99.)
        RNUPTAKERT = Tensor(-99.)
        RNUPTAKESO = Tensor(-99.)

        RPUPTAKELV = Tensor(-99.)  
        RPUPTAKEST = Tensor(-99.)
        RPUPTAKERT = Tensor(-99.)
        RPUPTAKESO = Tensor(-99.)

        RKUPTAKELV = Tensor(-99.)  
        RKUPTAKEST = Tensor(-99.)
        RKUPTAKERT = Tensor(-99.)
        RKUPTAKESO = Tensor(-99.)

        RNUPTAKE = Tensor(-99.)  
        RPUPTAKE = Tensor(-99.)  
        RKUPTAKE = Tensor(-99.)  
        RNFIXATION = Tensor(-99.)  

        NDEMANDLV = Tensor(-99.)  
        NDEMANDST = Tensor(-99.)
        NDEMANDRT = Tensor(-99.)
        NDEMANDSO = Tensor(-99.)

        PDEMANDLV = Tensor(-99.)  
        PDEMANDST = Tensor(-99.)
        PDEMANDRT = Tensor(-99.)
        PDEMANDSO = Tensor(-99.)

        KDEMANDLV = Tensor(-99.)  
        KDEMANDST = Tensor(-99.)
        KDEMANDRT = Tensor(-99.)
        KDEMANDSO = Tensor(-99.)

        NDEMAND = Tensor()  
        PDEMAND = Tensor()
        KDEMAND = Tensor()

    def __init__(self, day:date, kiosk, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        self.rates = self.RateVariables(kiosk=self.kiosk,
            publish=["RNUPTAKE", "RPUPTAKE", "RKUPTAKE", "RNFIXATION", "RKFIXATION", "RPFIXATION",
                     "RNUPTAKELV", "RNUPTAKEST", "RNUPTAKERT", "RNUPTAKESO",
                     "RPUPTAKELV", "RPUPTAKEST", "RPUPTAKERT", "RPUPTAKESO",
                     "RKUPTAKELV", "RKUPTAKEST", "RKUPTAKERT", "RKUPTAKESO"])
        self.zero_tens = torch.tensor([0.]).to(self.device)
        
    def calc_rates(self, day:date, drv):
        """Calculate rates
        """
        r = self.rates
        p = self.params
        k = self.kiosk

        delt = 1.0
        mc = self._compute_NPK_max_concentrations()

        r.NDEMANDLV = torch.max(mc.NMAXLV * k.WLV - k.NAMOUNTLV, self.zero_tens) + torch.max(k.GRLV * mc.NMAXLV, self.zero_tens) * delt
        r.NDEMANDST = torch.max(mc.NMAXST * k.WST - k.NAMOUNTST, self.zero_tens) + torch.max(k.GRST * mc.NMAXST, self.zero_tens) * delt
        r.NDEMANDRT = torch.max(mc.NMAXRT * k.WRT - k.NAMOUNTRT, self.zero_tens) + torch.max(k.GRRT * mc.NMAXRT, self.zero_tens) * delt
        r.NDEMANDSO = torch.max(mc.NMAXSO * k.WSO - k.NAMOUNTSO, self.zero_tens)

        r.PDEMANDLV = torch.max(mc.PMAXLV * k.WLV - k.PAMOUNTLV, self.zero_tens) + torch.max(k.GRLV * mc.PMAXLV, self.zero_tens) * delt
        r.PDEMANDST = torch.max(mc.PMAXST * k.WST - k.PAMOUNTST, self.zero_tens) + torch.max(k.GRST * mc.PMAXST, self.zero_tens) * delt
        r.PDEMANDRT = torch.max(mc.PMAXRT * k.WRT - k.PAMOUNTRT, self.zero_tens) + torch.max(k.GRRT * mc.PMAXRT, self.zero_tens) * delt
        r.PDEMANDSO = torch.max(mc.PMAXSO * k.WSO - k.PAMOUNTSO, self.zero_tens)

        r.KDEMANDLV = torch.max(mc.KMAXLV * k.WLV - k.KAMOUNTLV, self.zero_tens) + torch.max(k.GRLV * mc.KMAXLV, self.zero_tens) * delt
        r.KDEMANDST = torch.max(mc.KMAXST * k.WST - k.KAMOUNTST, self.zero_tens) + torch.max(k.GRST * mc.KMAXST, self.zero_tens) * delt
        r.KDEMANDRT = torch.max(mc.KMAXRT * k.WRT - k.KAMOUNTRT, self.zero_tens) + torch.max(k.GRRT * mc.KMAXRT, self.zero_tens) * delt
        r.KDEMANDSO = torch.max(mc.KMAXSO * k.WSO - k.KAMOUNTSO, self.zero_tens)

        r.NDEMAND = r.NDEMANDLV + r.NDEMANDST + r.NDEMANDRT
        r.PDEMAND = r.PDEMANDLV + r.PDEMANDST + r.PDEMANDRT
        r.KDEMAND = r.KDEMANDLV + r.KDEMANDST + r.KDEMANDRT

        r.RNUPTAKESO = torch.min(r.NDEMANDSO, k.NTRANSLOCATABLE)/p.TCNT
        r.RPUPTAKESO = torch.min(r.PDEMANDSO, k.PTRANSLOCATABLE)/p.TCPT
        r.RKUPTAKESO = torch.min(r.KDEMANDSO, k.KTRANSLOCATABLE)/p.TCKT

        if k.RFTRA > 0.01:
            NutrientLIMIT = 1.0
        else:
            NutrientLIMIT = 0.

        r.RNFIXATION = (torch.max(self.zero_tens, p.NFIX_FR * r.NDEMAND) * NutrientLIMIT)

        if k.DVS < p.DVS_NPK_STOP:
            r.RNUPTAKE = (torch.max(self.zero_tens, torch.min(r.NDEMAND - r.RNFIXATION, torch.min(k.NAVAIL, p.RNUPTAKEMAX))) * NutrientLIMIT)
            r.RPUPTAKE = (torch.max(self.zero_tens, torch.min(r.PDEMAND, torch.min(k.PAVAIL, p.RPUPTAKEMAX))) * NutrientLIMIT)
            r.RKUPTAKE = (torch.max(self.zero_tens, torch.min(r.KDEMAND, torch.min(k.KAVAIL, p.RKUPTAKEMAX))) * NutrientLIMIT)
        else:
            r.RNUPTAKE = r.RPUPTAKE = r.RKUPTAKE = 0

        if r.NDEMAND == 0.:
            r.RNUPTAKELV = r.RNUPTAKEST = r.RNUPTAKERT = 0.
        else:
            r.RNUPTAKELV = (r.NDEMANDLV / r.NDEMAND) * (r.RNUPTAKE + r.RNFIXATION)
            r.RNUPTAKEST = (r.NDEMANDST / r.NDEMAND) * (r.RNUPTAKE + r.RNFIXATION)
            r.RNUPTAKERT = (r.NDEMANDRT / r.NDEMAND) * (r.RNUPTAKE + r.RNFIXATION)

        if r.PDEMAND == 0.:
            r.RPUPTAKELV = r.RPUPTAKEST = r.RPUPTAKERT = 0.
        else:
            r.RPUPTAKELV = (r.PDEMANDLV / r.PDEMAND) * r.RPUPTAKE
            r.RPUPTAKEST = (r.PDEMANDST / r.PDEMAND) * r.RPUPTAKE
            r.RPUPTAKERT = (r.PDEMANDRT / r.PDEMAND) * r.RPUPTAKE

        if r.KDEMAND == 0.:
            r.RKUPTAKELV = r.RKUPTAKEST = r.RKUPTAKERT = 0.
        else:
            r.RKUPTAKELV = (r.KDEMANDLV / r.KDEMAND) * r.RKUPTAKE
            r.RKUPTAKEST = (r.KDEMANDST / r.KDEMAND) * r.RKUPTAKE
            r.RKUPTAKERT = (r.KDEMANDRT / r.KDEMAND) * r.RKUPTAKE

        self.rates._update_kiosk()
    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate states - no states to integrate in NPK Demand Uptake
        """
        pass

    def _compute_NPK_max_concentrations(self):
        """Computes the maximum N/P/K concentrations in leaves, stems, roots and storage organs.
        
        Note that max concentrations are first derived from the dilution curve for leaves. 
        Maximum concentrations for stems and roots are computed as a fraction of the 
        concentration for leaves. Maximum concentration for storage organs is directly taken from
        the parameters N/P/KMAXSO.
        """

        p = self.params
        k = self.kiosk
        NMAXLV = p.NMAXLV_TB(k.DVS)
        PMAXLV = p.PMAXLV_TB(k.DVS)
        KMAXLV = p.KMAXLV_TB(k.DVS)
        max_NPK_conc = MaxNutrientConcentrations(
            
            NMAXLV = NMAXLV,
            PMAXLV = PMAXLV,
            KMAXLV = KMAXLV,
            
            NMAXST = (p.NMAXST_FR * NMAXLV),
            NMAXRT = p.NMAXRT_FR * NMAXLV,
            NMAXSO = p.NMAXSO,

            PMAXST = p.PMAXST_FR * PMAXLV,
            PMAXRT = p.PMAXRT_FR * PMAXLV,
            PMAXSO = p.PMAXSO,

            KMAXST = p.KMAXST_FR * KMAXLV,
            KMAXRT = p.KMAXRT_FR * KMAXLV,
            KMAXSO = p.KMAXSO
        )

        return max_NPK_conc

    def reset(self, day:date):
        """Reset states and rates
        """
        self.rates = self.RateVariables(kiosk=self.kiosk,
            publish=["RNUPTAKE", "RPUPTAKE", "RKUPTAKE", "RNFIXATION", "RKFIXATION", "RPFIXATION",
                     "RNUPTAKELV", "RNUPTAKEST", "RNUPTAKERT", "RNUPTAKESO",
                     "RPUPTAKELV", "RPUPTAKEST", "RPUPTAKERT", "RPUPTAKESO",
                     "RKUPTAKELV", "RKUPTAKEST", "RKUPTAKERT", "RKUPTAKESO"])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.rates.NDEMAND
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.rates.trait_names():
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