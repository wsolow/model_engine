"""
Performs bookkeeping for how NPK is translocated around roots, leaves, and stems

Written by: Will Solow, 2025
"""

from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class NPK_Translocation_TensorBatch(BatchTensorModel):
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

    def __init__(self, day:date, kiosk, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk,
            NTRANSLOCATABLELV=0., NTRANSLOCATABLEST=0., NTRANSLOCATABLERT=0., PTRANSLOCATABLELV=0., PTRANSLOCATABLEST=0.,
            PTRANSLOCATABLERT=0., KTRANSLOCATABLELV=0., KTRANSLOCATABLEST=0. ,KTRANSLOCATABLERT=0.,
            NTRANSLOCATABLE=0., PTRANSLOCATABLE=0., KTRANSLOCATABLE=0.,
            publish=["NTRANSLOCATABLE", "PTRANSLOCATABLE", "KTRANSLOCATABLE"])
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, 
                                        publish=["RNTRANSLOCATIONLV", "RNTRANSLOCATIONST", "RNTRANSLOCATIONRT",
                                                 "RPTRANSLOCATIONLV", "RPTRANSLOCATIONST", "RPTRANSLOCATIONRT",
                                                 "RKTRANSLOCATIONLV", "RKTRANSLOCATIONST", "RKTRANSLOCATIONRT"])
        
        self.max_tensor = torch.tensor([0.]).to(self.device)

    def calc_rates(self, day:date, drv, _emerging):
        """Calculate rates for integration
        """
        r = self.rates
        s = self.states
        k = self.kiosk

        r.RNTRANSLOCATIONLV = torch.where(s.NTRANSLOCATABLE > 0, k.RNUPTAKESO * s.NTRANSLOCATABLELV / s.NTRANSLOCATABLE, 0.0)
        r.RNTRANSLOCATIONST = torch.where(s.NTRANSLOCATABLE > 0, k.RNUPTAKESO * s.NTRANSLOCATABLEST / s.NTRANSLOCATABLE, 0.0)
        r.RNTRANSLOCATIONRT = torch.where(s.NTRANSLOCATABLE > 0, k.RNUPTAKESO * s.NTRANSLOCATABLERT / s.NTRANSLOCATABLE, 0.0)

        r.RPTRANSLOCATIONLV = torch.where(s.PTRANSLOCATABLE > 0, k.RPUPTAKESO * s.PTRANSLOCATABLELV / s.PTRANSLOCATABLE, 0.0)
        r.RPTRANSLOCATIONST = torch.where(s.PTRANSLOCATABLE > 0, k.RPUPTAKESO * s.PTRANSLOCATABLEST / s.PTRANSLOCATABLE, 0.0)
        r.RPTRANSLOCATIONRT = torch.where(s.PTRANSLOCATABLE > 0, k.RPUPTAKESO * s.PTRANSLOCATABLERT / s.PTRANSLOCATABLE, 0.0)

        r.RKTRANSLOCATIONLV = torch.where(s.KTRANSLOCATABLE > 0, k.RKUPTAKESO * s.KTRANSLOCATABLELV / s.KTRANSLOCATABLE, 0.0)
        r.RKTRANSLOCATIONST = torch.where(s.KTRANSLOCATABLE > 0, k.RKUPTAKESO * s.KTRANSLOCATABLEST / s.KTRANSLOCATABLE, 0.0)
        r.RKTRANSLOCATIONRT = torch.where(s.KTRANSLOCATABLE > 0, k.RKUPTAKESO * s.KTRANSLOCATABLERT / s.KTRANSLOCATABLE, 0.0)

        r.RNTRANSLOCATIONLV = torch.where(_emerging, 0.0, r.RNTRANSLOCATIONLV)
        r.RNTRANSLOCATIONST = torch.where(_emerging, 0.0, r.RNTRANSLOCATIONST)  
        r.RNTRANSLOCATIONRT = torch.where(_emerging, 0.0, r.RNTRANSLOCATIONRT)

        r.RPTRANSLOCATIONLV = torch.where(_emerging, 0.0, r.RPTRANSLOCATIONLV)
        r.RPTRANSLOCATIONST = torch.where(_emerging, 0.0, r.RPTRANSLOCATIONST)
        r.RPTRANSLOCATIONRT = torch.where(_emerging, 0.0, r.RPTRANSLOCATIONRT)  

        r.RKTRANSLOCATIONLV = torch.where(_emerging, 0.0, r.RKTRANSLOCATIONLV)
        r.RKTRANSLOCATIONST = torch.where(_emerging, 0.0, r.RKTRANSLOCATIONST)
        r.RKTRANSLOCATIONRT = torch.where(_emerging, 0.0, r.RKTRANSLOCATIONRT)

        self.rates._update_kiosk()

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        p = self.params
        s = self.states
        k = self.kiosk
        
        s.NTRANSLOCATABLELV = torch.max(self.max_tensor, k.NAMOUNTLV - k.WLV * p.NRESIDLV)
        s.NTRANSLOCATABLEST = torch.max(self.max_tensor, k.NAMOUNTST - k.WST * p.NRESIDST)
        s.NTRANSLOCATABLERT = torch.max(self.max_tensor, k.NAMOUNTRT - k.WRT * p.NRESIDRT)

        s.PTRANSLOCATABLELV = torch.max(self.max_tensor, k.PAMOUNTLV - k.WLV * p.PRESIDLV)
        s.PTRANSLOCATABLEST = torch.max(self.max_tensor, k.PAMOUNTST - k.WST * p.PRESIDST)
        s.PTRANSLOCATABLERT = torch.max(self.max_tensor, k.PAMOUNTRT - k.WRT * p.PRESIDRT)

        s.KTRANSLOCATABLELV = torch.max(self.max_tensor, k.KAMOUNTLV - k.WLV * p.KRESIDLV)
        s.KTRANSLOCATABLEST = torch.max(self.max_tensor, k.KAMOUNTST - k.WST * p.KRESIDST)
        s.KTRANSLOCATABLERT = torch.max(self.max_tensor, k.KAMOUNTRT - k.WRT * p.KRESIDRT)

        s.NTRANSLOCATABLE = torch.where(k.DVS > p.DVS_NPK_TRANSL, s.NTRANSLOCATABLELV + s.NTRANSLOCATABLEST + s.NTRANSLOCATABLERT, 0.0)
        s.PTRANSLOCATABLE = torch.where(k.DVS > p.DVS_NPK_TRANSL, s.PTRANSLOCATABLELV + s.PTRANSLOCATABLEST + s.PTRANSLOCATABLERT, 0.0)
        s.KTRANSLOCATABLE = torch.where(k.DVS > p.DVS_NPK_TRANSL, s.KTRANSLOCATABLELV + s.KTRANSLOCATABLEST + s.KTRANSLOCATABLERT, 0.0)
        
        self.states._update_kiosk()

    def reset(self, day:date):
        """Reset states and rates
        """ 

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk,
            NTRANSLOCATABLELV=0., NTRANSLOCATABLEST=0., NTRANSLOCATABLERT=0., PTRANSLOCATABLELV=0., PTRANSLOCATABLEST=0.,
            PTRANSLOCATABLERT=0., KTRANSLOCATABLELV=0., KTRANSLOCATABLEST=0. ,KTRANSLOCATABLERT=0.,
            NTRANSLOCATABLE=0., PTRANSLOCATABLE=0., KTRANSLOCATABLE=0.,
            publish=["NTRANSLOCATABLE", "PTRANSLOCATABLE", "KTRANSLOCATABLE"])
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=[])
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.NTRANSLOCATABLE
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[:,i] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[:,i] = getattr(self.rates,v)
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