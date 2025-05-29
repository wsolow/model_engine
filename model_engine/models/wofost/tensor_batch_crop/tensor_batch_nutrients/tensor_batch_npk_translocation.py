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
        
        if k.DVS > p.DVS_NPK_TRANSL:
            s.NTRANSLOCATABLE = s.NTRANSLOCATABLELV + s.NTRANSLOCATABLEST + s.NTRANSLOCATABLERT
            s.PTRANSLOCATABLE = s.PTRANSLOCATABLELV + s.PTRANSLOCATABLEST + s.PTRANSLOCATABLERT
            s.KTRANSLOCATABLE = s.KTRANSLOCATABLELV + s.KTRANSLOCATABLEST + s.KTRANSLOCATABLERT
        else:
            s.NTRANSLOCATABLE = s.PTRANSLOCATABLE = s.KTRANSLOCATABLE = 0

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