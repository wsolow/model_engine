"""
tensor_grape_coldhardiness.py
Implementation of Feguson Model for Grape Cold Hardiness

Written by Will Solow, 2025
"""
import datetime
import torch

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
     
class Grape_ColdHardiness_Tensor(TensorModel):

    _STAGE_VAL = {"endodorm":0, "ecodorm":1}
    _STAGE  = "endodorm"
    _HC_YESTERDAY = Tensor(-99.)

    class Parameters(ParamTemplate):
        HCINIT     = Tensor(-99.) # Initial Cold Hardiness
        HCMIN      = Tensor(-99.) # Minimum cold hardiness (negative)
        HCMAX      = Tensor(-99.) # Maximum cold hardiness (negative)
        TENDO      = Tensor(-99.) # Endodorm temp
        TECO       = Tensor(-99.) # Ecodorm temp
        ENACCLIM   = Tensor(-99.) # Endo rate of acclimation
        ECACCLIM   = Tensor(-99.) # Eco rate of acclimation
        ENDEACCLIM = Tensor(-99.) # Endo rate of deacclimation
        ECDEACCLIM = Tensor(-99.) # Eco rate of deacclimation
        THETA      = Tensor(-99.) # Theta param for acclimation
        ECOBOUND   = Tensor(-99.) # Temperature threshold for onset of ecodormancy
        LTE10M     = Tensor(-99.) # Regression coefficient for LTE10
        LTE10B     = Tensor(-99.) # Regression coefficient for LTE10
        LTE90M     = Tensor(-99.) # Regression coefficient for LTE90
        LTE90B     = Tensor(-99.) # Regression coefficient for LTE90

    class RateVariables(RatesTemplate):
        DCU       = Tensor(-99.) # Daily heat accumulation
        DHR    = Tensor(-99.) # Daily heating rate
        DCR    = Tensor(-99.) # Daily chilling rate
        DACC   = Tensor(-99.) # Deacclimation rate
        ACC    = Tensor(-99.) # Acclimation rate
        HCR    = Tensor(-99.) # Change in acclimation

    class StateVariables(StatesTemplate):
        CSUM      = Tensor(-99.) # Daily temperature sum for phenology
        DHSUM     = Tensor(-99.) # Daily heating sum
        DCSUM     = Tensor(-99.) # Daily chilling sum
        HC        = Tensor(-99.) # Cold hardiness
        PREDBB    = Tensor(-99.) # Predicted bud break
        LTE50     = Tensor(-99.) # Predicted LTE50 for cold hardiness
        LTE10     = Tensor(-99.) # Predicted LTE10 for cold hardiness
        LTE90     = Tensor(-99.) # Predicted LTE90 for cold hardiness
             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        # Define initial states
        p = self.params
        self._STAGE = "endodorm"
        LTE10 = p.HCINIT * p.LTE10M + p.LTE10B
        LTE90 = p.HCINIT * p.LTE90M + p.LTE90B
        self.states = self.StateVariables(DHSUM=0., DCSUM=0.,HC=p.HCINIT.detach(),
                                          PREDBB=0., LTE50=p.HCINIT.detach(), CSUM=0.,
                                          LTE10=LTE10.detach(), LTE90=LTE90.detach())
        
        self.rates = self.RateVariables()
        self.min_tensor = torch.tensor([0.]).to(self.device)
        self.base_tensor = torch.tensor([10.]).to(self.device)
        self._HC_YESTERDAY = p.HCINIT.detach().clone()

    def calc_rates(self, day, drv):
        """
        Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        r.DCU = 0.
        r.DHR = 0.
        r.DCR = 0.
        r.DACC = 0.
        r.ACC = 0.
        r.HCR = 0.

        r.DCU = torch.min(self.min_tensor, drv.TEMP - self.base_tensor)
        if self._STAGE == "endodorm":
            r.DHR = torch.max(self.min_tensor, drv.TEMP-p.TENDO)
            r.DCR = torch.min(self.min_tensor, drv.TEMP-p.TENDO)
            if s.DCSUM != 0:
                r.DACC = r.DHR * p.ENDEACCLIM * (1 - ((self._HC_YESTERDAY-p.HCMAX) / (p.HCMIN-p.HCMAX)))
            else:
                r.DACC = 0
            r.ACC = r.DCR * p.ENACCLIM * (1-((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN-p.HCMAX)))
            r.HCR = r.DACC + r.ACC
        
        elif self._STAGE == "ecodorm":
            r.DHR = torch.max(self.min_tensor, drv.TEMP-p.TECO)
            r.DCR = torch.min(self.min_tensor, drv.TEMP-p.TECO)
            
            if s.DCSUM != 0:
                r.DACC = r.DHR * p.ECDEACCLIM * (1 - (((self._HC_YESTERDAY-p.HCMAX) / (p.HCMIN-p.HCMAX)) ** p.THETA))
            else:
                r.DACC = 0
            r.ACC = r.DCR * p.ECACCLIM * (1-((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN-p.HCMAX)))
            
            r.HCR = r.DACC + r.ACC

        else: 
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self._STAGE)
        
    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """
        p = self.params
        r = self.rates
        s = self.states

        s.CSUM = s.CSUM + r.DCU 

        s.HC = torch.clamp(p.HCMAX, p.HCMIN, s.HC+r.HCR)
        self._HC_YESTERDAY = s.HC

        s.DCSUM = s.DCSUM + r.DCR 
        s.LTE50 = s.HC * 100
        s.LTE10 = s.LTE50 * p.LTE10M + p.LTE10B
        s.LTE90 = s.LTE50 * p.LTE90M + p.LTE90B

        # Use HCMIN to determine if vinifera or labrusca
        if p.HCMIN == -1.2:   
            if self._HC_YESTERDAY < -2.2:
                if s.HC >= -2.2:
                    s.PREDBB = s.HC
        if p.HCMIN == -2.5:  
            if self._HC_YESTERDAY < -6.4:
                if s.HC >= -6.4:
                    s.PREDBB = s.HC

        # Check if a new stage is reached
        if self._STAGE == "endodorm":
            if s.CSUM <= p.ECOBOUND:
                self._STAGE = "ecodorm"

        elif self._STAGE == "ecodorm":
            pass
                        
        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self._STAGE)     

    def get_output(self, vars:list=None):
        """
        Return the LTE50 for cold hardiness
        """
        if vars is None:
            return torch.unsqueeze(self.states.LTE50, -1)
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.rates,v)
            return output_vars

    def reset(self, day:datetime.date):
        """
        Reset the model
        """

        p = self.params
        self._STAGE = "endodorm"
        LTE10 = p.HCINIT * p.LTE10M + p.LTE10B
        LTE90 = p.HCINIT * p.LTE90M + p.LTE90B
        self.states = self.StateVariables(DHSUM=0., DCSUM=0.,HC=p.HCINIT.detach(),
                                          PREDBB=0., LTE50=p.HCINIT.detach(), CSUM=0.,
                                          LTE10=LTE10.detach(), LTE90=LTE90.detach())
        self.rates = self.RateVariables()
        self._HC_YESTERDAY = p.HCINIT.detach().clone()

    def get_extra_states(self):
        """
        Get extra states
        """
        return {"_STAGE":self._STAGE, "_HC_YESTERDAY":self._HC_YESTERDAY}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        if k == "THETA":
            setattr(self.params, k, torch.floor(v).detach()+(v-v.detach()))
        else:
            setattr(self.params, k, v)