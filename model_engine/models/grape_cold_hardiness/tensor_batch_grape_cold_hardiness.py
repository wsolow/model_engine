"""
tensor_batch_grape_coldhardiness.py
Implementation of Feguson Model for Grape Cold Hardiness supporting batches

Written by Will Solow, 2025
"""

import datetime
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class Grape_ColdHardiness_TensorBatch(BatchTensorModel):

    _STAGE_VAL = {"endodorm":0, "ecodorm":1}
    _STAGE  = NDArray(["endodorm"])
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
        ECOBOUND     = Tensor(-99.) # Temperature threshold for onset of ecodormancy
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
             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        """
        :param day: start date of the simulation
        :param parvalues: providing parameters as key/value pairs
        """
        super().__init__(day, kiosk, parvalues, device, num_models=num_models)

        # Define initial states
        p=self.params
        self.num_models = num_models
        self.num_stages = len(self._STAGE_VAL)
        self.stages = list(self._STAGE_VAL.keys())
        self._STAGE = ["endodorm" for _ in range(self.num_models)]
        LTE10 = p.HCINIT * p.LTE10M + p.LTE10B
        LTE90 = p.HCINIT * p.LTE90M + p.LTE90B
        self.states = self.StateVariables(num_models=self.num_models, DHSUM=0., DCSUM=0.,HC=p.HCINIT[0].detach().cpu(),
                                          PREDBB=0., LTE50=p.HCINIT[0].detach().cpu(), CSUM=0.,
                                          LTE10=LTE10[0].detach().cpu(), LTE90=LTE90.detach().cpu())
        
        self.rates = self.RateVariables(num_models=self.num_models)
        self.min_tensor = torch.tensor([0.]).to(self.device)
        self.base_tensor = torch.tensor([10.]).to(self.device)
        self._HC_YESTERDAY = p.HCINIT[0].detach().clone()

    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        r.DCU = torch.zeros(size=(self.num_models,))
        r.DHR = torch.zeros(size=(self.num_models,))
        r.DCR = torch.zeros(size=(self.num_models,))
        r.DACC = torch.zeros(size=(self.num_models,))
        r.ACC = torch.zeros(size=(self.num_models,))
        r.HCR = torch.zeros(size=(self.num_models,))
        r.DCU = torch.min(self.min_tensor, drv.TEMP - self.base_tensor )

        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device) # create masks
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)]) # one hot encoding matrix
        self._endodorm, self._ecodorm = stage_masks # upack for readability

        r.DHR = torch.where(self._endodorm, torch.max(self.min_tensor, drv.TEMP-p.TENDO), torch.where(
            self._ecodorm, torch.max(self.min_tensor, drv.TEMP-p.TECO), torch.ones_like(p.TECO)
            ))
        r.DCR = torch.where(self._endodorm, torch.min(self.min_tensor, drv.TEMP-p.TENDO), torch.where(
            self._ecodorm, torch.min(self.min_tensor, drv.TEMP-p.TECO), torch.ones_like(p.TECO)
        ))

        r.DACC = torch.where(self._endodorm, torch.where(s.DCSUM != 0, 
                        r.DHR * p.ENDEACCLIM * (1 - ((self._HC_YESTERDAY-p.HCMAX) / (p.HCMIN-p.HCMAX).clamp(min=1e-6))), 0),
                        torch.where(self._ecodorm, torch.where(s.DCSUM != 0, 
                        r.DHR * p.ECDEACCLIM * (1 - ((self._HC_YESTERDAY-p.HCMAX) / (p.HCMIN-p.HCMAX).clamp(min=1e-6)) ** p.THETA), 0), 
                        torch.ones_like(p.HCMAX)  
        ))

        r.ACC = torch.where(self._endodorm, r.DCR * p.ENACCLIM * (1-((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN-p.HCMAX).clamp(min=1e-6))), 
                            torch.where(self._ecodorm, r.DCR * p.ECACCLIM * (1-((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN-p.HCMAX).clamp(min=1e-6))),
                                        torch.ones_like(p.HCMAX)))

        r.HCR = r.DACC + r.ACC

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states

        # Integrate phenologic states
        s.CSUM = s.CSUM + r.DCU 
        s.HC = torch.clamp(p.HCMAX, p.HCMIN, s.HC+r.HCR)
        self._HC_YESTERDAY = s.HC
        s.DCSUM = s.DCSUM + r.DCR 
        s.LTE50 = s.HC
        s.LTE10 = (s.LTE50 * p.LTE10M + p.LTE10B)
        s.LTE90 = (s.LTE50 * p.LTE90M + p.LTE90B)

        # Use HCMIN to determine if vinifera or labrusca
        s.PREDBB = torch.where((s.HC >= -2.2) & (self._HC_YESTERDAY < -2.2) & (p.HCMIN == -1.2), s.HC, 
                        torch.where((s.HC >= -6.4) & (self._HC_YESTERDAY < -6.4) & (p.HCMIN == -2.5), s.HC, torch.zeros_like(s.HC)))

        # Check if a new stage is reached
        self._STAGE[(self._endodorm & (s.CSUM <= p.ECOBOUND)).cpu().numpy()] = "ecodorm"

    def get_output(self, vars:list=None):
        """
        Return the LTE50
        """
        if vars is None:
            return torch.unsqueeze(self.states.LTE50, -1)
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[:,i] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[:,i] = getattr(self.rates,v)
            return output_vars
        
    def reset(self, day:datetime.date):
        """
        Reset the model
        """

        p = self.params
        self._STAGE = ["endodorm" for _ in range(self.num_models)]
        LTE10 = p.HCINIT * p.LTE10M + p.LTE10B
        LTE90 = p.HCINIT * p.LTE90M + p.LTE90B
        self.states = self.StateVariables(num_models=self.num_models, DHSUM=0., DCSUM=0.,HC=p.HCINIT[0].detach().cpu(),
                                          PREDBB=0., LTE50=p.HCINIT[0].detach().cpu(), CSUM=0.,
                                          LTE10=LTE10[0].detach().cpu(), LTE90=LTE90[0].detach().cpu())
        self.rates = self.RateVariables(num_models=self.num_models)
        self._HC_YESTERDAY = p.HCINIT[0].detach().clone()

    def get_extra_states(self):
        """Get extra states"""
        return {"_STAGE":self._STAGE, "_HC_YESTERDAY":self._HC_YESTERDAY}

    def set_model_specific_params(self, k, v):
        """Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        if k == "THETA":
            setattr(self.params, k, torch.floor(v).detach()+(v-v.detach()))
        else:
            setattr(self.params, k, v)
