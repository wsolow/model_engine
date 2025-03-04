"""Implementation of Feguson Model for Grape Cold Hardiness

Written by Will Solow, 2025
"""
import datetime
import torch

from traitlets_pcse import  Enum, Dict
from model_engine.models.base_model import BaseModel
from model_engine.util import Tensor
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
     

class Grape_ColdHardiness(BaseModel):
    """Implements Feguson grape cold hardiness model
    """

    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})
    _STAGE  = Enum(["endodorm", "ecodorm", "budbreak", "flowering", "verasion", "ripe"], allow_none=True)

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
        DORMBD     = Tensor(-99.) # Temperature threshold for onset of ecodormancy

    class RateVariables(RatesTemplate):
        DCU       = Tensor(-99.) # Daily heat accumulation
        DHR    = Tensor(-99.) # Daily heating rate
        DCR    = Tensor(-99.) # Daily chilling rate
        DACC   = Tensor(-99.) # Deacclimation rate
        ACC    = Tensor(-99.) # Acclimation rate
        HCR    = Tensor(-99.) # Change in acclimation
        STAGE  = Tensor(-99.) # Stage (endodormancy or ecodormancy)

    class StateVariables(StatesTemplate):
        CSUM      = Tensor(-99.) # Daily temperature sum for phenology
        DHSUM     = Tensor(-99.) # Daily heating sum
        DCSUM     = Tensor(-99.) # Daily chilling sum
        HC        = Tensor(-99.) # Cold hardiness
        PREDBB    = Tensor(-99.) # Predicted bud break
        LTE50     = Tensor(-99.) # Predicted LTE50 for cold hardiness
        PHENOLOGY = Tensor(-.99) # Int of Stage
             
    def __init__(self, day:datetime.date, parvalues:dict, device):
        """
        :param day: start date of the simulation
        :param parvalues: providing parameters as key/value pairs
        """
        super().__init__(self, parvalues, device)

        # Define initial states
        p = self.params
        self._STAGE = "endodorm"
        self.states = self.StateVariables(DHSUM=0., DCSUM=0.,HC=p.HCINIT,
                                          PREDBB=0., LTE50=p.HCINIT, 
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        
        self.rates = self.RateVariables()

    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
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
        r.DCU = torch.max(0., drv.TEMP - 10.)

        if self._STAGE == "endodorm":
            r.DHR = torch.max(0., drv.TEMP-p.TENDO)
            r.DCR = torch.min(0., drv.TEMP-p.TENDO)
            if s.DCSUM != 0:
                r.DACC = r.DHR * p.ENDEACCLIM * (1 - ((self._HC_YESTERDAY-p.HCMAX) / (p.HCMIN-p.HCMAX)))
            else:
                r.DACC = 0
            r.ACC = r.DCR * p.ENACCLIM * (1-((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN-p.HCMAX)))
            r.HCR = r.DACC + r.ACC

        elif self._STAGE == "ecodorm":
            r.DHR = torch.max(0., drv.TEMP-p.TECO)
            r.DCR = torch.min(0., drv.TEMP-p.TECO)
            if s.DCSUM != 0:
                r.DACC = r.DHR * p.ECDEACCLIM * (1 - ((self._HC_YESTERDAY-p.HCMAX) / (p.HCMIN-p.HCMAX)) ** p.THETA)
            else:
                r.DACC = 0
            r.ACC = r.DCR * p.ECACCLIM * (1-((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN-p.HCMAX)))
            r.HCR = r.DACC + r.ACC

        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self._STAGE)
        

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states

        # Integrate phenologic states
        s.CSUM = s.CSUM + r.DCU 
        s.PHENOLOGY = self._STAGE_VAL[s.STAGE]

        self._HC_YESTERDAY = s.HC
        s.HC = torch.clamp(p.HCMAX, p.HCMIN, s.HC+r.HCR)
        s.DCSUM = s.DCSUM + r.DCR 
        s.LTE50 = torch.round(s.HC, 2)

        # Use HCMIN to determine if vinifera or labrusca
        if p.HCMIN == -1.2:    # Assume vinifera with budbreak at -2.2
            if self._HC_YESTERDAY < -2.2:
                if s.HC >= -2.2:
                    s.PREDBB = torch.round(s.HC, 2)
        if p.HCMIN == -2.5:    # Assume labrusca with budbreak at -6.4
            if self._HC_YESTERDAY < -6.4:
                if s.HC >= -6.4:
                    s.PREDBB = torch.round(s.HC, 2)

        # Check if a new stage is reached
        if self._STAGE == "endodorm":
            if s.CSUM >= p.DORMBD:
                self._STAGE = "ecodorm"
                s.PHENOLOGY = self._STAGE_VAL[self._STAGE]


        elif self._STAGE == "ecodorm":
            pass
                        
        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self._STAGE)     

    def get_output(self):
        """
        Return the LTE50 for cold hardiness
        """

        return self.states.LTE50
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        p = self.params
        self._STAGE = "endodorm"
        self.states = self.StateVariables(DHSUM=0., DCSUM=0.,HC=p.HCINIT,
                                          PREDBB=0., LTE50=p.HCINIT, 
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        
        self.rates = self.RateVariables()