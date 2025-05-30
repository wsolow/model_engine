"""
tensor_grape_phenology.py

Implementation of the grape phenology model based on the GDD model
with pytorch tensors

Written by Will Solow, 2025
"""

import datetime
import torch

from model_engine.inputs.util import daylength
from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
EPS = 1e-12

class Grape_Phenology_Tensor(TensorModel):
    """Implements grape phenology GDD model
    """
    
    _STAGE_VAL = {"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5}
    _DAY_LENGTH = Tensor(12.0) 
    _STAGE  = "ecodorm"

    class Parameters(ParamTemplate):
        TBASEM = Tensor(-99.)  # Base temp. for bud break
        TEFFMX = Tensor(-99.)  # Max eff temperature for grow daily units
        TSUMEM = Tensor(-99.)  # Temp. sum for bud break

        TSUM1  = Tensor(-99.)  # Temperature sum budbreak to flowering
        TSUM2  = Tensor(-99.)  # Temperature sum flowering to verasion
        TSUM3  = Tensor(-99.)  # Temperature sum from verasion to ripe
        TSUM4  = Tensor(-99.)  # Temperature sum from ripe onwards
        MLDORM = Tensor(-99.)  # Daylength at which a plant will go into dormancy
        Q10C   = Tensor(-99.)  # Parameter for chilling unit accumulation
        CSUMDB = Tensor(-99.)  # Chilling unit sum for dormancy break

    class RateVariables(RatesTemplate):
        DTSUME = Tensor(-99.)  # increase in temperature sum for emergence
        DTSUM  = Tensor(-99.)  # increase in temperature sum
        DVR    = Tensor(-99.)  # development rate
        DCU    = Tensor(-99.)  # Daily chilling units

    class StateVariables(StatesTemplate):
        PHENOLOGY = Tensor(-.99) # Int of Stage
        DVS       = Tensor(-99.)  # Development stage
        TSUME     = Tensor(-99.)  # Temperature sum for emergence state
        TSUM      = Tensor(-99.)  # Temperature sum state
        CSUM      = Tensor(-99.)  # Chilling sum state
      
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device):
        super().__init__(day, kiosk, parvalues, device)

        # Define initial states
        self._STAGE = "ecodorm"
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        
        self.rates = self.RateVariables()

        self.min_tensor = torch.tensor([0.]).to(self.device)

    def calc_rates(self, day, drv):
        """
        Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        # Day length sensitivity
        if hasattr(drv, "DAYL"):
            self._DAY_LENGTH = drv.DAYL
        elif hasattr(drv, "LAT"):
            self._DAY_LENGTH = daylength(day, drv.LAT)
        r.DTSUME = 0.
        r.DTSUM = 0.
        r.DVR = 0.
        # Development rates
        if self._STAGE == "endodorm":
            r.DTSUM = torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / (p.TSUM4+EPS)

        elif self._STAGE == "ecodorm":
            r.DTSUME = torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUME / (p.TSUMEM+EPS)

        elif self._STAGE == "budbreak":
            r.DTSUM = torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / (p.TSUM1+EPS)

        elif self._STAGE == "flowering":
            r.DTSUM = torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / (p.TSUM2+EPS)

        elif self._STAGE == "verasion":
            r.DTSUM = torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / (p.TSUM3+EPS)

        elif self._STAGE == "ripe":
            r.DTSUM = torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / (p.TSUM4+EPS)

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

        # Integrate phenologic states
        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        
        s.TSUM = s.TSUM + r.DTSUM
        s.CSUM = s.CSUM + r.DCU
        s.PHENOLOGY = torch.floor(s.DVS).detach() + (s.DVS - s.DVS.detach()) 

        # Check if a new stage is reached
        if self._STAGE == "endodorm":
            if s.CSUM >= p.CSUMDB:
                self._STAGE = "ecodorm"
                s.TSUM  = 0.
                s.TSUME = 0.
                s.DVS = 0.0
                s.CSUM = 0

        elif self._STAGE == "ecodorm":
            if s.TSUME >= p.TSUMEM:
                self._STAGE = "budbreak"

        elif self._STAGE == "budbreak":
            if s.DVS >= 2.0:
                self._STAGE = "flowering"

        elif self._STAGE == "flowering":
            if s.DVS >= 3.0:
                self._STAGE = "verasion"

        elif self._STAGE == "verasion":
            if s.DVS >= 4.0:
                self._STAGE = "ripe"

            if self._DAY_LENGTH <= p.MLDORM:
                self._STAGE = "endodorm"

        elif self._STAGE == "ripe":
            if self._DAY_LENGTH <= p.MLDORM:
                self._STAGE = "endodorm"

        else:
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self._STAGE)   
        
    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return torch.unsqueeze(self.states.DVS, -1)
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
        self._STAGE = "ecodorm"
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        self.rates = self.RateVariables()


    def daily_temp_units(self, drv):
        """
        Compute the daily temperature units using the BRIN model.
        Used for predicting budbreak in grapes.

        Slightly modified to not use the min temp at day n+1, but rather reuse the min
        temp at day n
        # CURRENTLY NOT IN USE
        # Makes computational graph too large
        """
        p = self.params
        A_c = torch.tensor([0.]).to(self.device)._requires_grad(False)

        for h in range(1, 25):
            # Perform linear interpolation between the hours 1 and 24
            if h <= 12:
                T_n = drv.TMIN + h * ((drv.TMAX - drv.TMIN) / 12)
            else:
                T_n = drv.TMAX - (h - 12) * ((drv.TMAX - drv.TMIN) / 12)

            ## Limit the interpolation based on parameters
            T_n = torch.clamp(T_n - p.TBASEM, self.min_tensor, p.TEFFMX - p.TBASEM)._requires_grad(False)
            A_c = A_c + T_n
        return A_c / 24   
    
    def get_extra_states(self):
        """
        Get extra states
        """
        return {"_STAGE": self._STAGE, "_DAY_LENGTH": self._DAY_LENGTH}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)