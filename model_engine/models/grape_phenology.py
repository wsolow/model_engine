"""Implementation of the grape phenology model based on the GDD model with the Triangular
temperature accumulation function

Written by Will Solow, 2025
"""
import datetime
import torch
import copy

from traitlets_pcse import Float, Enum, Dict

from model_engine.weather.util import daylength
from model_engine.models.base_model import BaseModel
from model_engine.util import Tensor
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

def daily_temp_units(drv, T0BC: torch.Tensor, TMBC: torch.Tensor, TMIN_TENSOR, TMAX_TENSOR, A_c):
    """
    Compute the daily temperature units using the BRIN model.
    Used for predicting budbreak in grapes.

    Slightly modified to not use the min temp at day n+1, but rather reuse the min
    temp at day n
    """
    
    c_min = copy.deepcopy(A_c)
    for h in range(1, 25):
        # Perform linear interpolation between the hours 1 and 24
        TMAX = torch.sum(drv * TMAX_TENSOR)
        TMIN = torch.sum(drv * TMIN_TENSOR)
        if h <= 12:
            T_n = TMIN + h * ((TMAX - TMIN) / 12)
        else:
            T_n = TMAX - (h - 12) * ((TMAX - TMIN) / 12)

        # Limit the interpolation based on parameters
        T_n = torch.clamp(T_n - T0BC, c_min, TMBC - T0BC)
        A_c = A_c + T_n

    return A_c / 24          

class Grape_Phenology(BaseModel):
    """Implements grape phenology GDD model
    """

    _DAY_LENGTH = Float(12.0) # Helper variable for daylength
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})
    # Based on the Elkhorn-Lorenz Grape Phenology Stage
    _STAGE  = Enum(["endodorm", "ecodorm", "budbreak", "flowering", "verasion", "ripe"], allow_none=True)

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
        DVS    = Tensor(-99.)  # Development stage
        TSUME  = Tensor(-99.)  # Temperature sum for emergence state
        TSUM   = Tensor(-99.)  # Temperature sum state
        CSUM   = Tensor(-99.)  # Chilling sum state
      
    def __init__(self, day:datetime.date, parvalues:dict, device):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        super().__init__(self, parvalues, device)

        # Define initial states
        self._STAGE = "ecodorm"
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        
        self.rates = self.RateVariables()

        self.TMIN_TENSOR = torch.tensor([0.,1.,0.,0.,0.,0.,0.]).to(self.device)
        self.TMAX_TENSOR = torch.tensor([0.,0.,1.,0.,0.,0.,0.]).to(self.device)
        self.TBASEM_TENSOR = torch.tensor([0.]).to(self.device)

    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        # Day length sensitivity
        self._DAY_LENGTH = daylength(day, drv[-2])

        r.DTSUME = 0.
        r.DTSUM = 0.
        r.DVR = 0.
        # Development rates

        A_c = torch.tensor([0.]).to(self.device)

        if self._STAGE == "endodorm":
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM4

        elif self._STAGE == "ecodorm":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUME = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUME / p.TSUMEM

        elif self._STAGE == "budbreak":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM1

        elif self._STAGE == "flowering":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM2

        elif self._STAGE == "verasion":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM3

        elif self._STAGE == "ripe":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM4

        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self.stateself._STAGE)
        

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

        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self._STAGE)   

    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return self.states.DVS[:,None]
        else:
            output_vars = []
            for v in vars:
                if v in self.states.trait_names():
                    output_vars.append(getattr(self.states, v))
                elif v in self.rates.trait_names():
                    output_vars.append(getattr(self.rates, v))
            return torch.cat(output_vars)[:,None]
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        self.rates = self.RateVariables()
