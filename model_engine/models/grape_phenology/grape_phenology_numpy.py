"""Implementation of the grape phenology model based on the GDD model with the Triangular
temperature accumulation function

Written by Will Solow, 2025
"""
import datetime
import numpy as np

from traitlets_pcse import  Enum, Dict

from model_engine.inputs.util import daylength
from model_engine.models.base_model import NumpyModel
from model_engine.models.states_rates import NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
class Grape_Phenology_Numpy(NumpyModel):
    """Implements grape phenology GDD model
    """

    _DAY_LENGTH = NDArray(12.0) # Helper variable for daylength
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})
    # Based on the Elkhorn-Lorenz Grape Phenology Stage
    _STAGE  = Enum(["endodorm", "ecodorm", "budbreak", "flowering", "verasion", "ripe"], allow_none=True)

    class Parameters(ParamTemplate):
        TBASEM = NDArray(-99.)  # Base temp. for bud break
        TEFFMX = NDArray(-99.)  # Max eff temperature for grow daily units
        TSUMEM = NDArray(-99.)  # Temp. sum for bud break

        TSUM1  = NDArray(-99.)  # Temperature sum budbreak to flowering
        TSUM2  = NDArray(-99.)  # Temperature sum flowering to verasion
        TSUM3  = NDArray(-99.)  # Temperature sum from verasion to ripe
        TSUM4  = NDArray(-99.)  # Temperature sum from ripe onwards
        MLDORM = NDArray(-99.)  # Daylength at which a plant will go into dormancy
        Q10C   = NDArray(-99.)  # Parameter for chilling unit accumulation
        CSUMDB = NDArray(-99.)  # Chilling unit sum for dormancy break

    class RateVariables(RatesTemplate):
        DTSUME = NDArray(-99.)  # increase in temperature sum for emergence
        DTSUM  = NDArray(-99.)  # increase in temperature sum
        DVR    = NDArray(-99.)  # development rate
        DCU    = NDArray(-99.)  # Daily chilling units

    class StateVariables(StatesTemplate):
        PHENOLOGY = NDArray(-.99) # Int of Stage
        DVS    = NDArray(-99.)  # Development stage
        TSUME  = NDArray(-99.)  # Temperature sum for emergence state
        TSUM   = NDArray(-99.)  # Temperature sum state
        CSUM   = NDArray(-99.)  # Chilling sum state
      
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

        self.min_tensor = np.array([0.])

    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
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
            r.DTSUM = np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM4

        elif self._STAGE == "ecodorm":
            r.DTSUME = np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUME / p.TSUMEM

        elif self._STAGE == "budbreak":
            r.DTSUM = np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM1

        elif self._STAGE == "flowering":
            r.DTSUM = np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM2

        elif self._STAGE == "verasion":
            r.DTSUM = np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM3

        elif self._STAGE == "ripe":
            r.DTSUM = np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM4

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
        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        
        s.TSUM = s.TSUM + r.DTSUM
        s.CSUM = s.CSUM + r.DCU
        s.PHENOLOGY = np.floor(s.DVS)

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
            return torch.unsqueeze(self.states.DVS, -1)
        else:
            output_vars = np.empty(shape=(len(vars),1))
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.states,v)
            return output_vars
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        self._STAGE = "ecodorm"
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        self.rates = self.RateVariables()


    def daily_temp_units(self, drv):
        # CURRENTLY NOT IN USE
        # Makes computational graph too large
        """
        Compute the daily temperature units using the BRIN model.
        Used for predicting budbreak in grapes.

        Slightly modified to not use the min temp at day n+1, but rather reuse the min
        temp at day n
        """
        p = self.params
        A_c = np.array([0.])

        for h in range(1, 25):
            # Perform linear interpolation between the hours 1 and 24
            if h <= 12:
                T_n = drv.TMIN + h * ((drv.TMAX - drv.TMIN) / 12)
            else:
                T_n = drv.TMAX - (h - 12) * ((drv.TMAX - drv.TMIN) / 12)

            ## Limit the interpolation based on parameters
            T_n = np.clip(T_n - p.TBASEM, self.min_tensor, p.TEFFMX - p.TBASEM)
            A_c = A_c + T_n
        return A_c / 24   