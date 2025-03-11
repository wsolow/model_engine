"""
Implementation of the grape phenology model based on the GDD model
with pytorch tensors to simulate multiple models
Written by Will Solow, 2025
"""
import datetime
import numpy as np

from traitlets_pcse import Dict


from model_engine.weather.util import daylength
from model_engine.models.base_model import BatchNumpyModel
from model_engine.models.states_rates import NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
class Grape_Phenology_NumpyBatch(BatchNumpyModel):
    """Implements grape phenology GDD model
    """

    _DAY_LENGTH = NDArray(12.0) # Helper variable for daylength
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})
    # Based on the Elkhorn-Lorenz Grape Phenology Stage
    _STAGE  = NDArray(["endodorm"])

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
      
    def __init__(self, day:datetime.date, parvalues:dict, device, num_models:int=1):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.num_models = num_models
        super().__init__(self, parvalues, device, self.num_models)

        # Define initial states
        self._STAGE = ["ecodorm" for _ in range(self.num_models)]
        self.states = self.StateVariables(num_models=self.num_models, TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL["ecodorm"])
        
        self.rates = self.RateVariables(num_models=self.num_models)
        self.min_tensor = np.array([0.])


    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        # Day length sensitivity
        self._DAY_LENGTH = daylength(day, drv.LAT)
        r.DTSUME = np.zeros(shape=(self.num_models,))
        r.DTSUM = np.zeros(shape=(self.num_models,))
        r.DVR = np.zeros(shape=(self.num_models,))
        endodorm = np.array(self._STAGE == "endodorm")
        ecodorm = np.array(self._STAGE == "ecodorm")
        budbreak = np.array(self._STAGE == "budbreak")
        flowering = np.array(self._STAGE == "flowering")
        verasion = np.array(self._STAGE == "verasion")
        ripe = np.array(self._STAGE == "ripe")

        r.DTSUM = np.where(endodorm, np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUME = np.where(ecodorm, np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = np.where(budbreak, np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = np.where(flowering, np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = np.where(verasion, np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = np.where(ripe, np.clip(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)

        r.DVR = np.where(endodorm, r.DTSUM / p.TSUM4, r.DVR)
        r.DVR = np.where(ecodorm, r.DTSUME / p.TSUMEM, r.DVR)
        r.DVR = np.where(budbreak, r.DTSUM / p.TSUM1, r.DVR)
        r.DVR = np.where(flowering, r.DTSUM / p.TSUM2, r.DVR)
        r.DVR = np.where(verasion, r.DTSUM / p.TSUM3, r.DVR)
        r.DVR = np.where(ripe, r.DTSUM / p.TSUM4, r.DVR)

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
        for i in range(self.num_models):
            if self._STAGE[i] == "endodorm":
                if s.CSUM[i] >= p.CSUMDB[i]:
                    self._STAGE[i] = "ecodorm"
                    s.TSUM[i]  = 0.
                    s.TSUME[i] = 0.
                    s.DVS[i] = 0.0
                    s.CSUM[i] = 0.

            elif self._STAGE[i] == "ecodorm":
                if s.TSUME[i] >= p.TSUMEM[i]:
                    self._STAGE[i] = "budbreak"

            elif self._STAGE[i] == "budbreak":
                if s.DVS[i] >= 2.0:
                    self._STAGE[i] = "flowering"

            elif self._STAGE[i] == "flowering":
                if s.DVS[i] >= 3.0:
                    self._STAGE[i] = "verasion"

            elif self._STAGE[i] == "verasion":
                if s.DVS[i] >= 4.0:
                    self._STAGE[i] = "ripe"
                if self._DAY_LENGTH[i] <= p.MLDORM[i]:
                    self._STAGE[i] = "endodorm"

            elif self._STAGE[i] == "ripe":
                if self._DAY_LENGTH[i] <= p.MLDORM[i]:
                    self._STAGE[i] = "endodorm"

            else:  # Problem: no stage defined
                msg = "Unrecognized STAGE defined in phenology submodule: %s."
                raise Exception(msg, self._STAGE[i]) 
        
    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return np.expand_dims(self.states.DVS, -1)
        else:
            output_vars = np.empty(shape=(self.num_models, len(vars)))
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[:,i] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[:,i] = getattr(self.states, v)
            return output_vars
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        self._STAGE = ["ecodorm" for _ in range(self.num_models)]
        self.states = self.StateVariables(num_models=self.num_models,TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL["ecodorm"])
        self.rates = self.RateVariables(num_models=self.num_models)


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