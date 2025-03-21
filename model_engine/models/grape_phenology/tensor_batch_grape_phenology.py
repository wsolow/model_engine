"""
Implementation of the grape phenology model based on the GDD model
with pytorch tensors to simulate multiple models
Written by Will Solow, 2025
"""
import datetime
import torch

from traitlets_pcse import Dict


from model_engine.inputs.util import daylength
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

EPS = 1e-12

class Grape_Phenology_TensorBatch(BatchTensorModel):
    """Implements grape phenology GDD model
    """

    _DAY_LENGTH = Tensor(12.0) # Helper variable for daylength
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "veraison":3, "ripe":4, "endodorm":5})
    # Based on the Elkhorn-Lorenz Grape Phenology Stage
    _STAGE  = NDArray(["ecodorm"])

    class Parameters(ParamTemplate):
        TBASEM = Tensor(-99.)  # Base temp. for bud break
        TEFFMX = Tensor(-99.)  # Max eff temperature for grow daily units
        TSUMEM = Tensor(-99.)  # Temp. sum for bud break

        TSUM1  = Tensor(-99.)  # Temperature sum budbreak to flowering
        TSUM2  = Tensor(-99.)  # Temperature sum flowering to veraison
        TSUM3  = Tensor(-99.)  # Temperature sum from veraison to ripe
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
      
    def __init__(self, day:datetime.date, parvalues:dict, device, num_models:int=1):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.num_models = num_models
        self.num_stages = len(self._STAGE_VAL)
        self.stages = list(self._STAGE_VAL.keys())
        super().__init__(self, parvalues, device, self.num_models)

        # Define initial states
        self._STAGE = ["ecodorm" for _ in range(self.num_models)]
        self.states = self.StateVariables(num_models=self.num_models, TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL["ecodorm"])
        
        self.rates = self.RateVariables(num_models=self.num_models)
        self.min_tensor = torch.tensor([0.]).to(self.device)

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
        if self._DAY_LENGTH.ndim == 0:
            self._DAY_LENGTH = torch.tile(self._DAY_LENGTH, (self.num_models,))[:self.num_models]
        elif len(self._DAY_LENGTH) < self.num_models:
            self._DAY_LENGTH = torch.tile(self._DAY_LENGTH, (self.num_models // len(self._DAY_LENGTH) + 1,))[:self.num_models]

        r.DTSUME = torch.zeros(size=(self.num_models,))
        r.DTSUM = torch.zeros(size=(self.num_models,))
        r.DVR = torch.zeros(size=(self.num_models,))

        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device) # create masks
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)]) # one hot encoding matrix
        self._ecodorm, self._budbreak, self._flowering, self._veraison, self._ripe, self._endodorm = stage_masks # upack for readability
        
        # Compute DTSUM values
        dtsum_update = torch.clamp(drv.TEMP - p.TBASEM, self.min_tensor, p.TEFFMX)
        # Apply DTSUM updates only where each stage condition is met
        r.DTSUM = torch.where(
            self._endodorm | self._budbreak | self._flowering | self._veraison | self._ripe, dtsum_update, r.DTSUM
        )
        r.DTSUME = torch.where(self._ecodorm, dtsum_update, r.DTSUME)

        # Stack all TSUM values for vectorized selection
        TSUM_stack = torch.where(self._ripe, p.TSUM4, torch.where(
            self._veraison, p.TSUM3, torch.where(
                self._flowering, p.TSUM2, torch.where(
                    self._budbreak, p.TSUM1, torch.where(
                        self._ecodorm, p.TSUMEM, torch.where(
                            self._endodorm, p.TSUM4, torch.ones_like(p.TSUM4)
                        )
                    )
                )
            )
        ))
        # Compute DVR in a single operation
        r.DVR = torch.where(self._ecodorm, r.DTSUME / (TSUM_stack+EPS), r.DTSUM / (TSUM_stack+EPS) ) # Add epsilon to prevent division by zero

        ''' # Code that is somewhat readable but suffers readability
        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device) # create masks
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)]) # one hot encoding matrix
        ecodorm, budbreak, flowering, veraison, ripe, endodorm = stage_masks # upack for readability

        # Compute DTSUM update once
        dtsum_update = torch.clamp(drv.TEMP - p.TBASEM, self.min_tensor, p.TEFFMX)

        # Apply DTSUM updates efficiently
        r.DTSUM = torch.where(
            endodorm | budbreak | flowering | veraison | ripe, dtsum_update, r.DTSUM
        )
        r.DTSUME = torch.where(ecodorm, dtsum_update, r.DTSUME)
        # Stack TSUM values for indexing
        TSUM_values = torch.stack([p.TSUMEM, p.TSUM1, p.TSUM2, p.TSUM3, p.TSUM4, p.TSUM4])
        col_indices = torch.arange(TSUM_values.shape[1]).to(stage_tensor.device)  # Shape: (2,)

        # Use advanced indexing to select values row-wise
        TSUM_selected = TSUM_values[stage_tensor, col_indices]  # Shape: (batch_size,)

        # Compute DVR in a single step
        r.DVR = torch.where(ecodorm, r.DTSUME / (TSUM_selected+EPS), r.DTSUM / (TSUM_selected+EPS) )'''

        ''' # Oldest code that is most readable 
        r.DTSUME = torch.zeros(size=(self.num_models,))
        r.DTSUM = torch.zeros(size=(self.num_models,))
        r.DVR = torch.zeros(size=(self.num_models,))
        endodorm = torch.tensor(self._STAGE == "endodorm").to(self.device)
        ecodorm = torch.tensor(self._STAGE == "ecodorm").to(self.device)
        budbreak = torch.tensor(self._STAGE == "budbreak").to(self.device)
        flowering = torch.tensor(self._STAGE == "flowering").to(self.device)
        veraison = torch.tensor(self._STAGE == "veraison").to(self.device)
        ripe = torch.tensor(self._STAGE == "ripe").to(self.device)
        
        r.DTSUM = torch.where(endodorm, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUME = torch.where(ecodorm, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUME)
        r.DTSUM = torch.where(budbreak, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = torch.where(flowering, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = torch.where(veraison, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)
        r.DTSUM = torch.where(ripe, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r.DTSUM)

        r.DVR = torch.where(endodorm, r.DTSUM / (p.TSUM4+EPS), r.DVR)
        r.DVR = torch.where(ecodorm, r.DTSUME / (p.TSUMEM+EPS), r.DVR)
        r.DVR = torch.where(budbreak, r.DTSUM / (p.TSUM1+EPS), r.DVR)
        r.DVR = torch.where(flowering, r.DTSUM / (p.TSUM2+EPS), r.DVR)
        r.DVR = torch.where(veraison, r.DTSUM / (p.TSUM3+EPS), r.DVR)
        r.DVR = torch.where(ripe, r.DTSUM / (p.TSUM4+EPS), r.DVR)'''

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
        
        # Stage transitions for "endodorm" -> "ecodorm"
        self._STAGE[(self._endodorm & (s.CSUM >= p.CSUMDB)).cpu()] = "ecodorm"
        s.TSUM[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0
        s.TSUME[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0
        s.DVS[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0
        s.CSUM[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0

        # Stage transitions for "ecodorm" -> "budbreak"
        self._STAGE[(self._ecodorm & (s.TSUME >= p.TSUMEM)).cpu()] = "budbreak"

        # Stage transitions for "budbreak" -> "flowering"
        self._STAGE[(self._budbreak & (s.DVS >= 2.0)).cpu()] = "flowering"

        # Stage transitions for "flowering" -> "veraison"
        self._STAGE[(self._flowering & (s.DVS >= 3.0)).cpu()] = "veraison"

        # Stage transitions for "veraison" -> "ripe" and check for dormancy
        self._STAGE[(self._veraison & (s.DVS >= 4.0)).cpu()] = "ripe"
        self._STAGE[(self._veraison & (self._DAY_LENGTH <= p.MLDORM)).cpu()] = "endodorm"

        # Stage transitions for "ripe" -> "endodorm"
        self._STAGE[(self._ripe & (self._DAY_LENGTH <= p.MLDORM)).cpu()] = "endodorm"
        
        # Legacy code that is a bit slower
        '''# Check if a new stage is reached
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
                    self._STAGE[i] = "veraison"

            elif self._STAGE[i] == "veraison":
                if s.DVS[i] >= 4.0:
                    self._STAGE[i] = "ripe"
                if self._DAY_LENGTH[i] <= p.MLDORM[i]:
                    self._STAGE[i] = "endodorm"

            elif self._STAGE[i] == "ripe":
                if self._DAY_LENGTH[i] <= p.MLDORM[i]:
                    self._STAGE[i] = "endodorm"

            else:  # Problem: no stage defined
                msg = "Unrecognized STAGE defined in phenology submodule: %s."
                raise Exception(msg, self._STAGE[i]) '''
        
    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return torch.unsqueeze(self.states.DVS, -1)
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[:,i] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[:,i] = getattr(self.states,v)
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
        A_c = torch.Tensor([0.]).to(self.device)._requires_grad(False)

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