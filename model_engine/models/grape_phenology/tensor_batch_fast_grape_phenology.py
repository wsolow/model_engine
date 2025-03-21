"""
Implementation of the grape phenology model based on the GDD model
with pytorch tensors to simulate multiple models
Written by Will Solow, 2025
"""
import datetime
import torch

from traitlets_pcse import Dict


from model_engine.inputs.util import daylength
from model_engine.models.base_model import BatchTensorModelFast
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
class Grape_Phenology_TensorBatchFast(BatchTensorModelFast):
    """Implements grape phenology GDD model
    """

    _DAY_LENGTH = Tensor(12.0) # Helper variable for daylength
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})
    # Based on the Elkhorn-Lorenz Grape Phenology Stage
    _STAGE  = NDArray(["endodorm"])

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

    P = dict(zip(["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2", "TSUM3", "TSUM4", "MLDORM", "Q10C", "CSUMDB"], range(10)))
    S = dict(zip(["PHENOLOGY", "DVS", "TSUME", "TSUM", "CSUM"], range(5)))
    R = dict(zip(["DTSUME", "DTSUM", "DVR", "DCU"], range(5)))
      
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
        self.states = torch.zeros(size=(self.num_models, len(self.S))).to(self.device)
        self.rates = torch.zeros(size=(self.num_models, len(self.R))).to(self.device)
        
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
    
        r[:,self.R["DTSUME"]] = torch.zeros(size=(self.num_models,))
        r[:,self.R["DTSUM"]] = torch.zeros(size=(self.num_models,))
        r[:,self.R["DVR"]] = torch.zeros(size=(self.num_models,))
        endodorm = torch.tensor(self._STAGE == "endodorm").to(self.device)
        ecodorm = torch.tensor(self._STAGE == "ecodorm").to(self.device)
        budbreak = torch.tensor(self._STAGE == "budbreak").to(self.device)
        flowering = torch.tensor(self._STAGE == "flowering").to(self.device)
        verasion = torch.tensor(self._STAGE == "verasion").to(self.device)
        ripe = torch.tensor(self._STAGE == "ripe").to(self.device)
        
        r[:,self.R["DTSUM"]] = torch.where(endodorm, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r[:,self.R["DTSUM"]])
        r[:,self.R["DTSUME"]] = torch.where(ecodorm, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r[:,self.R["DTSUME"]])
        r[:,self.R["DTSUM"]] = torch.where(budbreak, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r[:,self.R["DTSUM"]])
        r[:,self.R["DTSUM"]] = torch.where(flowering, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r[:,self.R["DTSUM"]])
        r[:,self.R["DTSUM"]] = torch.where(verasion, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r[:,self.R["DTSUM"]])
        r[:,self.R["DTSUM"]] = torch.where(ripe, torch.clamp(drv.TEMP-p.TBASEM, self.min_tensor, p.TEFFMX), r[:,self.R["DTSUM"]])

        r = r.clone() # TODO: make this better, Clunky, but avoids in place operation
        r[:,self.R["DVR"]] = torch.where(endodorm, r[:,self.R["DTSUM"]] / p.TSUM4, r[:,self.R["DVR"]])
        r[:,self.R["DVR"]] = torch.where(ecodorm, r[:,self.R["DTSUME"]] / p.TSUMEM, r[:,self.R["DVR"]])
        r[:,self.R["DVR"]] = torch.where(budbreak, r[:,self.R["DTSUM"]] / p.TSUM1, r[:,self.R["DVR"]])
        r[:,self.R["DVR"]] = torch.where(flowering, r[:,self.R["DTSUM"]] / p.TSUM2, r[:,self.R["DVR"]])
        r[:,self.R["DVR"]] = torch.where(verasion, r[:,self.R["DTSUM"]] / p.TSUM3, r[:,self.R["DVR"]])
        r[:,self.R["DVR"]] = torch.where(ripe, r[:,self.R["DTSUM"]] / p.TSUM4, r[:,self.R["DVR"]])

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states

        # Integrate phenologic states
        s[:,self.S["TSUME"]] = s[:,self.S["TSUME"]] + r[:,self.R["DTSUME"]]
        s[:,self.S["DVS"]] = s[:,self.S["DVS"]] + r[:,self.R["DVR"]]
        
        s[:,self.S["TSUM"]] = s[:,self.S["TSUM"]] + r[:,self.R["DTSUM"]]
        s[:,self.S["CSUM"]] = s[:,self.S["CSUM"]] + r[:,self.R["DCU"]]
        s[:,self.S["PHENOLOGY"]] = torch.floor(s[:,self.S["DVS"]]).detach() + (s[:,self.S["DVS"]] - s[:,self.S["DVS"]].detach()) 

        # Check if a new stage is reached
        for i in range(self.num_models):
            if self._STAGE[i] == "endodorm":
                if s[:,self.S["CSUM"]][i] >= p.CSUMDB[i]:
                    self._STAGE[i] = "ecodorm"
                    s[:,self.S["TSUM"]][i]  = 0.
                    s[:,self.S["TSUME"]][i] = 0.
                    s[:,self.S["DVS"]][i] = 0.0
                    s[:,self.S["CSUM"]][i] = 0.
                    s[:,self.S["PHENOLOGY"]][i] = 0.

            elif self._STAGE[i] == "ecodorm":
                if s[:,self.S["TSUME"]][i] >= p.TSUMEM[i]:
                    self._STAGE[i] = "budbreak"

            elif self._STAGE[i] == "budbreak":
                if s[:,self.S["DVS"]][i] >= 2.0:
                    self._STAGE[i] = "flowering"

            elif self._STAGE[i] == "flowering":
                if s[:,self.S["DVS"]][i] >= 3.0:
                    self._STAGE[i] = "verasion"

            elif self._STAGE[i] == "verasion":
                if s[:,self.S["DVS"]][i] >= 4.0:
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
            return torch.unsqueeze(self.states.DVS, -1)
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)

            for i, v in enumerate(vars):
                if v in self.S.keys():
                    output_vars[:,i] = self.states[:,self.S[v]]
                elif v in self.R.keys():
                    output_vars[:,i] = self.rates[:,self.R[v]]
            return output_vars
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        self._STAGE = ["ecodorm" for _ in range(self.num_models)]
        self.states = torch.zeros(size=(self.num_models, len(self.S))).to(self.device)
        self.rates = torch.zeros(size=(self.num_models, len(self.R))).to(self.device)

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