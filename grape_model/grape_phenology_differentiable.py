"""Implementation of  models for phenological development in WOFOST

Classes defined here:
- DVS_Phenology: Implements the algorithms for phenologic development
- Vernalisation: 

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
import datetime
import pickle
import torch
import copy

from traitlets_pcse import HasTraits, Float, Instance, Enum, Bool, Int, Dict

from .util import limit, daylength, Tensor
from .states_rates import ParamTemplate, StatesTemplate, RatesTemplate

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

class Differentiable_Grape_Phenology(HasTraits):
    """Implements grape phenology based on many papers provided by Markus Keller
    at Washington State University
    """

    _DAY_LENGTH = Float(12.0) # Helper variable for daylength
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})

    class Parameters(ParamTemplate):
        CROP_START_TYPE = Enum(["predorm", "endodorm", "ecodorm"], allow_none=True, default="endodorm")
        DVSI   = Tensor(-99.)  # Initial development stage
        DVSM   = Tensor(-99.)  # Mature development stage
        DVSEND = Tensor(-99.)  # Final development stage
        
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

        # Based on the Elkhorn-Lorenz Grape Phenology Stage
        STAGE  = Enum(["endodorm", "ecodorm", "budbreak", "flowering", "verasion", "ripe"], allow_none=True)
             
    def __init__(self, day:datetime.date, parvalues:dict, device):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.params = self.Parameters(parvalues)
        self.device = device

        # Define initial states
        DVS, DOP, STAGE = self._get_initial_stage(day)
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=DVS, STAGE=STAGE, 
                                          DOP=DOP, CSUM=0., PHENOLOGY=self._STAGE_VAL[STAGE])
        
        self.rates = self.RateVariables()

        self.TMIN_TENSOR = torch.tensor([0.,1.,0.,0.,0.,0.,0.]).to(self.device)
        self.TMAX_TENSOR = torch.tensor([0.,0.,1.,0.,0.,0.,0.]).to(self.device)
        self.TBASEM_TENSOR = torch.tensor([0.]).to(self.device)

    def _get_initial_stage(self, day:datetime.date):
        """Set the initial state of the crop given the start type
        """
        p = self.params

        # Define initial stage type (emergence/sowing) and fill the
        # respective day of sowing/emergence (DOS/DOE)
        if p.CROP_START_TYPE == "predorm":
            STAGE = "endodorm"
            DVS = p.DVSI
            DOP = day
        elif p.CROP_START_TYPE == "endodorm":
            STAGE = "ecodorm"
            DOP = day
            DVS = p.DVSI
        elif p.CROP_START_TYPE == "ecodorm":
            STAGE = "budbreak"
            DVS = p.DVSI
            DOP = day
        else:
            msg = "Unknown start type: %s. Are you using the corect Phenology \
                module (Calling the correct Gym Environment)?" % p.CROP_START_TYPE
            raise Exception(msg) 
        return DVS, DOP, STAGE
 
    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        # TEMP TMIN TMAX, RAIN IRRAD

        # Day length sensitivity
        self._DAY_LENGTH = daylength(day, drv[-2])

        r.DTSUME = 0.
        r.DTSUM = 0.
        r.DVR = 0.
        # Development rates

        A_c = torch.tensor([0.]).to(self.device)

        if s.STAGE == "endodorm":
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM4

        elif s.STAGE == "ecodorm":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUME = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUME / p.TSUMEM

        elif s.STAGE == "budbreak":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM1

        elif s.STAGE == "flowering":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM2

        elif s.STAGE == "verasion":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM3

        elif s.STAGE == "ripe":
            #DSUM = torch.sum(drv * TEMP_TENSOR) - p.TBASEM
            DSUM = daily_temp_units(drv, p.TBASEM, p.TEFFMX, self.TMIN_TENSOR, self.TMAX_TENSOR, A_c)
            r.DTSUM = torch.clamp(DSUM, self.TBASEM_TENSOR, p.TEFFMX)
            r.DVR = r.DTSUM / p.TSUM4

        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self.states.STAGE)
        

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
        s.PHENOLOGY = self._STAGE_VAL[s.STAGE]

        # Check if a new stage is reached
        if s.STAGE == "endodorm":
            if s.CSUM >= p.CSUMDB:
                self._next_stage(day)
                s.DVS = p.DVSI
                s.CSUM = 0

        elif s.STAGE == "ecodorm":
            if s.TSUME >= p.TSUMEM:
                self._next_stage(day)
                #s.DVS = 0.

        elif s.STAGE == "budbreak":
            if s.DVS >= 1.0:
                self._next_stage(day)
                # s.DVS = 1.0

        elif s.STAGE == "flowering":
            if s.DVS >= p.DVSM:
                self._next_stage(day)
                #s.DVS = p.DVSM

        elif s.STAGE == "verasion":
            if s.DVS >= p.DVSEND:
                self._next_stage(day)
                #s.DVS = p.DVSEND

            if self._DAY_LENGTH <= p.MLDORM:
                s.STAGE = "endodorm"
                s.PHENOLOGY = self._STAGE_VAL[s.STAGE]

        elif s.STAGE == "ripe":
            if self._DAY_LENGTH <= p.MLDORM:
                s.STAGE = "endodorm"
                s.PHENOLOGY = self._STAGE_VAL[s.STAGE]

        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise Exception(msg, self.states.STAGE)
            
    def _next_stage(self, day):
        """Moves states.STAGE to the next phenological stage"""
        s = self.states
        p = self.params

        if s.STAGE == "endodorm":
            s.STAGE = "ecodorm"
            self._on_DORMANT(day)
        
        elif s.STAGE == "ecodorm":
            s.STAGE = "budbreak"
            s.PHENOLOGY = self._STAGE_VAL[s.STAGE]
                
        elif s.STAGE == "budbreak":
            s.STAGE = "flowering"
            s.PHENOLOGY = self._STAGE_VAL[s.STAGE]

        elif s.STAGE == "flowering":
            s.STAGE = "verasion"
            s.PHENOLOGY = self._STAGE_VAL[s.STAGE]
            
        elif s.STAGE == "verasion":
            s.STAGE = "ripe"
            s.PHENOLOGY = self._STAGE_VAL[s.STAGE]
            
        elif s.STAGE == "ripe":
            msg = "Cannot move to next phenology stage: maturity already reached!"
            raise Exception(msg)

        else: # Problem no stage defined
            msg = "No STAGE defined in phenology submodule."
            raise Exception(msg)
        
        msg = "Changed phenological stage '%s' to '%s' on %s"

    def _on_DORMANT(self, day:datetime.date):
        """Handler for dormant signal. Reset all nonessential states and rates to 0
        """
        
        s = self.states
        r = self.rates

        s.TSUM  = 0
        s.TSUME = 0

    def get_tensor_output(self):
        """
        Return the phenological stage as the floor value
        """

        #return torch.floor(self.states.DVS) + 1
        return self.states.DVS

    def get_output(self, var: list=None):
        """
        Return the output variables
        """
        output_vars = []
        if var is None:
            for s in self.states._find_valid_variables():
                output_vars.append(getattr(self.states, s))
            for r in self.rates._find_valid_variables():
                output_vars.append(getattr(self.rates, r))
        else:
            for v in var:
                if v in self.states.trait_names():
                    output_vars.append(getattr(self.states, v))
                elif v in self.rates.trait_names():
                    output_vars.append(getattr(self.rates, v))
        return output_vars
    
    def get_output_vars(self):
        """
        Return all output vars
        """
        return list(self.states._find_valid_variables().union(self.rates._find_valid_variables()))
       
    def get_param_dict(self):
        p = self.params
        return {"TBASEM":p.TBASEM,
                "TEFFMX":p.TEFFMX,
                "TSUMEM":p.TSUMEM,
                "TSUM1":p.TSUM1,
                "TSUM2":p.TSUM2,
                "TSUM3":p.TSUM3,
                "TSUM4":p.TSUM4,
                "MLDORM":p.MLDORM,
                "Q10C":p.Q10C,
                "CSUMDB":p.CSUMDB}
    
    def set_model_params(self, args:dict):
        """
        Set the model phenology parameters from dictionary
        """
        p = self.params
        # Phenology Parameters
        if "TBASEM" in args.keys():
            p.TBASEM = args["TBASEM"]
        if "TEFFMX" in args.keys():
            p.TEFFMX = args["TEFFMX"]
        if "TSUMEM" in args.keys():
            p.TSUMEM = args["TSUMEM"]
        if "TSUM1" in args.keys():
            p.TSUM1 = args["TSUM1"]
        if "TSUM2" in args.keys():
            p.TSUM2 = args["TSUM2"]
        if "TSUM3" in args.keys():
            p.TSUM3 = args["TSUM3"]
        if "TSUM4" in args.keys():
            p.TSUM4 = args["TSUM4"]
        if "MLDORM" in args.keys():
            p.MLDORM = args["MLDORM"]
        if "Q10C" in args.keys():
            p.Q10C = args["Q10C"]
        if "CSUMDB" in args.keys():
            p.CSUMDB = args["CSUMDB"]
        # Cold hardiness params
        if "HCINIT" in args.keys():
            p.HCINIT = args["HCINIT"]
        if "HCMIN" in args.keys():
            p.HCMIN = args["HCMIN"]
        if "HCMAX" in args.keys():
            p.HCMAX = args["HCMAX"]
        if "TENDO" in args.keys():
            p.TENDO = args["TENDO"]
        if "TECO" in args.keys():
            p.TECO = args["TECO"]
        if "ENACCLIM" in args.keys():
            p.ENACCLIM = args["ENACCLIM"]
        if "ECACCLIM" in args.keys():
            p.ECACCLIM = args["ECACCLIM"]
        if "ENDEACCLIM" in args.keys():
            p.ENDEACCLIM = args["ENDEACCLIM"]
        if "ECDEACCLIM" in args.keys():
            p.ECDEACCLIM = args["ECDEACCLIM"]
        if "THETA" in args.keys():
            p.THETA = args["THETA"]

    def save_model(self, path:str):
        p = self.params
        args = {"TBASEM":p.TBASEM,
                "TEFFMX":p.TEFFMX,
                "TSUMEM":p.TSUMEM,
                "TSUM1":p.TSUM1,
                "TSUM2":p.TSUM2,
                "TSUM3":p.TSUM3,
                "TSUM4":p.TSUM4,
                "MLDORM":p.MLDORM,
                "Q10C":p.Q10C,
                "CSUMDB":p.CSUMDB,
                # Cold Hardiness parameters
                "HCINIT":p.HCINIT,     
                "HCMIN":p.HCMIN,      
                "HCMAX":p.HCMAX,      
                "TENDO":p.TENDO,     
                "TECO":p.TECO,      
                "ENACCLIM":p.ENACCLIM,  
                "ECACCLIM":p.ECACCLIM,  
                "ENDEACCLIM":p.ENDEACCLIM, 
                "ECDEACCLIM":p.ECDEACCLIM,
                "THETA":p.THETA}
        with open(path, "wb") as fp:
            pickle.dump(args,fp)
        fp.close()

    def get_state_rates(self):
        """
        Return all states and rates
        """
        s = self.states
        r = self.rates
        return [
                # States
                s.PHENOLOGY,   
                s.DVS,   
                s.TSUME,  
                s.TSUM,   
                s.CSUM,    
                # Rates
                r.DTSUME, 
                r.DTSUM,
                r.DVR,    
                r.DCU,     
        ]
    
    def set_state_rates(self, arr):
        """
        Return all states and rates
        """
        s = self.states
        r = self.rates
        # States
        s.PHENOLOGY = arr[0] 
        s.DVS = arr[1] 
        s.TSUME = arr[2] 
        s.TSUM = arr[3] 
        s.CSUM = arr[4] 
        # Set phenological stage
        s.STAGE = [k for k, v in self._STAGE_VAL.items() if v == arr[0]][0]
        # Rates
        r.DTSUME = arr[5] 
        r.DTSUM = arr[6] 
        r.DVR = arr[7] 
        r.DCU = arr[8] 
        
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        p = self.params
        # Define initial states
        DVS, DOP, STAGE = self._get_initial_stage(day)
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=DVS, STAGE=STAGE, 
                                          OP=DOP, CSUM=0., PHENOLOGY=self._STAGE_VAL[STAGE],
                                          )
        
        self.rates = self.RateVariables()