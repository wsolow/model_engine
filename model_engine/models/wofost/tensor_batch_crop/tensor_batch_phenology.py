"""Implementation of  models for phenological development in WOFOST

Written by: Will Solow, 2025
"""
import datetime
import torch

from traitlets_pcse import Bool

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

from model_engine.inputs.util import daylength

class Vernalisation_TensorBatch(BatchTensorModel):
    """ Modification of phenological development due to vernalisation.
    """
    _force_vernalisation = Tensor(-99.) # Bool
    _IS_VERNALIZED = Tensor(-99.) # Bool

    class Parameters(ParamTemplate):
        VERNSAT = Tensor(-99.)     
        VERNBASE = Tensor(-99.)    
        VERNRTB = TensorBatchAfgenTrait()   
        VERNDVS = Tensor(-99.)    

    class StateVariables(StatesTemplate):
        VERN = Tensor(-99.)    

    class RateVariables(RatesTemplate):
        VERNR = Tensor(-99.)       
        VERNFAC = Tensor(-99.)     
                             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk,VERN=0.)
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["VERNFAC"])
        
    def calc_rates(self, day:datetime.date, drv, _VEGETATIVE):
        """Compute state rates for integration
        """
        r = self.rates
        s = self.states
        p = self.params

        DVS = self.kiosk.DVS
        r.VERNR = torch.where(_VEGETATIVE,
                    torch.where(~self._IS_VERNALIZED.to(torch.bool), 
                        torch.where(DVS < p.VERNDVS, p.VERNRTB(drv.TEMP), 0.0), 0.0 ), 0.0)
        r.VERNFAC = torch.where(_VEGETATIVE,
                        torch.where(~self._IS_VERNALIZED.to(torch.bool), 
                            torch.where(DVS < p.VERNDVS, torch.clamp((s.VERN - p.VERNBASE)/(p.VERNSAT-p.VERNBASE), \
                                        torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device)), 1.0), 1.0), 1.0)
        # TODO, check that this works with tensors 
        self._force_vernalisation = torch.where(_VEGETATIVE,
                                        torch.where(DVS < p.VERNDVS, self._force_vernalisation, True), self._force_vernalisation)

        self.rates._update_kiosk()

    def integrate(self, day:datetime.date, delt:float=1.0, _VEGETATIVE=None):
        """Integrate state rates
        """
        s = self.states
        r = self.rates
        p = self.params
        
        s.VERN = s.VERN + r.VERNR

        self._IS_VERNALIZED = torch.where(_VEGETATIVE,
                                torch.where(s.VERN >= p.VERNSAT, True,
                                    torch.where(self._force_vernalisation, True, False), self._IS_VERNALIZED), self._IS_VERNALIZED)
                                    # TODO: Check that self.force_vernalization does not just eval to true

        self.states._update_kiosk()

    def reset(self, day:datetime.date):
        """Reset states and rates
        """
        self.states = self.StateVariables(kiosk=self.kiosk,VERN=0.)
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=["VERNFAC"])

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.VERNDVS
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.rates,v)
            return output_vars

    def get_extra_states(self):
        """
        Get extra states
        """
        return {"_force_vernalisation", self._force_vernalisation,
                "_IS_VERNALIZED", self._IS_VERNALIZED}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
        
class WOFOST_Phenology_TensorBatch(BatchTensorModel):
    """Implements the algorithms for phenologic development in WOFOST.
    """

    _STAGE_VAL = {"sowing":0, "emerging":1, "vegetative":2, "reproductive":3, "mature":4, "dead":5}

    class Parameters(ParamTemplate):
        TSUMEM = Tensor(-99.)  
        TBASEM = Tensor(-99.)  
        TEFFMX = Tensor(-99.)  
        TSUM1  = Tensor(-99.)  
        TSUM2  = Tensor(-99.)  
        TSUM3  = Tensor(-99.)  
        IDSL   = Tensor(-99.)  
        DLO    = Tensor(-99.)  
        DLC    = Tensor(-99.)  
        DVSI   = Tensor(-99.)  
        DVSM   = Tensor(-99.)  
        DVSEND = Tensor(-99.)  
        DTSMTB = TensorBatchAfgenTrait() 
                              
        DTBEM  = Tensor(-99)

    class RateVariables(RatesTemplate):
        DTSUME = Tensor(-99.)  
        DTSUM  = Tensor(-99.)  
        DVR    = Tensor(-99.)  
        RDEM   = Tensor(-99.)    

    class StateVariables(StatesTemplate):
        DVS = Tensor(-99.)  
        TSUM = Tensor(-99.)  
        TSUME = Tensor(-99.)  
        DATBE = Tensor(-99)  

    def __init__(self, day:datetime.date, kiosk:dict, parvalues: dict, device, num_models:int=1):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.num_models = num_models
        self.num_stages = len(self._STAGE_VAL)
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        DVS = -0.1
        self._STAGE = ["emerging" for _ in range(self.num_models)]

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["DVS"],\
                                          TSUM=0., TSUME=0., DVS=DVS, DATBE=0)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk)

        if torch.any(self.params.IDSL >= 2):
            self.vernalisation = Vernalisation_TensorBatch(day, kiosk, parvalues, device, num_models=self.num_models)
            
        self.min_tensor = torch.tensor([0.]).to(self.device)

    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        DVRED = 1.
        if torch.any(self.params.IDSL >= 1):
            if hasattr(drv, "DAYL"):
                DAYLP = drv.DAYL
            elif hasattr(drv, "LAT"):
                DAYLP = torch.tensor(daylength(day, drv.LAT)).to(self.device)
            DVRED = torch.clamp(self.min_tensor, torch.tensor([1.]).to(self.device), (DAYLP - p.DLC)/(p.DLO - p.DLC))

        VERNFAC = 1.
        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device) # create masks
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)]) # one hot encoding matrix
        self._sowing, self._emerging, self._vegetative, self._reproductive, self._mature, self._dead = stage_masks # upack for readability
        if torch.any(self.params.IDSL >= 2):
            # if self._STAGE == 'vegetative':
            self.vernalisation.calc_rates(day, drv, self._vegetative)
            VERNFAC = self.kiosk.VERNFAC

        r.RDEM = torch.where(self._sowing, torch.where(drv.TEMP > p.TBASEM, 1, 0), 0)

        r.DTSUME = torch.where(self._emerging, torch.clamp(self.min_tensor, (p.TEFFMX - p.TBASEM), (drv.TEMP - p.TBASEM)), 0)

        r.DTSUM = torch.where(self._sowing | self._emerging | self._dead, 0,  
                        torch.where(self._vegetative, p.DTSMTB(drv.TEMP) * VERNFAC * DVRED, p.DTSMTB(drv.TEMP)))
        
        r.DVR = torch.where(self._sowing | self._dead, 0, 
                        torch.where(self._emerging, 0.1 * r.DTSUME / p.TSUMEM, 
                            torch.where(self._vegetative, r.DTSUM / p.TSUM1, 
                                torch.where(self._reproductive, r.DTSUM / p.TSUM2, 
                                    torch.where(self._mature, r.DTSUM / p.TSUM3, 0)))))

        self.rates._update_kiosk()

    def integrate(self, day, delt=1.0):
        """Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states
        
        if torch.any(self.params.IDSL >= 2):
            self.vernalisation.integrate(day, delt, self._vegetative)

        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        s.TSUM = s.TSUM + r.DTSUM
        s.DATBE = s.DATBE + r.RDEM

        # Stage transitions for "sowing" -> "emerging"
        self._STAGE[(self._sowing & (s.DATBE >= p.DTBEM)).cpu().numpy()] = "emerging"
        # Stage transitions for "emerging" -> "vegetative"
        self._STAGE[(self._emerging & (s.DVS >= 0.0)).cpu().numpy()] = "vegetative"
        # Stage transitions for "vegetative" -> "reproductive"
        self._STAGE[(self._vegetative & (s.DVS >= 1.0)).cpu().numpy()] = "reproductive"
        # Stage transitions for "veraison" -> "ripe"
        self._STAGE[(self._reproductive & (s.DVS >= p.DVSM)).cpu().numpy()] = "mature"
        # Stage transitions for "mature" -> "dead"
        self._STAGE[(self._mature & (s.DVS >= p.DVSEND)).cpu().numpy()] = "dead"

        self.states._update_kiosk()

    def reset(self, day:datetime.date):        
        DVS = -0.1
        self._STAGE = ["emerging" for _ in range(self.num_models)]
        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["DVS"],\
                                          TSUM=0., TSUME=0., DVS=DVS, DATBE=0)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk)

        if torch.any(self.params.IDSL >= 2):
            self.vernalisation.reset(day)

    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return self.states.DVS
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.rates,v)
                elif v in self.kiosk:
                    output_vars[:,i] = getattr(self.kiosk, v)
            return output_vars
        
    def get_extra_states(self):
        """
        Get extra states
        """
        return {"_STAGE": self._STAGE}
    
    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)