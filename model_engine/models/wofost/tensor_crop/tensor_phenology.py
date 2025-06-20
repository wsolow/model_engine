"""Implementation of  models for phenological development in WOFOST

Written by: Will Solow, 2025
"""
import datetime
import torch

from traitlets_pcse import Bool

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

from model_engine.inputs.util import daylength

class Vernalisation_Tensor(TensorModel):
    """ Modification of phenological development due to vernalisation.
    """
    _force_vernalisation = Bool(False)
    _IS_VERNALIZED = Bool(False)

    class Parameters(ParamTemplate):
        VERNSAT = Tensor(-99.)     
        VERNBASE = Tensor(-99.)    
        VERNRTB = TensorAfgenTrait()   
        VERNDVS = Tensor(-99.)    

    class StateVariables(StatesTemplate):
        VERN = Tensor(-99.)    

    class RateVariables(RatesTemplate):
        VERNR = Tensor(-99.)       
        VERNFAC = Tensor(-99.)     
                             
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        self.states = self.StateVariables(kiosk=self.kiosk,VERN=0.)
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=["VERNFAC"])
        
    def calc_rates(self, day:datetime.date, drv):
        """Compute state rates for integration
        """
        r = self.rates
        s = self.states
        p = self.params

        DVS = self.kiosk.DVS
        if not self._IS_VERNALIZED:
            if DVS < p.VERNDVS:
                r.VERNR = p.VERNRTB(drv.TEMP)
                r.VERNFAC = torch.clamp((s.VERN - p.VERNBASE)/(p.VERNSAT-p.VERNBASE), \
                                        torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device))
            else:
                r.VERNR = 0.
                r.VERNFAC = 1.0
                self._force_vernalisation = True
        else:
            r.VERNR = 0.
            r.VERNFAC = 1.0

        self.rates._update_kiosk()

    def integrate(self, day:datetime.date, delt:float=1.0):
        """Integrate state rates
        """
        s = self.states
        r = self.rates
        p = self.params
        
        s.VERN = s.VERN + r.VERNR
        
        if s.VERN >= p.VERNSAT:  
            self._ISVERNALISED = True

        elif self._force_vernalisation: 
            self._ISVERNALISED = True
        else: 
            self._ISVERNALISED = False

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
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
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
        
class WOFOST_Phenology_Tensor(TensorModel):
    """Implements the algorithms for phenologic development in WOFOST.
    """

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
        DTSMTB = TensorAfgenTrait() 
                              
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

    def __init__(self, day:datetime.date, kiosk:dict, parvalues: dict, device):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        super().__init__(day, kiosk, parvalues, device)

        DVS = -0.1
        self._STAGE = "emerging"
        self.states = self.StateVariables(kiosk=self.kiosk, publish=["DVS"],\
                                          TSUM=0., TSUME=0., DVS=DVS, DATBE=0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk)

        if self.params.IDSL >= 2:
            self.vernalisation = Vernalisation_Tensor(day, kiosk, parvalues, device)

        self.min_tensor = torch.tensor([0.]).to(self.device)

    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        DVRED = 1.
        if p.IDSL >= 1:
            if hasattr(drv, "DAYL"):
                DAYLP = drv.DAYL
            elif hasattr(drv, "LAT"):
                DAYLP = torch.tensor(daylength(day, drv.LAT)).to(self.device)
            DVRED = torch.clamp(self.min_tensor, torch.tensor([1.]).to(self.device), (DAYLP - p.DLC)/(p.DLO - p.DLC))

        VERNFAC = 1.
        if p.IDSL >= 2:
            if self._STAGE == 'vegetative':
                self.vernalisation.calc_rates(day, drv)
                VERNFAC = self.kiosk.VERNFAC

        if self._STAGE == "sowing":
            r.DTSUME = 0.
            r.DTSUM = 0.
            r.DVR = 0.
            if drv.TEMP > p.TBASEM:
                r.RDEM = 1
            else:
                r.RDEM = 0

        elif self._STAGE == "emerging":
            r.DTSUME = torch.clamp(self.min_tensor, (p.TEFFMX - p.TBASEM), (drv.TEMP - p.TBASEM))
            r.DTSUM = 0.
            r.DVR = 0.1 * r.DTSUME / p.TSUMEM
            r.RDEM = 0
        elif self._STAGE == 'vegetative':
            r.DTSUME = 0.
            r.DTSUM = p.DTSMTB(drv.TEMP) * VERNFAC * DVRED
            r.DVR = r.DTSUM / p.TSUM1
            r.RDEM = 0
        elif self._STAGE == 'reproductive':
            r.DTSUME = 0.
            r.DTSUM = p.DTSMTB(drv.TEMP)
            r.DVR = r.DTSUM / p.TSUM2
            r.RDEM = 0
        elif self._STAGE == 'mature':
            r.DTSUME = 0.
            r.DTSUM = p.DTSMTB(drv.TEMP)
            r.DVR = r.DTSUM / p.TSUM3
            r.RDEM = 0
        elif self._STAGE == 'dead':
            r.DTSUME = 0.
            r.DTSUM = 0.
            r.DVR = 0.
            r.RDEM = 0

        self.rates._update_kiosk()

    def integrate(self, day, delt=1.0):
        """Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states
        
        if p.IDSL >= 2:
            if self._STAGE == 'vegetative':
                self.vernalisation.integrate(day, delt)

        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        s.TSUM = s.TSUM + r.DTSUM
        s.DATBE = s.DATBE + r.RDEM

        if self._STAGE == "sowing":
            if s.DATBE >= p.DTBEM:
                self._next_stage(day)
                s.DVS = -0.1
                s.DATBE = 0
        elif self._STAGE == "emerging":
            if s.DVS >= 0.0:
                self._next_stage(day)
        elif self._STAGE == 'vegetative':
            if s.DVS >= 1.0:
                self._next_stage(day)
        elif self._STAGE == 'reproductive':
            if s.DVS >= p.DVSM:
                self._next_stage(day)
        elif self._STAGE == 'mature':
            if s.DVS >= p.DVSEND:
                self._next_stage(day)
        elif self._STAGE == 'dead':
            pass 
        
        self.states._update_kiosk()

    def _next_stage(self, day):
        """Moves stateself._STAGE to the next phenological stage"""
        s = self.states
        p = self.params

        current_STAGE = self._STAGE
        if self._STAGE == "sowing":
            self._STAGE = "emerging"

        elif self._STAGE == "emerging":
            self._STAGE = "vegetative"
            
        elif self._STAGE == "vegetative":
            self._STAGE = "reproductive"  
        elif self._STAGE == "reproductive":
            self._STAGE = "mature"
        elif self._STAGE == "mature":
            self._STAGE = "dead"
        elif self._STAGE == "dead":
            pass

    def reset(self, day:datetime.date):        
        DVS = -0.1
        self._STAGE = "emerging"
        self.states = self.StateVariables(kiosk=self.kiosk, publish=["DVS"],\
                                          TSUM=0., TSUME=0., DVS=DVS, DATBE=0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk)

        if self.params.IDSL >= 2:
            self.vernalisation.reset(day)

    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return self.states.DVS
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
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
        return {"_STAGE": self._STAGE}
    
    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)