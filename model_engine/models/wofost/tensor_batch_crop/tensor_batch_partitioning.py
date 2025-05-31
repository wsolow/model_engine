"""Implementation of model for partitioning in WOFOST

Written by: Will Solow, 2025
"""
from datetime import date
import torch

from traitlets_pcse import Bool

from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from model_engine.models.base_model import BatchTensorModel

class Partitioning_NPK_TensorBatch(BatchTensorModel):
    """Class for assimilate partitioning based on development stage (`DVS`)
    with influence of NPK stress.
    """
    _THRESHOLD_N_FLAG = Tensor(0.0)
    _THRESHOLD_N      = Tensor(0.)

    class Parameters(ParamTemplate):
        FRTB = TensorBatchAfgenTrait()
        FLTB = TensorBatchAfgenTrait()
        FSTB = TensorBatchAfgenTrait()
        FOTB = TensorBatchAfgenTrait()
        NPART = Tensor(-99.)  
        NTHRESH = Tensor(-99.) 
        PTHRESH = Tensor(-99.) 
        KTHRESH = Tensor(-99.) 

    class StateVariables(StatesTemplate):
        FR = Tensor(-99.)
        FL = Tensor(-99.)
        FS = Tensor(-99.)
        FO = Tensor(-99.)
    
    def __init__(self, day:date, kiosk:dict, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        k = self.kiosk
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        self._THRESHOLD_N_FLAG = torch.zeros((self.num_models,)).to(self.device)
        self._THRESHOLD_N = torch.zeros((self.num_models,)).to(self.device)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["FR", "FL", "FS", "FO"],
                                          FR=FR, FL=FL, FS=FS, FO=FO)
     
    def calc_rates(self, day:date, drv, _emerging):
        """ Return partitioning factors based on current DVS.
        """
        
        self._THRESHOLD_N_FLAG = torch.where(self.kiosk.SURFACE_N > self.params.NTHRESH, 1.0, 0.0)
        self._THRESHOLD_N = torch.where(self.kiosk.SURFACE_N > self.params.NTHRESH, self.kiosk.SURFACE_N, 0.0)

        self._THRESHOLD_N_FLAG = torch.where(_emerging, 0.0, self._THRESHOLD_N_FLAG)
        self._THRESHOLD_N      = torch.where(_emerging, 0.0, self._THRESHOLD_N)
     
    def integrate(self, day:date, delt:float=1.0):
        """
        Update partitioning factors based on development stage (DVS)
        and the Nitrogen nutrition Index (NNI)
        """
        p = self.params
        s = self.states
        k = self.kiosk
        
        FRTMOD = torch.max(torch.tensor([1.]).to(self.device), 1. / (k.RFTRA + 0.5) )
        FLVMOD = torch.exp(-p.NPART * (1.0 - k.NNI))

        s.FR = torch.where(k.RFTRA < k.NNI, torch.min(torch.tensor([0.6]).to(self.device), p.FRTB(k.DVS) * FRTMOD), p.FRTB(k.DVS))
        s.FL = torch.where(k.RFTRA < k.NNI, p.FLTB(k.DVS), p.FLTB(k.DVS) * FLVMOD)
        s.FS = torch.where(k.RFTRA < k.NNI, p.FSTB(k.DVS), p.FSTB(k.DVS) + p.FLTB(k.DVS) - s.FL)
        s.FO = p.FOTB(k.DVS)

        FSOMOD = 1 / torch.exp(-p.NPART * (1.0 - (self._THRESHOLD_N / p.NTHRESH)))
        s.FO = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FOTB(k.DVS) * FSOMOD, s.FO)
        s.FL = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FLTB(k.DVS) + p.FOTB(k.DVS) - s.FO, s.FL)
        s.FS = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FSTB(k.DVS), s.FS)
        s.FR = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FRTB(k.DVS), s.FR)

        s._update_kiosk()

    def reset(self, day:date):
        """Reset states adn rates
        """
        
        k = self.kiosk
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        self._THRESHOLD_N_FLAG = torch.zeros((self.num_models,)).to(self.device)
        self._THRESHOLD_N = torch.zeros((self.num_models,)).to(self.device)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["FR", "FL", "FS", "FO"],
                                          FR=FR, FL=FL, FS=FS, FO=FO)
    
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.FO
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
            return output_vars
        
    def get_extra_states(self):
        """
        Get extra states
        """
        return {"_THRESHOLD_N_FLAG", self._THRESHOLD_N_FLAG,
                "_THRESHOLD_N", self._THRESHOLD_N}


    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)