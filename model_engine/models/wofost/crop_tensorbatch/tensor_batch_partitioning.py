"""Implementation of model for partitioning in WOFOST

Written by: Will Solow, 2025
"""
from datetime import date
import torch

from traitlets_pcse import Bool

from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from model_engine.models.base_model import BatchTensorModel

class Partitioning_NPK_TensorBatch(BatchTensorModel):
    """Class for assimilate partitioning based on development stage (`DVS`)
    with influence of NPK stress.
    """
    _THRESHOLD_N_FLAG = Bool(False)
    _THRESHOLD_N      = Tensor(0.)

    class Parameters(ParamTemplate):
        FRTB = TensorAfgenTrait()
        FLTB = TensorAfgenTrait()
        FSTB = TensorAfgenTrait()
        FOTB = TensorAfgenTrait()
        NPART = Tensor(-99.)  
        NTHRESH = Tensor(-99.) 
        PTHRESH = Tensor(-99.) 
        KTHRESH = Tensor(-99.) 

    class StateVariables(StatesTemplate):
        FR = Tensor(-99.)
        FL = Tensor(-99.)
        FS = Tensor(-99.)
        FO = Tensor(-99.)
    
    def __init__(self, day:date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        k = self.kiosk
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        self.states = self.StateVariables(kiosk=self.kiosk, publish=["FR", "FL", "FS", "FO"],
                                          FR=FR, FL=FL, FS=FS, FO=FO)
     
    def calc_rates(self, day:date, drv):
        """ Return partitioning factors based on current DVS.
        """
        
        if self.kiosk.SURFACE_N > self.params.NTHRESH:
            self._THRESHOLD_N_FLAG = True
            self._THRESHOLD_N = self.kiosk.SURFACE_N
        else:
            self._THRESHOLD_N_FLAG = False
            self._THRESHOLD_N = 0
     
    def integrate(self, day:date, delt:float=1.0):
        """
        Update partitioning factors based on development stage (DVS)
        and the Nitrogen nutrition Index (NNI)
        """
        p = self.params
        s = self.states
        k = self.kiosk

        if k.RFTRA < k.NNI:
            FRTMOD = torch.max(torch.tensor([1.]).to(self.device), 1. / (k.RFTRA + 0.5) )
            s.FR = min(0.6, p.FRTB(k.DVS) * FRTMOD)
            s.FL = p.FLTB(k.DVS)
            s.FS = p.FSTB(k.DVS)
            s.FO = p.FOTB(k.DVS)
        else:
            FLVMOD = torch.exp(-p.NPART * (1.0 - k.NNI))
            s.FL = p.FLTB(k.DVS) * FLVMOD
            s.FS = p.FSTB(k.DVS) + p.FLTB(k.DVS) - s.FL
            s.FR = p.FRTB(k.DVS)
            s.FO = p.FOTB(k.DVS)
            
        if self._THRESHOLD_N_FLAG:
            FLVMOD = 1 / torch.exp(-p.NPART * (1.0 - (self._THRESHOLD_N / p.NTHRESH)))
            s.FO = p.FOTB(k.DVS) * FLVMOD
            s.FL = p.FLTB(k.DVS) + p.FOTB(k.DVS) - s.FO
            s.FS = p.FSTB(k.DVS)
            s.FR = p.FRTB(k.DVS)

        s._update_kiosk()

    def reset(self, day:date):
        """Reset states adn rates
        """
        
        k = self.kiosk
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        self.states = self.StateVariables(kiosk=self.kiosk, publish=["FR", "FL", "FS", "FO"],
                                          FR=FR, FL=FL, FS=FS, FO=FO)
    
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.FO
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
            return output_vars