


from collections import namedtuple
from math import exp
from datetime import date

from traitlets_pcse import Bool, Instance

from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from model_engine.models.base_model import TensorModel


class PartioningFactors(namedtuple("partitioning_factors", "FR FL FS FO")):
    """Template for namedtuple containing partitioning factors"""
    pass

class Partitioning_NPK(TensorModel):
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
        PF = Instance(PartioningFactors)

    def __init__(self, day:date, kiosk, parameters:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parameters: dictionary with WOFOST cropdata key/value pairs
        """
        self.params = self.Parameters(parameters)

        
        k = self.kiosk
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        
        PF = PartioningFactors(FR, FL, FS, FO)

        
        self.states = self.StateVariables(kiosk, publish=["FR", "FL", "FS", "FO", "PF"],
                                          FR=FR, FL=FL, FS=FS, FO=FO, PF=PF)
        self._check_partitioning()

    def _check_partitioning(self):
        """Check for partitioning errors.
        """
        FR = self.states.FR
        FL = self.states.FL
        FS = self.states.FS
        FO = self.states.FO
            
    def integrate(self, day:date, delt:float=1.0):
        """
        Update partitioning factors based on development stage (DVS)
        and the Nitrogen nutrition Index (NNI)
        """
        p = self.params
        s = self.states
        k = self.kiosk

        if k.RFTRA < k.NNI:
            
            FRTMOD = max(1., 1./(k.RFTRA + 0.5))
            s.FR = min(0.6, p.FRTB(k.DVS) * FRTMOD)
            s.FL = p.FLTB(k.DVS)
            s.FS = p.FSTB(k.DVS)
            s.FO = p.FOTB(k.DVS)
        else:
            
            FLVMOD = exp(-p.NPART * (1.0 - k.NNI))
            s.FL = p.FLTB(k.DVS) * FLVMOD
            s.FS = p.FSTB(k.DVS) + p.FLTB(k.DVS) - s.FL
            s.FR = p.FRTB(k.DVS)
            s.FO = p.FOTB(k.DVS)
            
        if self._THRESHOLD_N_FLAG:
            
            FLVMOD = 1 / exp(-p.NPART * (1.0 - (self._THRESHOLD_N / p.NTHRESH)))
            s.FO = p.FOTB(k.DVS) * FLVMOD
            s.FL = p.FLTB(k.DVS) + p.FOTB(k.DVS) - s.FO
            s.FS = p.FSTB(k.DVS)
            s.FR = p.FRTB(k.DVS)

        
        s.PF = PartioningFactors(s.FR, s.FL, s.FS, s.FO)

        self._check_partitioning()

    def calc_rates(self, day:date, drv):
        """ Return partitioning factors based on current DVS.
        """
        
        if self.kiosk.SURFACE_N > self.params.NTHRESH:
            self._THRESHOLD_N_FLAG = True
            self._THRESHOLD_N = self.kiosk.SURFACE_N
        else:
            self._THRESHOLD_N_FLAG = False
            self._THRESHOLD_N = 0

        
        return self.states.PF

    def reset(self):
        """Reset states adn rates
        """
        
        k = self.kiosk
        s = self.states
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)
        
        PF = PartioningFactors(FR, FL, FS, FO)

        s.FR=FR
        s.FL=FL
        s.FS=FS
        s.FO=FO
        s.PF=PF
 