"""SimulationObjects implementing |CO2| Assimilation for use with PCSE.

Written by: Will Solow, 2025
"""
from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from model_engine.inputs.util import astro_torch
from model_engine.util import tensor_pop, tensor_appendleft, EPS

def totass(DAYL, AMAX, EFF, LAI, KDIF, AVRAD, DIFPP, DSINBE, SINLD, COSLD):
    """ This routine calculates the daily total gross CO2 assimilation by
    performing a Gaussian integration over time. At three different times of
    the day, irradiance is computed and used to calculate the instantaneous
    canopy assimilation, whereafter integration takes place. More information
    on this routine is given by Spitters et al. (1988).

    FORMAL PARAMETERS:  (I=input,O=output,C=control,IN=init,T=time)
    name   type meaning                                    units  class
    ----   ---- -------                                    -----  -----
    DAYL    R4  Astronomical daylength (base = 0 degrees)     h      I
    AMAX    R4  Assimilation rate at light saturation      kg CO2/   I
                                                          ha leaf/h
    EFF     R4  Initial light use efficiency              kg CO2/J/  I
                                                          ha/h m2 s
    LAI     R4  Leaf area index                             ha/ha    I
    KDIF    R4  Extinction coefficient for diffuse light             I
    AVRAD   R4  Daily shortwave radiation                  J m-2 d-1 I
    DIFPP   R4  Diffuse irradiation perpendicular to direction of
                light                                      J m-2 s-1 I
    DSINBE  R4  Daily total of effective solar height         s      I
    SINLD   R4  Seasonal offset of sine of solar height       -      I
    COSLD   R4  Amplitude of sine of solar height             -      I
    DTGA    R4  Daily total gross assimilation           kg CO2/ha/d O

    Authors: Daniel van Kraalingen
    Date   : April 1991

    Python version:
    Authors: Allard de Wit
    Date   : September 2011
    Modified by: Will Solow
    To support PyTorch Tensors, 2025
    """

    
    XGAUSS = torch.tensor([[0.1127017], [0.5000000], [0.8872983]]).to(LAI.device)
    WGAUSS = torch.tensor([0.2777778, 0.4444444, 0.2777778]).to(LAI.device)
    
    DTGA = torch.zeros((LAI.size(0),)).to(LAI.device)
    HOUR = 12.0 + 0.5 * DAYL * XGAUSS
    SINB   = torch.max(torch.tensor([0.]).to(LAI.device), SINLD + COSLD * torch.cos(2. * torch.pi * (HOUR + 12.) / 24.))
    PAR    = 0.5 * AVRAD * SINB * (1. + 0.4 * SINB ) / DSINBE
    PARDIF = torch.min(PAR, SINB * DIFPP)
    PARDIR = PAR - PARDIF
    FGROS = assim(AMAX,EFF,LAI,KDIF,SINB,PARDIR,PARDIF)
    DTGA = DTGA + torch.sum(FGROS * WGAUSS.unsqueeze(1),dim=0)

    DTGA = torch.where( (AMAX > 0.) & (LAI > 0.) & (DAYL > 0.), DTGA * DAYL, 0.0)

    return DTGA

def assim(AMAX, EFF, LAI, KDIF, SINB, PARDIR, PARDIF):
    """This routine calculates the gross CO2 assimilation rate of
    the whole crop, FGROS, by performing a Gaussian integration
    over depth in the crop canopy. At three different depths in
    the canopy, i.e. for different values of LAI, the
    assimilation rate is computed for given fluxes of photosynthe-
    tically active radiation, whereafter integration over depth
    takes place. More information on this routine is given by
    Spitters et al. (1988). The input variables SINB, PARDIR
    and PARDIF are calculated in routine TOTASS.

    Subroutines and functions called: none.
    Called by routine TOTASS.

    Author: D.W.G. van Kraalingen, 1986

    Python version:
    Allard de Wit, 2011
    Modified by: Will Solow
    To support PyTorch Tensors, 2025
    """
    XGAUSS = torch.tensor([[0.1127017], [0.5000000], [0.8872983]]).to(LAI.device)
    WGAUSS = torch.tensor([[0.2777778], [0.4444444], [0.2777778]]).to(LAI.device)

    SCV = torch.tensor([0.2]).to(LAI.device)

    REFH = (1. - torch.sqrt(1. - SCV)) / (1. + torch.sqrt(1. - SCV))
    REFS = REFH * 2. / (1. + 1.6 * SINB)
    KDIRBL = (0.5 / SINB) * KDIF / (0.8 * torch.sqrt(1.-SCV))
    KDIRT = KDIRBL * torch.sqrt(1. - SCV)

    LAIC = (LAI * XGAUSS).unsqueeze(0).repeat(3,1,1)
    
    PARDIF = PARDIF.unsqueeze(1)
    KDIF = KDIF.unsqueeze(1)
    REFS = REFS.unsqueeze(1)
    KDIRT = KDIRT.unsqueeze(1)
    
    VISDF  = (1. - REFS) * PARDIF * KDIF.transpose(1,0) * torch.exp(-KDIF.transpose(1,0) * LAIC)
    VIST   = (1. - REFS) * PARDIR.unsqueeze(1) * KDIRT * torch.exp(-KDIRT * LAIC)
    VISD   = (1. - SCV) * PARDIR.unsqueeze(1)* KDIRBL.unsqueeze(1) * torch.exp(-KDIRBL.unsqueeze(1) * LAIC)
    VISSHD = VISDF + VIST - VISD

    FGRSH  = AMAX * (1. - torch.exp(-VISSHD * EFF / torch.max(torch.tensor([2.0]).to(LAI.device), AMAX)))
    VISPP  = ((1. - SCV) * PARDIR / SINB).unsqueeze(1)

    FGRSUN = torch.where(VISPP <= 0., AMAX * (1. - torch.exp(-VISSHD * EFF / torch.max(torch.tensor([2.0]).to(LAI.device), AMAX))), 
                AMAX * (1. - (AMAX - FGRSH) \
                     * (1. - torch.exp(-VISPP * EFF / torch.max(torch.tensor([2.0]).to(LAI.device), AMAX))) / (EFF * VISPP+EPS)))
    
    FSLLA  = torch.exp(-KDIRBL.unsqueeze(1) * LAIC)
    FGL    = FSLLA * FGRSUN + (1. - FSLLA) * FGRSH
    FGROS  = torch.sum(FGL * WGAUSS,dim=1) * LAI

    return FGROS

class WOFOST_Assimilation_TensorBatch(BatchTensorModel):
    """Class implementing a WOFOST/SUCROS style assimilation routine including
    effect of changes in atmospheric CO2 concentration.
    """

    _TMNSAV = Tensor(-99.)

    class Parameters(ParamTemplate):
        AMAXTB = TensorBatchAfgenTrait()
        EFFTB = TensorBatchAfgenTrait()
        KDIFTB = TensorBatchAfgenTrait()
        TMPFTB = TensorBatchAfgenTrait()
        TMNFTB = TensorBatchAfgenTrait()
        CO2AMAXTB = TensorBatchAfgenTrait()
        CO2EFFTB = TensorBatchAfgenTrait()
        CO2 = Tensor(-99.)

    class StateVariables(StatesTemplate):
        PGASS = Tensor(-99.)

    def __init__(self, day:date, kiosk, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models 
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        self._TMNSAV = torch.zeros((self.num_models,7)).to(self.device)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["PGASS"], PGASS=0)

    def __call__(self, day:date, drv, _emerging):
        """Computes the assimilation of CO2 into the crop
        """
        p = self.params
        k = self.kiosk

        DVS = k.DVS
        LAI = k.LAI
        
        self._TMNSAV = tensor_appendleft(self._TMNSAV, torch.where(_emerging, 0.0, drv.TMIN))

        TMINRA = torch.sum(self._TMNSAV, dim=1) / self._TMNSAV.size(1)

        DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT = astro_torch(day, drv.LAT, drv.IRRAD)

        AMAX = p.AMAXTB(DVS)
        AMAX = AMAX * p.CO2AMAXTB(p.CO2)
        AMAX = AMAX * p.TMPFTB(drv.TEMP)
        KDIF = p.KDIFTB(DVS)
        EFF  = p.EFFTB(drv.TEMP) * p.CO2EFFTB(p.CO2)
        DTGA = totass(DAYL, AMAX, EFF, LAI, KDIF, drv.IRRAD, DIFPP, DSINBE, SINLD, COSLD)
        DTGA = DTGA * p.TMNFTB(TMINRA)

        self.states.PGASS = DTGA * 30./44.

        # Only run when not emerging
        self.states.PGASS = torch.where(_emerging, 0.0, self.states.PGASS)

        self.states._update_kiosk()

        return self.states.PGASS

    def reset(self, day:date):
        """Reset states and rates
        """
        self._TMNSAV = torch.zeros((self.num_models,7)).to(self.device)
        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["PGASS"], PGASS=0)

    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.states.PGASS
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
        return {"_TMNSAV": self._TMNSAV}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)