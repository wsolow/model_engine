"""SimulationObjects implementing |CO2| Assimilation for use with PCSE.

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
from __future__ import print_function
from math import sqrt, exp, cos, pi
from collections import deque
from datetime import date

from traitlets_pcse import Instance
from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
from model_engine.inputs.util import astro

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
    """

    
    XGAUSS = [0.1127017, 0.5000000, 0.8872983]
    WGAUSS = [0.2777778, 0.4444444, 0.2777778]

    
    
    DTGA = 0.
    if (AMAX > 0. and LAI > 0. and DAYL > 0.):
        for i in range(3):
            HOUR   = 12.0+0.5*DAYL*XGAUSS[i]
            SINB   = max(0.,SINLD+COSLD*cos(2.*pi*(HOUR+12.)/24.))
            PAR    = 0.5*AVRAD*SINB*(1.+0.4*SINB)/DSINBE
            PARDIF = min(PAR,SINB*DIFPP)
            PARDIR = PAR-PARDIF
            FGROS = assim(AMAX,EFF,LAI,KDIF,SINB,PARDIR,PARDIF)
            DTGA += FGROS*WGAUSS[i]
    DTGA *= DAYL

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
    """
    
    XGAUSS = [0.1127017, 0.5000000, 0.8872983]
    WGAUSS = [0.2777778, 0.4444444, 0.2777778]

    SCV = 0.2

    
    REFH = (1.-sqrt(1.-SCV))/(1.+sqrt(1.-SCV))
    REFS = REFH*2./(1.+1.6*SINB)
    KDIRBL = (0.5/SINB)*KDIF/(0.8*sqrt(1.-SCV))
    KDIRT = KDIRBL*sqrt(1.-SCV)

    
    FGROS = 0.
    for i in range(3):
        LAIC = LAI*XGAUSS[i]
        
        
        VISDF  = (1.-REFS)*PARDIF*KDIF  *exp(-KDIF  *LAIC)
        VIST   = (1.-REFS)*PARDIR*KDIRT *exp(-KDIRT *LAIC)
        VISD   = (1.-SCV) *PARDIR*KDIRBL*exp(-KDIRBL*LAIC)

        
        VISSHD = VISDF+VIST-VISD
        FGRSH  = AMAX*(1.-exp(-VISSHD*EFF/max(2.0, AMAX)))

        
        
        VISPP  = (1.-SCV)*PARDIR/SINB
        if (VISPP <= 0.):
            FGRSUN = FGRSH
        else:
            FGRSUN = AMAX*(1.-(AMAX-FGRSH) \
                     *(1.-exp(-VISPP*EFF/max(2.0,AMAX)))/ (EFF*VISPP))

        
        
        FSLLA  = exp(-KDIRBL*LAIC)
        FGL    = FSLLA*FGRSUN+(1.-FSLLA)*FGRSH

        
        FGROS += FGL*WGAUSS[i]

    FGROS  = FGROS*LAI
    return FGROS

class WOFOST_Assimilation(TensorModel):
    """Class implementing a WOFOST/SUCROS style assimilation routine including
    effect of changes in atmospheric CO2 concentration.

    """

    _TMNSAV = Instance(deque)

    class Parameters(ParamTemplate):
        AMAXTB = TensorAfgenTrait()
        EFFTB = TensorAfgenTrait()
        KDIFTB = TensorAfgenTrait()
        TMPFTB = TensorAfgenTrait()
        TMNFTB = TensorAfgenTrait()
        CO2AMAXTB = TensorAfgenTrait()
        CO2EFFTB = TensorAfgenTrait()
        CO2 = Tensor(-99.)

    def __init__(self, day:date, kiosk, cropdata:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this Engine instance
        :param cropdata: dictionary with cropdata key/value pairs
        :returns: the assimilation rate using __call__()
        """

        self.params = self.Parameters(cropdata)
        self.kiosk = kiosk
        self._TMNSAV = deque(maxlen=7)

    def __call__(self, day:date, drv):
        """Computes the assimilation of CO2 into the crop
        """
        p = self.params
        k = self.kiosk

        
        DVS = k.DVS
        LAI = k.LAI

        self._TMNSAV.appendleft(drv.TMIN)
        TMINRA = sum(self._TMNSAV)/len(self._TMNSAV)

        DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT = astro(day, drv.LAT, drv.IRRAD)

        AMAX = p.AMAXTB(DVS)
        AMAX = AMAX * p.CO2AMAXTB(p.CO2)
        AMAX = AMAX * p.TMPFTB(drv.DTEMP)
        KDIF = p.KDIFTB(DVS)
        EFF  = p.EFFTB(drv.DTEMP) * p.CO2EFFTB(p.CO2)
        DTGA = totass(DAYL, AMAX, EFF, LAI, KDIF, drv.IRRAD, DIFPP, DSINBE, SINLD, COSLD)

        DTGA *= p.TMNFTB(TMINRA)

        PGASS = DTGA * 30./44.

        return PGASS

    def reset(self):
        """Reset states and rates
        """
        self._TMNSAV = deque(maxlen=7)