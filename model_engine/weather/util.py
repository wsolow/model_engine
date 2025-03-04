import datetime
from math import cos, sin, asin, sqrt, exp, pi, radians
from collections import namedtuple
import numpy as np

# Named tuple for returning results of ASTRO

astro_nt = namedtuple("AstroResults", "DAYL, DAYLP, SINLD, COSLD, DIFPP, "
                                      "ATMTR, DSINBE, ANGOT")

""" Used in the Crop Phenology module"""

def limit(vmin:float, vmax:float, v:float):
    """
    limits the range of v between min and max
    """

    if vmin > vmax:
        raise RuntimeError("Min value (%f) larger than max (%f)" % (vmin, vmax))
    
    if v < vmin:       # V below range: return min
        return vmin
    elif v < vmax:     # v within range: return v
        return v
    else:             # v above range: return max
        return vmax


def doy(day):
        """Converts a date or datetime object to day-of-year (Jan 1st = doy 1)
        """
        # Check if day is a date or datetime object
        if isinstance(day, (datetime.date, datetime.datetime)):
            return day.timetuple().tm_yday
        elif isinstance(day, np.datetime64):
            return day.astype('datetime64[D]').tolist().timetuple().tm_yday
        else:
            msg = "Parameter day is not a date or datetime object."
            raise RuntimeError(msg)
        
def astro(day, latitude, radiation, _cache={}):
    """python version of ASTRO routine by Daniel van Kraalingen.
    
    This subroutine calculates astronomic daylength, diurnal radiation
    characteristics such as the atmospheric transmission, diffuse radiation etc.

    :param day:         date/datetime object
    :param latitude:    latitude of location
    :param radiation:   daily global incoming radiation (J/m2/day)

    output is a `namedtuple` in the following order and tags::

        DAYL      Astronomical daylength (base = 0 degrees)     h      
        DAYLP     Astronomical daylength (base =-4 degrees)     h      
        SINLD     Seasonal offset of sine of solar height       -      
        COSLD     Amplitude of sine of solar height             -      
        DIFPP     Diffuse irradiation perpendicular to
                  direction of light                         J m-2 s-1 
        ATMTR     Daily atmospheric transmission                -      
        DSINBE    Daily total of effective solar height         s
        ANGOT     Angot radiation at top of atmosphere       J m-2 d-1
 
    Authors: Daniel van Kraalingen
    Date   : April 1991
 
    Python version
    Author      : Allard de Wit
    Date        : January 2011
    """

    # Check for range of latitude
    if abs(latitude) > 90.:
        msg = "Latitude not between -90 and 90"
        raise RuntimeError(msg)
    LAT = latitude
        
    # Determine day-of-year (IDAY) from day
    IDAY = doy(day)
    
    # reassign radiation
    AVRAD = radiation

    # Test if variables for given (day, latitude, radiation) were already calculated
    # in a previous run. If not (e.g. keysError) calculate the variables, store
    # in cache and return the value.
    try:
        return _cache[(IDAY, LAT, AVRAD)]
    except KeyError:
        pass

    # constants
    RAD = radians(1.)
    ANGLE = -4.

    # Declination and solar constant for this day
    DEC = -asin(sin(23.45*RAD)*cos(2.*pi*(float(IDAY)+10.)/365.))
    SC  = 1370.*(1.+0.033*cos(2.*pi*float(IDAY)/365.))

    # calculation of daylength from intermediate variables
    # SINLD, COSLD and AOB
    SINLD = sin(RAD*LAT)*sin(DEC)
    COSLD = cos(RAD*LAT)*cos(DEC)
    AOB = SINLD/COSLD

    # For very high latitudes and days in summer and winter a limit is
    # inserted to avoid math errors when daylength reaches 24 hours in 
    # summer or 0 hours in winter.

    # Calculate solution for base=0 degrees
    if abs(AOB) <= 1.0:
        DAYL  = 12.0*(1.+2.*asin(AOB)/pi)
        # integrals of sine of solar height
        DSINB  = 3600.*(DAYL*SINLD+24.*COSLD*sqrt(1.-AOB**2)/pi)
        DSINBE = 3600.*(DAYL*(SINLD+0.4*(SINLD**2+COSLD**2*0.5))+
                 12.*COSLD*(2.+3.*0.4*SINLD)*sqrt(1.-AOB**2)/pi)
    else:
        if AOB >  1.0: DAYL = 24.0
        if AOB < -1.0: DAYL = 0.0
        # integrals of sine of solar height	
        DSINB = 3600.*(DAYL*SINLD)
        DSINBE = 3600.*(DAYL*(SINLD+0.4*(SINLD**2+COSLD**2*0.5)))

    # Calculate solution for base=-4 (ANGLE) degrees
    AOB_CORR = (-sin(ANGLE*RAD)+SINLD)/COSLD
    if abs(AOB_CORR) <= 1.0:
        DAYLP = 12.0*(1.+2.*asin(AOB_CORR)/pi)
    elif AOB_CORR > 1.0:
        DAYLP = 24.0
    elif AOB_CORR < -1.0:
        DAYLP = 0.0

    # extraterrestrial radiation and atmospheric transmission
    ANGOT = SC*DSINB
    # Check for DAYL=0 as in that case the angot radiation is 0 as well
    if DAYL > 0.0:
        ATMTR = AVRAD/ANGOT
    else:
        ATMTR = 0.

    # estimate fraction diffuse irradiation
    if ATMTR > 0.75:
        FRDIF = 0.23
    elif (ATMTR <= 0.75) and (ATMTR > 0.35):
        FRDIF = 1.33-1.46*ATMTR
    elif (ATMTR <= 0.35) and (ATMTR > 0.07):
        FRDIF = 1.-2.3*(ATMTR-0.07)**2
    else:  # ATMTR <= 0.07
        FRDIF = 1.

    DIFPP = FRDIF*ATMTR*0.5*SC

    retvalue = astro_nt(DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT)
    _cache[(IDAY, LAT, AVRAD)] = retvalue

    return retvalue

def daylength(day, latitude, angle=-4, _cache={}):
    """Calculates the daylength for a given day, altitude and base.

    :param day:         date/datetime object
    :param latitude:    latitude of location
    :param angle:       The photoperiodic daylength starts/ends when the sun
        is `angle` degrees under the horizon. Default is -4 degrees.
    
    Derived from the WOFOST routine ASTRO.FOR and simplified to include only
    daylength calculation. Results are being cached for performance
    """
    #from unum.units import h

    # Check for range of latitude
    if abs(latitude) > 90.:
        msg = "Latitude not between -90 and 90"
        raise RuntimeError(msg)
    
    # Calculate day-of-year from date object day
    IDAY = doy(day)
    
    # Test if daylength for given (day, latitude, angle) was already calculated
    # in a previous run. If not (e.g. keysError) calculate the daylength, store
    # in cache and return the value.
    try:
        return _cache[(IDAY, latitude, angle)]
    except KeyError:
        pass
    
    # constants
    RAD = radians(1.)

    # calculate daylength
    ANGLE = angle
    LAT = latitude
    DEC = -asin(sin(23.45*RAD)*cos(2.*pi*(float(IDAY)+10.)/365.))
    SINLD = sin(RAD*LAT)*sin(DEC)
    COSLD = cos(RAD*LAT)*cos(DEC)
    AOB = (-sin(ANGLE*RAD)+SINLD)/COSLD

    # daylength
    if abs(AOB) <= 1.0:
        DAYLP = 12.0*(1.+2.*asin((-sin(ANGLE*RAD)+SINLD)/COSLD)/pi)
    elif AOB > 1.0:
        DAYLP = 24.0
    else:
        DAYLP =  0.0

    # store results in cache
    _cache[(IDAY, latitude, angle)] = DAYLP
    
    return DAYLP

""" Used for NASA POWER the first time that a location is loaded"""

Celsius2Kelvin = lambda x: x + 273.16
hPa2kPa = lambda x: x/10.

# Saturated Vapour pressure [kPa] at temperature temp [C]
SatVapourPressure = lambda temp: 0.6108 * exp((17.27 * temp) / (237.3 + temp))

def reference_ET(DAY, LAT, ELEV, TMIN, TMAX, IRRAD, VAP, WIND,
                 ANGSTA, ANGSTB, ETMODEL="PM", **kwargs):
    """Calculates reference evapotranspiration values E0, ES0 and ET0.

    The open water (E0) and bare soil evapotranspiration (ES0) are calculated with
    the modified Penman approach, while the references canopy evapotranspiration is
    calculated with the modified Penman or the Penman-Monteith approach, the latter
    is the default.

    Input variables::

        DAY     -  Python datetime.date object                      -
        LAT     -  Latitude of the site                          degrees
        ELEV    -  Elevation above sea level                        m
        TMIN    -  Minimum temperature                              C
        TMAX    -  Maximum temperature                              C
        IRRAD   -  Daily shortwave radiation                     J m-2 d-1
        VAP     -  24-hour average vapour pressure                 hPa
        WIND    -  24-hour average windspeed at 2 meter            m/s
        ANGSTA  -  Empirical constant in Angstrom formula           -
        ANGSTB  -  Empirical constant in Angstrom formula           -
        ETMODEL -  Indicates if the canopy reference ET should     PM|P
                   be calculated with the Penman-Monteith method
                   (PM) or the modified Penman method (P)

    Output is a tuple (E0, ES0, ET0)::

        E0      -  Penman potential evaporation from a free
                   water surface [mm/d]
        ES0     -  Penman potential evaporation from a moist
                   bare soil surface [mm/d]
        ET0     -  Penman or Penman-Monteith potential evapotranspiration from a
                   crop canopy [mm/d]

.. note:: The Penman-Monteith algorithm is valid only for a reference canopy, and
    therefore it is not used to calculate the reference values for bare soil and
    open water (ES0, E0).

    The background is that the Penman-Monteith model is basically a surface
    energy balance where the net solar radiation is partitioned over latent and
    sensible heat fluxes (ignoring the soil heat flux). To estimate this
    partitioning, the PM method makes a connection between the surface
    temperature and the air temperature. However, the assumptions
    underlying the PM model are valid only when the surface where this
    partitioning takes place is the same for the latent and sensible heat
    fluxes.

    For a crop canopy this assumption is valid because the leaves of the
    canopy form the surface where both latent heat flux (through stomata)
    and sensible heat flux (through leaf temperature) are partitioned.
    For a soil, this principle does not work because when the soil is
    drying the evaporation front will quickly disappear below the surface
    and therefore the assumption that the partitioning surface is the
    same does not hold anymore.

    For water surfaces, the assumptions underlying PM do not hold
    because there is no direct relationship between the temperature
    of the water surface and the net incoming radiation as radiation is
    absorbed by the water column and the temperature of the water surface
    is co-determined by other factors (mixing, etc.). Only for a very
    shallow layer of water (1 cm) the PM methodology could be applied.

    For bare soil and open water the Penman model is preferred. Although it
    partially suffers from the same problems, it is calibrated somewhat
    better for open water and bare soil based on its empirical wind
    function.

    Finally, in crop simulation models the open water evaporation and
    bare soil evaporation only play a minor role (pre-sowing conditions
    and flooded rice at early stages), it is not worth investing much
    effort in improved estimates of reference value for E0 and ES0.
    """
    if ETMODEL not in ["PM", "P"]:
        msg = "Variable ETMODEL can have values 'PM'|'P' only."
        raise RuntimeError(msg)

    E0, ES0, ET0 = penman(DAY, LAT, ELEV, TMIN, TMAX, IRRAD, VAP, WIND,
                          ANGSTA, ANGSTB)
    if ETMODEL == "PM":
        ET0 = penman_monteith(DAY, LAT, ELEV, TMIN, TMAX, IRRAD, VAP, WIND)

    return E0, ES0, ET0

def penman(DAY, LAT, ELEV, TMIN, TMAX, AVRAD, VAP, WIND2, ANGSTA, ANGSTB):
    """Calculates E0, ES0, ET0 based on the Penman model.
    
     This routine calculates the potential evapo(transpi)ration rates from
     a free water surface (E0), a bare soil surface (ES0), and a crop canopy
     (ET0) in mm/d. For these calculations the analysis by Penman is followed
     (Frere and Popov, 1979;Penman, 1948, 1956, and 1963).
     Subroutines and functions called: ASTRO, LIMIT.

    Input variables::
    
        DAY     -  Python datetime.date object                                    -
        LAT     -  Latitude of the site                        degrees   
        ELEV    -  Elevation above sea level                      m      
        TMIN    -  Minimum temperature                            C
        TMAX    -  Maximum temperature                            C      
        AVRAD   -  Daily shortwave radiation                   J m-2 d-1 
        VAP     -  24-hour average vapour pressure               hPa
        WIND2   -  24-hour average windspeed at 2 meter          m/s
        ANGSTA  -  Empirical constant in Angstrom formula         -
        ANGSTB  -  Empirical constant in Angstrom formula         -

    Output is a tuple (E0,ES0,ET0)::
    
        E0      -  Penman potential evaporation from a free water surface [mm/d]
        ES0     -  Penman potential evaporation from a moist bare soil surface [mm/d]
        ET0     -  Penman potential transpiration from a crop canopy [mm/d]
    """
    # psychrometric instrument constant (mbar/Celsius-1)
    # albedo for water surface, soil surface and canopy
    # latent heat of evaporation of water (J/kg=J/mm)
    # Stefan Boltzmann constant (in J/m2/d/K4, e.g multiplied by 24*60*60)
    PSYCON = 0.67; REFCFW = 0.05; REFCFS = 0.15; REFCFC = 0.25
    LHVAP = 2.45E6; STBC =  5.670373E-8 * 24*60*60 # (=4.9E-3)

    # preparatory calculations
    # mean daily temperature and temperature difference (Celsius)
    # coefficient Bu in wind function, dependent on temperature
    # difference
    TMPA = (TMIN+TMAX)/2.
    TDIF = TMAX - TMIN
    BU = 0.54 + 0.35 * limit(0.,1.,(TDIF-12.)/4.)

    # barometric pressure (mbar)
    # psychrometric constant (mbar/Celsius)
    PBAR = 1013.*exp(-0.034*ELEV/(TMPA+273.))
    GAMMA = PSYCON*PBAR/1013.

    # saturated vapour pressure according to equation of Goudriaan
    # (1977) derivative of SVAP with respect to temperature, i.e. 
    # slope of the SVAP-temperature curve (mbar/Celsius);
    # measured vapour pressure not to exceed saturated vapour pressure

    SVAP = 6.10588 * exp(17.32491*TMPA/(TMPA+238.102))
    DELTA = 238.102*17.32491*SVAP/(TMPA+238.102)**2
    VAP = min(VAP, SVAP)

    # the expression n/N (RELSSD) from the Penman formula is estimated
    # from the Angstrom formula: RI=RA(A+B.n/N) -> n/N=(RI/RA-A)/B,
    # where RI/RA is the atmospheric transmission obtained by a CALL
    # to ASTRO:

    r = astro(DAY, LAT, AVRAD)
    RELSSD = limit(0., 1., (r.ATMTR-abs(ANGSTA))/abs(ANGSTB))

    # Terms in Penman formula, for water, soil and canopy

    # net outgoing long-wave radiation (J/m2/d) acc. to Brunt (1932)
    RB = STBC*(TMPA+273.)**4*(0.56-0.079*sqrt(VAP))*(0.1+0.9*RELSSD)

    # net absorbed radiation, expressed in mm/d
    RNW = (AVRAD*(1.-REFCFW)-RB)/LHVAP
    RNS = (AVRAD*(1.-REFCFS)-RB)/LHVAP
    RNC = (AVRAD*(1.-REFCFC)-RB)/LHVAP

    # evaporative demand of the atmosphere (mm/d)
    EA  = 0.26 * max(0.,(SVAP-VAP)) * (0.5+BU*WIND2)
    EAC = 0.26 * max(0.,(SVAP-VAP)) * (1.0+BU*WIND2)

    # Penman formula (1948)
    E0  = (DELTA*RNW+GAMMA*EA)/(DELTA+GAMMA)
    ES0 = (DELTA*RNS+GAMMA*EA)/(DELTA+GAMMA) 
    ET0 = (DELTA*RNC+GAMMA*EAC)/(DELTA+GAMMA)

    # Ensure reference evaporation >= 0.
    E0  = max(0., E0)
    ES0 = max(0., ES0)
    ET0 = max(0., ET0)
    
    return E0, ES0, ET0

def penman_monteith(DAY, LAT, ELEV, TMIN, TMAX, AVRAD, VAP, WIND2):
    """Calculates reference ET0 based on the Penman-Monteith model.

     This routine calculates the potential evapotranspiration rate from
     a reference crop canopy (ET0) in mm/d. For these calculations the
     analysis by FAO is followed as laid down in the FAO publication
     `Guidelines for computing crop water requirements - FAO Irrigation
     and drainage paper 56 <http://www.fao.org/docrep/X0490E/x0490e00.htm#Contents>`_

    Input variables::

        DAY   -  Python datetime.date object                   -
        LAT   -  Latitude of the site                        degrees
        ELEV  - Elevation above sea level                      m
        TMIN  - Minimum temperature                            C
        TMAX  - Maximum temperature                            C
        AVRAD - Daily shortwave radiation                   J m-2 d-1
        VAP   - 24-hour average vapour pressure               hPa
        WIND2 - 24-hour average windspeed at 2 meter          m/s

    Output is:

        ET0   - Penman-Monteith potential transpiration
                rate from a crop canopy                     [mm/d]
    """

    # psychrometric instrument constant (kPa/Celsius)
    PSYCON = 0.665
    # albedo and surface resistance [sec/m] for the reference crop canopy
    REFCFC = 0.23; CRES = 70.
    # latent heat of evaporation of water [J/kg == J/mm] and
    LHVAP = 2.45E6
    # Stefan Boltzmann constant (J/m2/d/K4, e.g multiplied by 24*60*60)
    STBC = 4.903E-3
    # Soil heat flux [J/m2/day] explicitly set to zero
    G = 0.

    # mean daily temperature (Celsius)
    TMPA = (TMIN+TMAX)/2.

    # Vapour pressure to kPa
    VAP = hPa2kPa(VAP)

    # atmospheric pressure at standard temperature of 293K (kPa)
    T = 293.0
    PATM = 101.3 * pow((T - (0.0065*ELEV))/T, 5.26)

    # psychrometric constant (kPa/Celsius)
    GAMMA = PSYCON * PATM * 1.0E-3

    # Derivative of SVAP with respect to mean temperature, i.e.
    # slope of the SVAP-temperature curve (kPa/Celsius);
    SVAP_TMPA = SatVapourPressure(TMPA)
    DELTA = (4098. * SVAP_TMPA)/pow((TMPA + 237.3), 2)

    # Daily average saturated vapour pressure [kPa] from min/max temperature
    SVAP_TMAX = SatVapourPressure(TMAX)
    SVAP_TMIN = SatVapourPressure(TMIN)
    SVAP = (SVAP_TMAX + SVAP_TMIN) / 2.

    # measured vapour pressure not to exceed saturated vapour pressure
    VAP = min(VAP, SVAP)

    # Longwave radiation according at Tmax, Tmin (J/m2/d)
    # and preliminary net outgoing long-wave radiation (J/m2/d)
    STB_TMAX = STBC * pow(Celsius2Kelvin(TMAX), 4)
    STB_TMIN = STBC * pow(Celsius2Kelvin(TMIN), 4)
    RNL_TMP = ((STB_TMAX + STB_TMIN) / 2.) * (0.34 - 0.14 * sqrt(VAP))

    # Clear Sky radiation [J/m2/DAY] from Angot TOA radiation
    # the latter is found through a call to astro()
    r = astro(DAY, LAT, AVRAD)
    CSKYRAD = (0.75 + (2e-05 * ELEV)) * r.ANGOT

    if CSKYRAD > 0:
        # Final net outgoing longwave radiation [J/m2/day]
        RNL = RNL_TMP * (1.35 * (AVRAD/CSKYRAD) - 0.35)

        # radiative evaporation equivalent for the reference surface
        # [mm/DAY]
        RN = ((1-REFCFC) * AVRAD - RNL)/LHVAP

        # aerodynamic evaporation equivalent [mm/day]
        EA = ((900./(TMPA + 273)) * WIND2 * (SVAP - VAP))

        # Modified psychometric constant (gamma*)[kPa/C]
        MGAMMA = GAMMA * (1. + (CRES/208.*WIND2))

        # Reference ET in mm/day
        ET0 = (DELTA * (RN-G))/(DELTA + MGAMMA) + (GAMMA * EA)/(DELTA + MGAMMA)
        ET0 = max(0., ET0)
    else:
        ET0 = 0.

    return ET0

def check_angstromAB(xA, xB):
    """Routine checks validity of Angstrom coefficients.
    
    This is the  python version of the FORTRAN routine 'WSCAB' in 'weather.for'.
    """
    MIN_A = 0.1
    MAX_A = 0.4
    MIN_B = 0.3
    MAX_B = 0.7
    MIN_SUM_AB = 0.6
    MAX_SUM_AB = 0.9
    A = abs(xA)
    B = abs(xB)
    SUM_AB = A + B
    if A < MIN_A or A > MAX_A:
        msg = "invalid Angstrom A value!"
        raise Exception(msg)
    if B < MIN_B or B > MAX_B:
        msg = "invalid Angstrom B value!"
        raise Exception(msg)
    if SUM_AB < MIN_SUM_AB or SUM_AB > MAX_SUM_AB:
        msg = "invalid sum of Angstrom A & B values!"
        raise Exception(msg)

    return [A,B]
           