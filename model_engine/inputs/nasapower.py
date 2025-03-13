"""NASA POWER weather provider class. Provides global historical weather
data for the past ~40 years

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import requests
import logging
import pathlib
from math import exp

from model_engine.inputs.util import reference_ET, check_angstromAB
from model_engine.inputs.input_providers import SlotPickleMixin, WeatherDataProvider

# Define some lambdas to take care of unit conversions.
MJ_to_J = lambda x: x * 1e6
mm_to_cm = lambda x: x / 10.
tdew_to_hpa = lambda x: ea_from_tdew(x) * 10.
to_date = lambda d: d.date()


def ea_from_tdew(tdew):
    """
    Calculates actual vapour pressure, ea [kPa] from the dewpoint temperature
    using equation (14) in the FAO paper. As the dewpoint temperature is the
    temperature to which air needs to be cooled to make it saturated, the
    actual vapour pressure is the saturation vapour pressure at the dewpoint
    temperature. This method is preferable to calculating vapour pressure from
    minimum temperature.

    Taken from fao_et0.py written by Mark Richards

    Reference:
    Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998) Crop
        evapotranspiration. Guidelines for computing crop water requirements,
        FAO irrigation and drainage paper 56)

    Arguments:
    tdew - dewpoint temperature [deg C]
    """
    # Raise exception:
    if tdew < -95.0 or tdew > 65.0:
        # Are these reasonable bounds?
        msg = 'tdew=%g is not in range -95 to +60 deg C' % tdew
        raise ValueError(msg)

    tmp = (17.27 * tdew) / (tdew + 237.3)
    ea = 0.6108 * exp(tmp)
    return ea
 
class NASAWeatherDataContainer(SlotPickleMixin):
    """Class for storing weather data elements.

    Weather data elements are provided through keywords that are also the
    attribute names under which the variables can accessed in the
    WeatherDataContainer. So the keyword TMAX=15 sets an attribute
    TMAX with value 15.

    The following keywords are compulsory:

    :keyword LAT: Latitude of location (decimal degree)
    :keyword LON: Longitude of location (decimal degree)
    :keyword ELEV: Elevation of location (meters)
    :keyword DAY: the day of observation (python datetime.date)
    :keyword IRRAD: Incoming global radiaiton (J/m2/day)
    :keyword TMIN: Daily minimum temperature (Celsius)
    :keyword TMAX: Daily maximum temperature (Celsius)
    :keyword VAP: Daily mean vapour pressure (hPa)
    :keyword RAIN: Daily total rainfall (cm/day)
    :keyword WIND: Daily mean wind speed at 2m height (m/sec)
    :keyword E0: Daily evaporation rate from open water (cm/day)
    :keyword ES0: Daily evaporation rate from bare soil (cm/day)
    :keyword ET0: Daily evapotranspiration rate from reference crop (cm/day)

    There are two optional keywords arguments:

    :keyword TEMP: Daily mean temperature (Celsius), will otherwise be
                   derived from (TMAX+TMIN)/2.
    :keyword SNOWDEPTH: Depth of snow cover (cm)
    """
    sitevar = ["LAT", "LON", "ELEV"]
    required = ["IRRAD", "TMIN", "TMAX", "VAP", "RAIN", "E0", "ES0", "ET0", "WIND"]
    optional = ["SNOWDEPTH", "TEMP", "TMINRA"]
    # In the future __slots__ can be extended or attribute setting can be allowed
    # by add '__dict__' to __slots__.
    __slots__ = sitevar + required + optional + ["DAY"]

    units = {"IRRAD": "J/m2/day", "TMIN": "Celsius", "TMAX": "Celsius", "VAP": "hPa",
             "RAIN": "cm/day", "E0": "cm/day", "ES0": "cm/day", "ET0": "cm/day",
             "LAT": "Degrees", "LON": "Degrees", "ELEV": "m", "SNOWDEPTH": "cm",
             "TEMP": "Celsius", "TMINRA": "Celsius", "WIND": "m/sec"}

    # ranges for meteorological variables
    ranges = {"LAT": (-90., 90.),
              "LON": (-180., 180.),
              "ELEV": (-300, 6000),
              "IRRAD": (0., 40e6),
              "TMIN": (-50., 60.),
              "TMAX": (-50., 60.),
              "VAP": (0.06, 199.3),  # hPa, computed as sat. vapour pressure at -50, 60 Celsius
              "RAIN": (0, 25),
              "E0": (0., 2.5),
              "ES0": (0., 2.5),
              "ET0": (0., 2.5),
              "WIND": (0., 100.),
              "SNOWDEPTH": (0., 250.),
              "TEMP": (-50., 60.),
              "TMINRA": (-50., 60.)}

    def __init__(self, *args, **kwargs):
        # only keyword parameters should be used for weather data container
        if len(args) > 0:
            msg = ("WeatherDataContainer should be initialized by providing weather " +
                   "variables through keywords only. Got '%s' instead.")
            raise Exception(msg % args)

        # First assign site variables
        for varname in self.sitevar:
            try:
                setattr(self, varname, float(kwargs.pop(varname)))
            except (KeyError, ValueError) as e:
                msg = "Site parameter '%s' missing or invalid when building WeatherDataContainer: %s"
                raise Exception(msg, varname, e)

        # check if we have a DAY element
        if "DAY" not in kwargs:
            msg = "Date of observations 'DAY' not provided when building WeatherDataContainer."
            raise Exception(msg)
        self.DAY = kwargs.pop("DAY")

        # Loop over required arguments to see if all required variables are there
        for varname in self.required:
            value = kwargs.pop(varname, None)
            try:
                setattr(self, varname, float(value))
            except (KeyError, ValueError, TypeError) as e:
                msg = "%s: Weather attribute '%s' missing or invalid numerical value: %s"
                logging.warning(msg, self.DAY, varname, value)

        # Loop over optional arguments
        for varname in self.optional:
            value = kwargs.pop(varname, None)
            if value is None:
                continue
            else:
                try:
                    setattr(self, varname, float(value))
                except (KeyError, ValueError, TypeError) as e:
                    msg = "%s: Weather attribute '%s' missing or invalid numerical value: %s"
                    logging.warning(msg, self.DAY, varname, value)

        # Check for remaining unknown arguments
        if len(kwargs) > 0:
            msg = "WeatherDataContainer: unknown keywords '%s' are ignored!"
            logging.warning(msg, kwargs.keys())

    def __setattr__(self, key, value):

        # Range checking on known meteo variables.
        if key in self.ranges:
            vmin, vmax = self.ranges[key]
            if not vmin <= value <= vmax:
                msg = "Value (%s) for meteo variable '%s' outside allowed range (%s, %s)." % (
                value, key, vmin, vmax)
                raise Exception(msg)
        SlotPickleMixin.__setattr__(self, key, value)

    def add_variable(self, varname, value, unit):
        """Adds an attribute <varname> with <value> and given <unit>

        :param varname: Name of variable to be set as attribute name (string)
        :param value: value of variable (attribute) to be added.
        :param unit: string representation of the unit of the variable. Is
            only use for printing the contents of the WeatherDataContainer.
        """
        if varname not in self.units:
            self.units[varname] = unit
        setattr(self, varname, value)
       
class NASAPowerWeatherDataProvider(WeatherDataProvider):
    """WeatherDataProvider for using the NASA POWER database with PCSE

    :param latitude: latitude to request weather data for
    :param longitude: longitude to request weather data for
    :keyword force_update: Set to True to force to request fresh data
        from POWER website.
    :keyword ETmodel: "PM"|"P" for selecting penman-monteith or Penman
        method for reference evapotranspiration. Defaults to "PM".

    The NASA POWER database is a global database of daily weather data
    specifically designed for agrometeorological applications. The spatial
    resolution of the database is 0.5x0.5 degrees (as of 2018). It is
    derived from weather station observations in combination with satellite
    data for parameters like radiation.

    The weather data is updated with a delay of about 3 months which makes
    the database unsuitable for real-time monitoring, nevertheless the
    POWER database is useful for many other studies and it is a major
    improvement compared to the monthly weather data that were used with
    WOFOST in the past.

    For more information on the NASA POWER database see the documentation
    at: http://power.larc.nasa.gov/common/AgroclimatologyMethodology/Agro_Methodology_Content.html

    The `NASAPowerWeatherDataProvider` retrieves the weather from the
    th NASA POWER API and does the necessary conversions to be compatible
    with PCSE. After the data has been retrieved and stored, the contents
    are dumped to a binary cache file. If another request is made for the
    same location, the cache file is loaded instead of a full request to the
    NASA Power server.

    Cache files are used until they are older then 90 days. After 90 days
    the NASAPowerWeatherDataProvider will make a new request to obtain
    more recent data from the NASA POWER server. If this request fails
    it will fall back to the existing cache file. The update of the cache
    file can be forced by setting `force_update=True`.

    Finally, note that any latitude/longitude within a 0.5x0.5 degrees grid box
    will yield the same weather data, e.g. there is no difference between
    lat/lon 5.3/52.1 and lat/lon 5.1/52.4. Nevertheless slight differences
    in PCSE simulations may occur due to small differences in day length.

    """
    # Variable names in POWER data
    power_variables_old = ["ALLSKY_TOA_SW_DWN", "ALLSKY_SFC_SW_DWN", "T2M", "T2M_MIN",
                       "T2M_MAX", "T2MDEW", "WS2M", "PRECTOT"]
    power_variables = ["TOA_SW_DWN", "ALLSKY_SFC_SW_DWN", "T2M", "T2M_MIN",
                       "T2M_MAX", "T2MDEW", "WS2M", "PRECTOTCORR"]
    # other constants
    HTTP_OK = 200
    angstA = 0.29
    angstB = 0.49

    def __init__(self, latitude, longitude, force_update=False, ETmodel="PM"):

        WeatherDataProvider.__init__(self)

        if latitude < -90 or latitude > 90:
            msg = "Latitude should be between -90 and 90 degrees."
            raise ValueError(msg)
        if longitude < -180 or longitude > 180:
            msg = "Longitude should be between -180 and 180 degrees."
            raise ValueError(msg)

        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.ETmodel = ETmodel
        msg = "Retrieving weather data from NASA Power for lat/lon: (%f, %f)."
        self.logger.info(msg % (self.latitude, self.longitude))

        # Check for existence of a cache file
        cache_file = self._find_cache_file(self.latitude, self.longitude)
        if cache_file is None or force_update is True:
            msg = "No cache file or forced update, getting data from NASA Power."
            self.logger.debug(msg)
            # No cache file, we really have to get the data from the NASA server
            self._get_and_process_NASAPower(self.latitude, self.longitude)
            return

        # get age of cache file, if age < 90 days then try to load it. If loading fails retrieve data
        # from the NASA server .
        r = os.stat(cache_file)
        cache_file_date = dt.date.fromtimestamp(r.st_mtime)
        age = (dt.date.today() - cache_file_date).days
        if age < 90:
            msg = "Start loading weather data from cache file: %s" % cache_file
            self.logger.debug(msg)

            status = self._load_cache_file()
            if status is not True:
                msg = "Loading cache file failed, reloading data from NASA Power."
                self.logger.debug(msg)
                # Loading cache file failed!
                self._get_and_process_NASAPower(self.latitude, self.longitude)
        else:
            # Cache file is too old. Try loading new data from NASA
            try:
                msg = "Cache file older then 90 days, reloading data from NASA Power."
                self.logger.debug(msg)
                self._get_and_process_NASAPower(self.latitude, self.longitude)
            except Exception as e:
                msg = ("Reloading data from NASA failed, reverting to (outdated) " +
                       "cache file")
                self.logger.debug(msg)
                status = self._load_cache_file()
                if status is not True:
                    msg = "Outdated cache file failed loading."
                    raise Exception(msg)

    def _get_and_process_NASAPower(self, latitude, longitude):
        """Handles the retrieval and processing of the NASA Power data
        """
        powerdata = self._query_NASAPower_server(latitude, longitude)
        if not powerdata:
            msg = "Failure retrieving POWER data from server. This can be a connection problem with " \
                  "the NASA POWER server, retry again later."
            raise RuntimeError(msg)

        # Store the informational header then parse variables
        self.description = [powerdata["header"]["title"]]
        self.elevation = float(powerdata["geometry"]["coordinates"][2])
        df_power = self._process_POWER_records(powerdata)

        # Determine Angstrom A/B parameters
        self.angstA, self.angstB = self._estimate_AngstAB(df_power)

        # Convert power records to PCSE compatible structure
        df_pcse = self._POWER_to_PCSE(df_power)

        # Start building the weather data containers
        self._make_WeatherDataContainers(df_pcse.to_dict(orient="records"))

        # dump contents to a cache file
        cache_filename = self._get_cache_filename(latitude, longitude)
        self._dump(cache_filename)

    def _estimate_AngstAB(self, df_power):
        """Determine Angstrom A/B parameters from Top-of-Atmosphere (ALLSKY_TOA_SW_DWN) and
        top-of-Canopy (ALLSKY_SFC_SW_DWN) radiation values.

        :param df_power: dataframe with POWER data
        :return: tuple of Angstrom A/B values

        The Angstrom A/B parameters are determined by dividing swv_dwn by toa_dwn
        and taking the 0.05 percentile for Angstrom A and the 0.98 percentile for
        Angstrom A+B: toa_dwn*(A+B) approaches the upper envelope while
        toa_dwn*A approaches the lower envelope of the records of swv_dwn
        values.
        """

        msg = "Start estimation of Angstrom A/B values from POWER data."
        self.logger.debug(msg)

        # check if sufficient data is available to make a reasonable estimate:
        # As a rule of thumb we want to have at least 200 days available
        if len(df_power) < 200:
            msg = ("Less then 200 days of data available. Reverting to " +
                   "default Angstrom A/B coefficients (%f, %f)")
            self.logger.warn(msg % (self.angstA, self.angstB))
            return self.angstA, self.angstB

        # calculate relative radiation (swv_dwn/toa_dwn) and percentiles
        relative_radiation = df_power.ALLSKY_SFC_SW_DWN/df_power.TOA_SW_DWN
        ix = relative_radiation.notnull()
        angstrom_a = float(np.percentile(relative_radiation[ix].values, 5))
        angstrom_ab = float(np.percentile(relative_radiation[ix].values, 98))
        angstrom_b = angstrom_ab - angstrom_a

        try:
            check_angstromAB(angstrom_a, angstrom_b)
        except Exception as e:
            msg = ("Angstrom A/B values (%f, %f) outside valid range: %s. " +
                   "Reverting to default values.")
            msg = msg % (angstrom_a, angstrom_b, e)
            self.logger.warn(msg)
            return self.angstA, self.angstB

        msg = "Angstrom A/B values estimated: (%f, %f)." % (angstrom_a, angstrom_b)
        self.logger.debug(msg)

        return angstrom_a, angstrom_b

    def _query_NASAPower_server(self, latitude, longitude):
        """Query the NASA Power server for data on given latitude/longitude
        """

        start_date = dt.date(1983,7,1)
        end_date = dt.date.today()

        # build URL for retrieving data, using new NASA POWER api
        server = "https://power.larc.nasa.gov/api/temporal/daily/point"
        payload = {"request": "execute",
                   "parameters": ",".join(self.power_variables),
                   "latitude": latitude,
                   "longitude": longitude,
                   "start": start_date.strftime("%Y%m%d"),
                   "end": end_date.strftime("%Y%m%d"),
                   "community": "AG",
                   "format": "JSON",
                   "user": "anonymous"
                   }
        msg = "Starting retrieval from NASA Power"
        self.logger.debug(msg)
        req = requests.get(server, params=payload)

        if req.status_code != self.HTTP_OK:
            msg = ("Failed retrieving POWER data, server returned HTTP " +
                   "code: %i on following URL %s") % (req.status_code, req.url)
            raise Exception(msg)

        msg = "Successfully retrieved data from NASA Power"
        self.logger.debug(msg)
        return req.json()

    def _find_cache_file(self, latitude, longitude):
        """Try to find a cache file for given latitude/longitude.

        Returns None if the cache file does not exist, else it returns the full path
        to the cache file.
        """
        cache_filename = self._get_cache_filename(latitude, longitude)
        if os.path.exists(cache_filename):
            return cache_filename
        else:
            return None

    def _get_cache_filename(self, latitude, longitude):
        """Constructs the filename used for cache files given latitude and longitude

        The latitude and longitude is coded into the filename by truncating on
        0.1 degree. So the cache filename for a point with lat/lon 52.56/-124.78 will be:
        NASAPowerWeatherDataProvider_LAT00525_LON-1247.cache
        """
        
        user_path = pathlib.Path(__file__).parent.resolve()
        PCSE_USER_HOME = os.path.join(user_path, ".pcse")
        METEO_CACHE_DIR = os.path.join(PCSE_USER_HOME, "meteo_cache")

        fname = "%s_LAT%05i_LON%05i.cache" % (self.__class__.__name__,
                                              int(latitude*10), int(longitude*10))
        cache_filename = os.path.join(METEO_CACHE_DIR, fname)
        return cache_filename

    def _write_cache_file(self):
        """Writes the meteo data from NASA Power to a cache file.
        """
        cache_filename = self._get_cache_filename(self.latitude, self.longitude)
        try:
            self._dump(cache_filename)
        except (IOError, EnvironmentError) as e:
            msg = "Failed to write cache to file '%s' due to: %s" % (cache_filename, e)
            self.logger.warning(msg)

    def _load_cache_file(self):
        """Loads the data from the cache file. Return True if successful.
        """
        cache_filename = self._get_cache_filename(self.latitude, self.longitude)
        try:
            self._load(cache_filename)
            msg = "Cache file successfully loaded."
            self.logger.debug(msg)
            return True
        except (IOError, EnvironmentError, EOFError) as e:
            msg = "Failed to load cache from file '%s' due to: %s" % (cache_filename, e)
            self.logger.warning(msg)
            return False

    def _make_WeatherDataContainers(self, recs):
        """Create a WeatherDataContainers from recs, compute ET and store the WDC's.
        """

        for rec in recs:
            # Reference evapotranspiration in mm/day
            try:
                E0, ES0, ET0 = reference_ET(rec["DAY"], rec["LAT"], rec["ELEV"], rec["TMIN"], rec["TMAX"], rec["IRRAD"],
                                            rec["VAP"], rec["WIND"], self.angstA, self.angstB, self.ETmodel)
            except ValueError as e:
                msg = (("Failed to calculate reference ET values on %s. " % rec["DAY"]) +
                       ("With input values:\n %s.\n" % str(rec)) +
                       ("Due to error: %s" % e))
                raise Exception(msg)

            # update record with ET values value convert to cm/day
            rec.update({"E0": E0/10., "ES0": ES0/10., "ET0": ET0/10.})

            # Build weather data container from dict 't'
            wdc = NASAWeatherDataContainer(**rec)

            # add wdc to dictionary for thisdate
            self._store_WeatherDataContainer(wdc, wdc.DAY)

    def _process_POWER_records(self, powerdata):
        """Process the meteorological records returned by NASA POWER
        """
        msg = "Start parsing of POWER records from URL retrieval."
        self.logger.debug(msg)

        fill_value = float(powerdata["header"]["fill_value"])

        df_power = {}
        for varname in self.power_variables:
            s = pd.Series(powerdata["properties"]["parameter"][varname])
            s[s == fill_value] = np.nan
            df_power[varname] = s
        df_power = pd.DataFrame(df_power)
        df_power["DAY"] = pd.to_datetime(df_power.index, format="%Y%m%d")

        # find all rows with one or more missing values (NaN)
        ix = df_power.isnull().any(axis=1)
        # Get all rows without missing values
        df_power = df_power[~ix]

        return df_power

    def _POWER_to_PCSE(self, df_power):

        # Convert POWER data to a dataframe with PCSE compatible inputs
        df_pcse = pd.DataFrame({"TMAX": df_power.T2M_MAX,
                                "TMIN": df_power.T2M_MIN,
                                "TEMP": df_power.T2M,
                                "IRRAD": df_power.ALLSKY_SFC_SW_DWN.apply(MJ_to_J),
                                "RAIN": df_power.PRECTOTCORR.apply(mm_to_cm),
                                "WIND": df_power.WS2M,
                                "VAP": df_power.T2MDEW.apply(tdew_to_hpa),
                                "DAY": df_power.DAY.apply(to_date),
                                "LAT": self.latitude,
                                "LON": self.longitude,
                                "ELEV": self.elevation})

        return df_pcse
