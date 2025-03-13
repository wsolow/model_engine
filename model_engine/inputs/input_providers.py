import torch
import numpy as np
import datetime as dt
import logging
import pickle

from model_engine.models.base_model import Model, NumpyModel, BatchNumpyModel, TensorModel, BatchTensorModel
from model_engine.inputs.util import DEVICE

class SlotPickleMixin(object):
    """This mixin makes it possible to pickle/unpickle objects with __slots__ defined.

    In many programs, one or a few classes have a very large number of instances.
    Adding __slots__ to these classes can dramatically reduce the memory footprint
    and improve execution speed by eliminating the instance dictionary. Unfortunately,
    the resulting objects cannot be pickled. This mixin makes such classes pickleable
    again and even maintains compatibility with pickle files created before adding
    __slots__.

    Recipe taken from:
    http://code.activestate.com/recipes/578433-mixin-for-pickling-objects-with-__slots__/
    """

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)


class WeatherDataProvider(object):
    """Base class for all weather data providers.

    """

    # Descriptive items for a WeatherDataProvider
    longitude = None
    latitude = None
    elevation = None
    description = []
    _first_date = None
    _last_date = None
    angstA = None
    angstB = None
    # model used for reference ET
    ETmodel = "PM"

    def __init__(self):
        self.store = {}

    @property
    def logger(self):
        loggername = "%s.%s" % (self.__class__.__module__,
                                self.__class__.__name__)
        return logging.getLogger(loggername)

    def _dump(self, cache_fname):
        """Dumps the contents into cache_fname using pickle.

        Dumps the values of self.store, longitude, latitude, elevation and description
        """
        with open(cache_fname, "wb") as fp:
            dmp = (self.store, self.elevation, self.longitude, self.latitude, self.description, self.ETmodel)
            pickle.dump(dmp, fp, pickle.HIGHEST_PROTOCOL)

    def _load(self, cache_fname):
        """Loads the contents from cache_fname using pickle.

        Loads the values of self.store, longitude, latitude, elevation and description
        from cache_fname and also sets the self.first_date, self.last_date
        """
        with open(cache_fname, "rb") as fp:
            (store, self.elevation, self.longitude, self.latitude, self.description, ETModel) = pickle.load(fp)

        # Check if the reference ET from the cache file is calculated with the same model as
        # specified by self.ETmodel
        if ETModel != self.ETmodel:
            msg = "Mismatch in reference ET from cache file."
            raise Exception(msg)

        self.store.update(store)

    def export(self):
        """Exports the contents of the WeatherDataProvider as a list of dictionaries.

        The results from export can be directly converted to a Pandas dataframe
        which is convenient for plotting or analyses.
        """
        weather_data = []

        days = sorted([r[0] for r in self.store.keys()])
        for day in days:
            wdc = self(day)
            r = {key: getattr(wdc, key) for key in wdc.__slots__ if hasattr(wdc, key)}
            weather_data.append(r)
        return weather_data

    @property
    def first_date(self):
        try:
            self._first_date = min(self.store)[0]
        except ValueError:
            pass
        return self._first_date

    @property
    def last_date(self):
        try:
            self._last_date = max(self.store)[0]
        except ValueError:
            pass
        return self._last_date

    @property
    def missing(self):
        missing = (self.last_date - self.first_date).days - len(self.store) + 1
        return missing

    @property
    def missing_days(self):
        numdays = (self.last_date - self.first_date).days
        all_days = {self.first_date + dt.timedelta(days=i) for i in range(numdays)}
        avail_days = {t[0] for t in self.store.keys()}
        return sorted(all_days - avail_days)

    def check_keydate(self, key):
        """Check representations of date for storage/retrieval of weather data.

        The following formats are supported:

        1. a date object
        2. a datetime object
        3. a string of the format YYYYMMDD
        4. a string of the format YYYYDDD

        Formats 2-4 are all converted into a date object internally.
        """

        import datetime as dt
        if isinstance(key, dt.datetime):
            return key.date()
        elif isinstance(key, dt.date):
            return key
        elif isinstance(key, (str, int)):
            date_formats = {7: "%Y%j", 8: "%Y%m%d", 10: "%Y-%m-%d"}
            skey = str(key).strip()
            l = len(skey)
            if l not in date_formats:
                msg = "Key for WeatherDataProvider not recognized as date: %s"
                raise KeyError(msg % key)

            dkey = dt.datetime.strptime(skey, date_formats[l])
            return dkey.date()
        elif isinstance(key, np.datetime64):
            return key.astype('datetime64[D]').tolist()
        elif isinstance(key, np.ndarray):
            return np.array([self.check_keydate(k) for k in key])
        elif isinstance(key, list):
            return np.array([self.check_keydate(k) for k in key])
        else:
            msg = "Key for WeatherDataProvider not recognized as date: %s"
            raise KeyError(msg % key)

    def _store_WeatherDataContainer(self, wdc, keydate):
        """Stores the WDC under given keydate.
        """
        kd = self.check_keydate(keydate)

        self.store[(kd, 0)] = wdc

    def __call__(self, day, model=Model):
        keydate = self.check_keydate(day)
        msg = "Retrieving weather data for day %s" % keydate
        self.logger.debug(msg)
        try:
            if isinstance(keydate, np.ndarray):
                slots = self.store[keydate[0], 0].__slots__
                if issubclass(model, BatchTensorModel):
                    vals = dict(zip(slots, [np.empty(shape=len(keydate),dtype=object)] + [torch.empty(size=(len(keydate),)).to(DEVICE) for _ in range(len(slots)-1)]))
                elif issubclass(model, BatchNumpyModel):
                    vals = dict(zip(slots, [np.empty(shape=len(keydate),dtype=object)] + [np.empty(shape=(len(keydate),)) for _ in range(len(slots)-1)]))
                else:
                    raise Exception(f"Unexpected Model Type `{model}` with date list")
                for i, key in enumerate(keydate):
                    weather = self.store[key, 0]
                    for s in slots:
                        vals[s][i] = getattr(weather, s)
                if issubclass(model, BatchTensorModel):
                    return DFTensorWeatherDataContainer(**vals)
                elif issubclass(model, BatchNumpyModel):
                    return DFNumpyWeatherDataContainer(**vals)
            else:
                return self.store[(keydate, 0)]
        except KeyError as e:
            msg = "No weather data for %s." % keydate
            raise Exception(msg)


class DFTensorWeatherDataContainer(SlotPickleMixin):
    """
    Class for storing weather data elements.
    """

    # In the future __slots__ can be extended or attribute setting can be allowed
    # by add '__dict__' to __slots__.
    __slots__ = []

    def __init__(self, *args,**kwargs):
        self.__slots__ = list(kwargs.keys())
        # only keyword parameters should be used for weather data container
        if len(args) > 0:
            msg = ("WeatherDataContainer should be initialized by providing weather " +
                   "variables through keywords only. Got '%s' instead.")
            raise Exception(msg % args)
        # Set all attributes
        for k,v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                setattr(self, k, torch.Tensor([v]).to(DEVICE))
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.to(DEVICE))
            else:
                setattr(self, k, v)

    def __setattr__(self, key, value):
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

class DFNumpyWeatherDataContainer(SlotPickleMixin):
    """
    Class for storing weather data elements.
    """

    # In the future __slots__ can be extended or attribute setting can be allowed
    # by add '__dict__' to __slots__.
    __slots__ = []

    def __init__(self, *args,**kwargs):
        self.__slots__ = list(kwargs.keys())
        # only keyword parameters should be used for weather data container
        if len(args) > 0:
            msg = ("WeatherDataContainer should be initialized by providing weather " +
                   "variables through keywords only. Got '%s' instead.")
            raise Exception(msg % args)

        # Set all attributes
        for k,v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                setattr(self, k, np.array([v]))
            elif isinstance(v, np.ndarray):
                setattr(self, k, v)
            else:
                setattr(self, k, v)

    def __setattr__(self, key, value):
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

class DFTensorWeatherDataProvider(WeatherDataProvider):

    def __init__(self, df, force_update=False, ETmodel="PM"):

        WeatherDataProvider.__init__(self)

        self._get_and_process_DF(df)


    def _get_and_process_DF(self, df):
        """
        Handles the retrieval and processing of the NASA Power data
        """

        # Start building the weather data containers
        self._make_WeatherDataContainers(df.to_dict(orient="records"))

    def _make_WeatherDataContainers(self, recs):
        """
        Create a WeatherDataContainers from recs, compute ET and store the WDC's.
        """

        for rec in recs:
            # Build weather data container from dict 't'
            wdc = DFTensorWeatherDataContainer(**rec)

            # add wdc to dictionary for thisdate
            self._store_WeatherDataContainer(wdc, wdc.DAY)

class DFNumpyWeatherDataProvider(WeatherDataProvider):

    def __init__(self, df, force_update=False, ETmodel="PM"):

        WeatherDataProvider.__init__(self)

        self._get_and_process_DF(df)


    def _get_and_process_DF(self, df):
        """
        Handles the retrieval and processing of the NASA Power data
        """

        # Start building the weather data containers
        self._make_WeatherDataContainers(df.to_dict(orient="records"))

    def _make_WeatherDataContainers(self, recs):
        """
        Create a WeatherDataContainers from recs, compute ET and store the WDC's.
        """

        for rec in recs:
            # Build weather data container from dict 't'
            wdc = DFNumpyWeatherDataContainer(**rec)

            # add wdc to dictionary for thisdate
            self._store_WeatherDataContainer(wdc, wdc.DAY)


