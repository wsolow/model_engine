"""Base class for for State Rates and Parameters that each simulation object
in the WOFOST model has.

In general these classes are not to be used directly, but are to be subclassed
when creating PCSE simulation units.

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from traitlets_pcse import (Float, Int, Instance, Bool, HasTraits, TraitType)
import torch
import numpy as np
from collections.abc import Iterable

from bisect import bisect_left

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tensor(TraitType):
    """An AFGEN table trait"""
    default_value = torch.tensor([0.])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj, value):
        if isinstance(value, torch.Tensor):
           return value.to(torch.float32).to(device)
        elif isinstance(value, Iterable):
           return torch.tensor(value,dtype=torch.float32).to(device)
        elif isinstance(value, float):
            return torch.tensor([value],dtype=torch.float32).to(device)
        elif isinstance(value, int):
            return torch.tensor([float(value)],dtype=torch.float32).to(device)
        self.error(obj, value)

class NDArray(TraitType):
    """An AFGEN table trait"""
    default_value = torch.tensor([0.])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj, value):
        if isinstance(value, np.ndarray):
           return value
        elif isinstance(value, Iterable):
           return np.array(value,dtype=object)
        elif isinstance(value, float):
            return np.array([value])
        elif isinstance(value, int):
            return np.array([value])
        self.error(obj, value)

class Afgen(object):
    """Emulates the AFGEN function in WOFOST.
    """
    
    def _check_x_ascending(self, tbl_xy):
        """Checks that the x values are strictly ascending.
        """
        x_list = tbl_xy[0::2]
        y_list = tbl_xy[1::2]
        n = len(x_list)
        
        # Check if x range is ascending continuously
        rng = list(range(1, n))
        x_asc = [True if (x_list[i] > x_list[i-1]) else False for i in rng]
        
        # Check for breaks in the series where the ascending sequence stops.
        # Only 0 or 1 breaks are allowed. Use the XOR operator '^' here
        sum_break = sum([1 if (x0 ^ x1) else 0 for x0,x1 in zip(x_asc, x_asc[1:])])
        if sum_break == 0:
            x = x_list
            y = y_list
        elif sum_break == 1:
            x = [x_list[0]]
            y = [y_list[0]]
            for i,p in zip(rng, x_asc):
                if p is True:
                    x.append(x_list[i])
                    y.append(y_list[i])
        else:
            msg = ("X values for AFGEN input list not strictly ascending: %s"
                   % x_list)
            raise ValueError(msg)
        
        return x, y            

    def __init__(self, tbl_xy):
        
        x_list, y_list = self._check_x_ascending(tbl_xy)
        x_list = self.x_list = list(map(float, x_list))
        y_list = self.y_list = list(map(float, y_list))
        intervals = list(zip(x_list, x_list[1:], y_list, y_list[1:]))
        self.slopes = [(y2 - y1)/(x2 - x1) for x1, x2, y1, y2 in intervals]

    def __call__(self, x):

        if x <= self.x_list[0]:
            return self.y_list[0]
        if x >= self.x_list[-1]:
            return self.y_list[-1]

        i = bisect_left(self.x_list, x) - 1
        v = self.y_list[i] + self.slopes[i] * (x - self.x_list[i])

        return v

class TensorAfgenTrait(TraitType):
    """An AFGEN table trait"""
    default_value = Afgen([0,0,1,1])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj, value):
        if isinstance(value, Afgen):
           return value
        elif isinstance(value, Iterable):
           return Afgen(value)
        self.error(obj, value)


class ParamTemplate(HasTraits):
    """
    Template for storing parameter values.
    """

    def __init__(self, parvalues:dict, num_models:int=None):
        """Initialize parameter template
        Args:
            parvalues - parameter values to include 
        """
        HasTraits.__init__(self)

        for parname in self.trait_names():
            # Check if the parname is available in the dictionary of parvalues
            if parname not in parvalues:
                msg = "Value for parameter %s missing." % parname
                raise Exception(msg)
            if num_models is None:
                value = parvalues[parname]
            else: 
                value = np.tile(parvalues[parname], num_models).astype(np.float32)
            # Single value parameter
            setattr(self, parname, value)

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            HasTraits.__setattr__(self, attr, value)
        elif hasattr(self, attr):
            HasTraits.__setattr__(self, attr, value)
        else:
            msg = "Assignment to non-existing attribute '%s' prevented." % attr
            raise Exception(msg)
        
    def __str__(self):
        string = f""
        for parname in self.trait_names():
            string += f"{parname}: {getattr(self, parname)}\n"
        return string
    
class StatesRatesCommon(HasTraits):
    """
    Base class for States/Rates Templates. Includes all commonalitities
    between the two templates
    """
    _valid_vars = Instance(set)

    def __init__(self):
        """Set up the common stuff for the states and rates template
        including variables that have to be published in the kiosk
        """

        HasTraits.__init__(self)

        # Determine the rate/state attributes defined by the user
        self._valid_vars = self._find_valid_variables()

    def _find_valid_variables(self):
        """
        Returns a set with the valid state/rate variables names. Valid rate
        variables have names not starting with 'trait' or '_'.
        """

        valid = lambda s: not (s.startswith("_") or s.startswith("trait"))
        r = [name for name in self.trait_names() if valid(name)]
        return set(r)
    
    def __str__(self):
        string = f""
        for parname in self.trait_names():
            string += f"{parname}: {getattr(self, parname)}\n"
        return string


class StatesTemplate(StatesRatesCommon):
    """
    Takes care of assigning initial values to state variables
    and monitoring assignments to variables that are published.
    """

    def __init__(self, num_models:int=None, **kwargs):
        """Initialize the StatesTemplate class
        
        Args:
            kiosk - VariableKiosk to handle default parameters
        """
        StatesRatesCommon.__init__(self)

        # set initial state value
        for attr in self._valid_vars:
            if attr in kwargs:
                value = kwargs.pop(attr)
                if num_models is None:
                    setattr(self, attr, value)
                else:
                    setattr(self, attr, np.tile(value, num_models).astype(np.float32))
            else:
                msg = "Initial value for state %s missing." % attr
                raise Exception(msg)

class RatesTemplate(StatesRatesCommon):
    """
    Takes care of registering variables in the kiosk and monitoring
    assignments to variables that are published.
    """

    def __init__(self, num_models:int=None):
        """Set up the RatesTemplate and set monitoring on variables that
        have to be published.
        """
        self.num_models = num_models
        StatesRatesCommon.__init__(self)

        # Determine the zero value for all rate variable if possible
        self._rate_vars_zero = self._find_rate_zero_values()

        # Initialize all rate variables to zero or False
        self.zerofy()

    def _find_rate_zero_values(self):
        """Returns a dict with the names with the valid rate variables names as keys and
        the values are the zero values used by the zerofy() method. This means 0 for Int,
        0.0 for Float en False for Bool.
        """

        # Define the zero value for Float, Int and Bool
        if self.num_models is None:
            tensor = torch.tensor([0.]).to(device)
        else:
            tensor = torch.tensor(np.tile(0., self.num_models).astype(np.float32)).to(device)
        zero_value = {Bool: False, Int: 0, Float: 0., Tensor: tensor}

        d = {}
        for name, value in self.traits().items():
            if name not in self._valid_vars:
                continue
            try:
                d[name] = zero_value[value.__class__]
            except KeyError:
                continue
        return d

    def zerofy(self):
        """
        Sets the values of all rate values to zero (Int, Float)
        or False (Boolean).
        """
        self._trait_values.update(self._rate_vars_zero)