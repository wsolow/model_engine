
"""
Initial entry point for the grape model
"""

# Import first to avoid circular imports
from . import util
from . import states_rates
import os

import pathlib
user_path = pathlib.Path(__file__).parent.resolve()

# Make .pcse cache folder in the current working directory
pcse_user_home = os.path.join(user_path, ".pcse")
os.makedirs(pcse_user_home,exist_ok=True)

# Make folder in .pcse for weather data
meteo_cache_dir = os.path.join(pcse_user_home, "meteo_cache")
os.makedirs(meteo_cache_dir,exist_ok=True)


