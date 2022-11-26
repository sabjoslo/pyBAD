import configparser
import os

CONFIG_FILE = os.path.abspath(f"{os.path.dirname(__file__)}/../settings")

configParser = configparser.RawConfigParser()
configParser.read(CONFIG_FILE)

# Paths
paths = dict(configParser.items("PATHS"))

# Miscellaneous settings
settings = dict(configParser.items("SETTINGS"))
settings["nopython"] = configParser["SETTINGS"].getboolean("nopython")
settings["rtol"] = configParser["SETTINGS"].getfloat("rtol")
settings["atol"] = configParser["SETTINGS"].getfloat("atol")