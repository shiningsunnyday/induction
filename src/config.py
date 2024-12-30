import os
import logging
import yaml

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    l.info(f"CONFIG: {CONFIG_FILE}")
    for name, value in globals().items():
        if not name.startswith('__'):  # Optionally filter out special Python variables
            l.info(f'{name}: {value}')    
    return l


CONFIG_FILE = os.getenv("config")
assert CONFIG_FILE, "please set environment variable config to a yaml file"
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)
for key in config:
    globals()[key] = config[key]
