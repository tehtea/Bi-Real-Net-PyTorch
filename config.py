import os
import json

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_from_config(configFileName):
  with open(os.path.join('config', configFileName)) as f:
    config = json.load(f)
  return config
args = read_from_config(os.getenv('BIREAL_CFGFILE', 'train.json'))