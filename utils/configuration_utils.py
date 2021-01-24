import json

def read_from_config(configFileName):
  with open(configFileName) as f:
    config = json.load(f)
  # for key, val in config.items():
  #   print('{}: {} (type: {})'.format(key, val, type(val)))

  return config