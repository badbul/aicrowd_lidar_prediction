import yaml

def read_config(filename):
    with open(filename) as fp:
        return yaml.safe_load(fp)