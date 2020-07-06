from types import SimpleNamespace


def load_config(cfg_file):
    """
    Load configurations from a plain text file and use a namespace to store them
    """
    config = SimpleNamespace()
    with open(cfg_file) as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.split('#', 1)[0].split('/', 1)[0].strip().replace(' ', '')
        if '=' in line:
            key, value = line.split('=')
            if '\'' in value or '\"' in value:
                value = value.replace('\'', '').replace('\"', '')
            else:
                value = float(value)
            setattr(config, key, value)

    return config
