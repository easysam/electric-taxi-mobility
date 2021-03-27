import pandas as pd
import logging


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return


def configure_pandas(_width=1000, _columns=20):
    pd.set_option('display.width', _width)
    pd.set_option('display.max_columns', _columns)
    return
