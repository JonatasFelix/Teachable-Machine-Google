import os
import warnings

def setup_environment():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", category=UserWarning, module='keras')

setup_environment()
