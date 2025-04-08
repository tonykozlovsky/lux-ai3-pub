import getpass
import os

BOARD_SIZE = (24, 24)
MAX_UNITS = 16

if os.getenv("TRAINING") == "1":
    MODEL_PATH = None
else:
    MODEL_PATH = 'submission_model'
