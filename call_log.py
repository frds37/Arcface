import time
import random
import logging
import sys

def make_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(filename="log.txt")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
