import logging
import sys


def build_stdout_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


DIAGNOSTICS_LOGGER = build_stdout_logger('algorithms.diagnostics')


def log_candidate_pois(message, *args, level=logging.INFO, exc_info=False):
    DIAGNOSTICS_LOGGER.log(level, message, *args, exc_info=exc_info)
