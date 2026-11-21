"""Functions for handling logging more easily."""

import logging
from atum_ndp.config import Config


def get_logger(log_level=None):
    "Create a logger object."

    config = Config()

    if log_level is None:
        try:
            log_level = config.get_config_value("logger_config.level")
        except KeyError:
            log_level = "INFO"

    logger = logging.getLogger(__name__)

    # Handle where the logs go: stdout/file?
    logger.handlers = []  # Clean handlers list
    handler = logging.StreamHandler()

    # Format of our logs
    msg_fmt = "%(asctime)s - %(levelname)s - %(filename)s - [%(funcName)s: %(lineno)d] - %(message)s"
    time_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(msg_fmt, time_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger
