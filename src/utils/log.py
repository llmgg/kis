import logging
import logging.config
import os
import sys
from typing import Optional

FORMATTERS = {
    'verbose': {
        # 'format': '[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d] %(message)s',
        # 'datefmt': "%Y-%m-%d:%H:%M:%S",
        'format': '[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d] %(message)s',
        'datefmt': "%m-%d:%H:%M:%S",
    },
    'simple': {
        'format': '[%(levelname)s:%(name)s] %(message)s'
    },
}

FILE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'rotating': {
            'level': 'INFO',
            'formatter': 'verbose',
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 10000000,
            'backupCount': 5,
            'filename': 'kis.log',
        }
    },
    'root': {
        'handlers': ['rotating'],
        'level': 'DEBUG',
    }
}

CONSOLE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'simple',
            'class': 'logging.StreamHandler',
            'stream': None
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    }
}

FILE_CONSOLE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'console': CONSOLE_LOGGING["handlers"]["console"],
        'rotating': FILE_LOGGING["handlers"]["rotating"],
    },
    'root': {
        'handlers': ['console', 'rotating'],
        'level': 'DEBUG',
    }
}

NO_LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
}

LOGGING_CONFIGS = {
    "file_only": FILE_LOGGING,
    "console_only": CONSOLE_LOGGING,
    "file_console": FILE_CONSOLE_LOGGING,
    "none": NO_LOGGING,
}


def setup_main_logger(file=True, console=True, path: Optional[str] = None, level=logging.INFO,
                      console_level=None):
    """
    Configures logging for the main application.

    :param file: Whether to log to a file.
    :param console: Whether to log to the console.
    :param path: Optional path to write logfile to.
    :param level: Log level. Default: INFO.
    :param console_level: Optionally specify a separate log level for the console.
    """
    if file and console:
        log_config = LOGGING_CONFIGS["file_console"]
    elif file:
        log_config = LOGGING_CONFIGS["file_only"]
    elif console:
        log_config = LOGGING_CONFIGS["console_only"]
    else:
        log_config = LOGGING_CONFIGS["none"]

    if path is not None:
        log_config["handlers"]["rotating"]["filename"] = path  # type: ignore
    if os.path.exists(log_config["handlers"]["rotating"]["filename"]):
        os.remove(log_config["handlers"]["rotating"]["filename"])

    for _, handler_config in log_config['handlers'].items():  # type: ignore
        handler_config['level'] = level

    if 'console' in log_config['handlers'] and console_level is not None:  # type: ignore
        log_config['handlers']['console']['level'] = console_level  # type: ignore

    logging.config.dictConfig(log_config)  # type: ignore

    def exception_hook(exc_type, exc_value, exc_traceback):
        logging.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_hook

    return logging.getLogger("kis")


def log_torch_version(logger):
    try:
        from torch import __version__, __file__
        info = f'PyTorch: {__version__} ({__file__})'
    except ImportError:
        info = 'PyTorch unavailable'
    logger.info(info)


kis_logger = setup_main_logger()
