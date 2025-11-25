"""Simple logging utilities used by the SDK.

This keeps logging consistent and small; apps can replace or extend it.
"""
import logging


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
