#!/usr/bin/env python
import numpy as np
import sys
from basis_set_exchange import lut
from . import calculator
from . import commandline
from . import settings
from . import math

LOG_LEVEL_USAGE = {}


def count_log_level_usage(_, __, event_dict):
    level = event_dict["level"]
    if level not in LOG_LEVEL_USAGE:
        LOG_LEVEL_USAGE[level] = 0
    LOG_LEVEL_USAGE[level] += 1
    return event_dict


# setup output
def _setup_logging():
    import structlog

    my_styles = structlog.dev.ConsoleRenderer.get_default_level_styles()
    my_styles["RESULT"] = my_styles["info"]

    if sys.stdout.isatty():
        renderer = structlog.dev.ConsoleRenderer(level_styles=my_styles)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M", utc=False),
            structlog.processors.StackInfoRenderer(),
            count_log_level_usage,
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


try:
    log = _setup_logging()
except TypeError:
    # mocking environment
    log = None


def get_methods():
    calculators = [
        getattr(Calculator, _)
        for _ in dir(Calculator)
        if "Calculator" in _ and _ != "Calculator"
    ]
    methods = []
    for calculator in calculators:
        c = calculator._methods.keys()
        methods += c
    return sorted(set(methods))


def get_element_number(element):
    return lut.element_Z_from_sym(element)


def read_xyz(fn):
    with open(fn) as fh:
        lines = fh.readlines()
    numatoms = int(lines[0].strip())
    lines = lines[2 : 2 + numatoms]
    nuclear_numbers = []
    coordinates = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            break
        parts = line.split()
        try:
            nuclear_numbers.append(int(parts[0]))
        except:
            nuclear_numbers.append(get_element_number(parts[0]))
        coordinates.append([float(_) for _ in parts[1:4]])
    return np.array(nuclear_numbers), np.array(coordinates)
