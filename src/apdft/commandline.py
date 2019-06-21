#!/usr/bin/env python
import argparse

def build_main_commandline():
    """ Builds an argparse object of the user-facing command line interface."""

    parser = argparse.ArgumentParser(description='QM calculations on multiple systems.')