#!/usr/bin/env python

import setuptools
import os

__version__ = "2019.8.2"

if __name__ == "__main__":
    setuptools.setup(
        name="apdft",
        version=__version__,
        author="Guido Falk von Rudorff",
        author_email="guido@vonrudorff.de",
        description="APDFT calculates quantumchemical results for many molecules at once.",
        long_description="""
	APDFT is a software to allow quantum-chemistry calculations of many isoelectronic molecules at once rather than evaluating them one-by-one. This is achieved through Alchemical Perturbation Density Functional Theory (https://arxiv.org/abs/1809.01647) where the change in external potential between molecules is treated as perturbation. This concept works just as fine for post-HF methods.

	All gaussian basis sets from the EMSL Basis Set Exchange and a variety of methods (HF, LDA, PBE, PBE0, CCSD) are supported. APDFT does not reinvent the wheel but leverages other QM software in the background.""",
        url="https://github.com/ferchault/apdft",
        packages=setuptools.find_packages("src/"),
        package_dir={"": "src"},
        entry_points={"console_scripts": ["apdft=apdft.commandline:entry_cli"]},
        python_requires=">=3.6,<4",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Programming Language :: Python",
        ],
        install_requires=[
            "jinja2 == 3.0.3",
            "basis_set_exchange == 0.9",
            "scipy == 1.5.4",
            "numpy>=1.21",
            "pandas == 1.1.5",
            "pyscf == 2.0.1",
            "structlog == 21.5.0",
            "colorama == 0.4.4",
        ],
    )
