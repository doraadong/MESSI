#!/usr/bin/env python
# -*- coding: utf-8 -*-

# partly borrowed from from https://github.com/navdeep-G/setup.py/blob/master/setup.py
import io
import os
from setuptools import setup, find_packages
import pathlib

# Package meta-data.
NAME = 'messi'
DESCRIPTION = 'Mixture of Experts for Spatial Signaling genes Identification'
URL = "https://github.com/doraadong/MESSI"
EMAIL = 'dongshul@andrew.cmu.edu'
AUTHOR = 'Dora Li'
REQUIRES_PYTHON = '>=3.6'
VERSION = '1.0.0'

# What packages are required for this module to be executed?
REQUIRED = ['scipy>=1.3.0', 'numpy>=1.16.3', 'scikit-learn>=0.22.1', 'pandas>=0.25.3',
            "importlib_resources ; python_version<'3.7'"]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['messi'],
    package_data={'messi': ['method_diagram.png','input/ligand_receptor_pairs2.txt']},
    entry_points={'console_scripts': ['messi=messi.messi:main']},
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
