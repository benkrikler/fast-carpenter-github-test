#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()


def get_version():
    _globals = {}
    with open(os.path.join("fast_carpenter", "version.py")) as version_file:
        exec(version_file.read(), _globals)
    return _globals["__version__"]


requirements = ['atuproot==0.1.13', 'atsge==0.2.1', 'atpbar==1.0.8', 'mantichora==0.9.7',
                'alphatwirl==0.25.5', 'fast-flow>0.5.0', 'fast-curator', 'awkward<1.0.0',
                'pandas>=1.1', 'numpy>=1.16.5', 'numba', 'numexpr', 'uproot >=3,<4']
repositories = []

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'flake8', 'pytest-cov']

setup(
    author="Ben Krikler",
    author_email='fast-hep@cern.ch',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="F.A.S.T. package for summarizing ROOT TTrees",
    entry_points={
        'console_scripts': [
            'fast_carpenter=fast_carpenter.__main__:main',
        ],
    },
    install_requires=requirements,
    dependency_links=repositories,
    license="Apache Software License 2.0",
    long_description=readme,  # + '\n\n' + history,
    include_package_data=True,
    keywords=['ROOT', 'pandas', 'analysis', 'particle physics', 'HEP', 'F.A.S.T'],
    name='fast-carpenter',
    packages=find_packages(include=['fast_carpenter*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/FAST-HEP/fast-carpenter',
    version=get_version(),
    zip_safe=True,
)
