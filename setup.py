# This code is licensed under the MIT License (see LICENSE file for details)

import setuptools

setuptools.setup(
    name = 'elegant',
    version = '1',
    description = 'tools and pipelines for zplab C. elegans data',
    packages = setuptools.find_packages(),
    package_data = {'elegant': ['width_data/*.pickle']},
)
