# This code is licensed under the MIT License (see LICENSE file for details)

import setuptools

setuptools.setup(
    name = 'elegant',
    version = '1',
    description = 'tools and pipelines for zplab C. elegans data',
    packages = setuptools.find_packages(),
    package_data = {'elegant': ['width_data/*.pickle']},
    entry_points = {
        'console_scripts': [
            'segment_experiment=elegant.process_experiment:segment_main',
            'compress_experiment=elegant.process_experiment:compress_main',
            'update_experiment_metadata=elegant.process_experiment:update_metadata_main'
        ],
    }
)
