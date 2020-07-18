from setuptools import setup
from setuptools import find_packages

setup(
    name='ashic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'click',
        'scipy',
        'plotly',
        'scikit-learn',
        'iced',
        'statsmodels',
    ],
    entry_points={
        'console_scripts': [
            'ashic = ashic.__main__:cli',
            'ashic-data = ashic.cli.ashic_data:cli',
            'ashic-simul = ashic.cli.ashic_simul:cli',
            # 'ashic-utils = ashic.cli.ashic_utils:cli'
        ]
    },
)