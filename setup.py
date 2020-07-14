from setuptools import setup

setup(
    name='ashic',
    version='0.1.0',
    packages=['ashic'],
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
            'ashic = ashic.__main__:cli'
        ]
    },
)