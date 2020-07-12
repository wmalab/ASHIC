from setuptools import setup

setup(
    name='allelichicem',
    version='0.1.0',
    packages=['allelichicem'],
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
            'allelichicem = allelichicem.__main__:cli'
        ]
    },
)