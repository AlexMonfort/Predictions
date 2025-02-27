from setuptools import setup, find_packages

setup(
    name="predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
    ],
)