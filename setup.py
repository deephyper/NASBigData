from setuptools import setup, find_packages

setup(
    name="nas_big_data",
    packages=find_packages(),
    install_requires=[
        "deephyper",
        "deepspace>=0.0.3",
        # "autosklearn",
        # "emcee",
        # "pyDOE",
        ],
)