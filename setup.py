import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "soepy"
DESCRIPTION = (
    "soepy is an open-source Python package for the simulation and estimation of a "
    "dynamic model of human capital accumulation tailored to the German Socio-Economic"
    " Panel (SOEP)."
)
URL = "http://soepy.readthedocs.io"
EMAIL = "bilieva@diw.de"
AUTHOR = "Boryana Ilieva"

# What packages are required for this module to be executed?
REQUIRED = ["numpy", "flake8", "pytest", "pandas", "oyaml"]


here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    license="MIT",
    include_package_data=True,
)
