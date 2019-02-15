import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

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


class PublishCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


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
    cmdclass={"publish": PublishCommand},
)
