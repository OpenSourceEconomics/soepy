"""This module contains auxiliary functions for our testing processes."""
import glob
import os


def cleanup(options=None):
    """The function deletes package related output files."""
    fnames = glob.glob("*.soepy.*")

    if options is None:
        for f in fnames:
            os.remove(f)
    elif options == "regression":
        for f in fnames:
            if f.startswith("regression"):
                pass
            else:
                os.remove(f)
    elif options == "init_file":
        for f in fnames:
            if f.startswith("test.soepy"):
                pass
            else:
                os.remove(f)
