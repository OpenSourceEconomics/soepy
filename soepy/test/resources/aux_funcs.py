import glob
import os


def create_chosen_flow_util(data):
    out = data["Flow_Utility_N"].copy()
    out.loc[data["Choice"] == 1] = data[data["Choice"] == 1]["Flow_Utility_P"]
    out.loc[data["Choice"] == 2] = data[data["Choice"] == 2]["Flow_Utility_F"]
    return out


def create_disc_sum_av_utility(data, delta):
    flow_util = create_chosen_flow_util(data)
    disc_av = (
        (flow_util * (delta ** data["Period"])).groupby(data["Identifier"]).sum().mean()
    )
    return disc_av


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
