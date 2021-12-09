def create_chosen_flow_util(data):
    out = data["Flow_Utility_N"].copy()
    out.loc[data["Choice"] == "Part"] = data[data["Choice"] == "1"]["Flow_Utility_P"]
    out.loc[data["Choice"] == "Full"] = data[data["Choice"] == "2"]["Flow_Utility_F"]
    return out


def create_disc_sum_av_utility(data, model_params_df):
    flow_util = create_chosen_flow_util(data)
    delta = model_params_df.loc[("delta", slice(None)), "value"].to_numpy()
    disc_av = (flow_util * (delta ** data["Period"])).groupby("Identifier").sum().mean()
    return disc_av
