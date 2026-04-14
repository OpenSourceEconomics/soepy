# from types import SimpleNamespace
#
# import numpy as np
# import pandas as pd
#
# from soepy.exogenous_processes.children import gen_prob_child_vector
#
#
# def test_child_prob_vector_conditional_shape(tmp_path):
#     num_periods = 3
#     num_educ_levels = 3
#     index = pd.MultiIndex.from_product(
#         [
#             range(num_periods),
#             range(num_educ_levels),
#             [0, 1],
#             [0, 1],
#         ],
#         names=["period", "educ_level", "partner_present", "has_prior_kid"],
#     )
#     values = np.arange(len(index), dtype=float)
#     df = pd.DataFrame({"prob_child_values": values}, index=index)
#     file_path = tmp_path / "child_info.pkl"
#     df.to_pickle(file_path)
#
#     model_spec = SimpleNamespace(
#         num_periods=num_periods,
#         num_educ_levels=num_educ_levels,
#         child_info_file_name=file_path,
#     )
#
#     prob_child = gen_prob_child_vector(model_spec)
#     assert prob_child.shape == (num_periods, num_educ_levels, 2, 2)
#     assert prob_child[0, 0, 0, 0] == values[0]
#     assert prob_child[-1, -1, 1, 1] == values[-1]
#
#
# def test_child_prob_vector_partner_haskids_shape(tmp_path):
#     num_periods = 3
#     num_educ_levels = 3
#     index = pd.MultiIndex.from_product(
#         [
#             range(num_periods),
#             range(num_educ_levels),
#             [0, 1],
#             [0, 1, 2, 3, 4],
#         ],
#         names=["period", "educ_level", "partner_present", "child_state"],
#     )
#     values = np.arange(len(index), dtype=float)
#     df = pd.DataFrame({"prob_child_values": values}, index=index)
#     file_path = tmp_path / "child_info_childstate.pkl"
#     df.to_pickle(file_path)
#
#     model_spec = SimpleNamespace(
#         num_periods=num_periods,
#         num_educ_levels=num_educ_levels,
#         child_info_file_name=file_path,
#     )
#
#     prob_child = gen_prob_child_vector(model_spec)
#     assert prob_child.shape == (num_periods, num_educ_levels, 2, 5)
#     assert prob_child[0, 0, 0, 0] == values[0]
#     assert prob_child[-1, -1, 1, 4] == values[-1]
