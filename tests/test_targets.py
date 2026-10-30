import unittest

import numpy as np
import pandas as pd

from bramex.features import build_switch_feature_table
from bramex.targets import build_switch_targets, possession_switch_mask


class PossessionSwitchTests(unittest.TestCase):
    def test_switch_requires_two_live_frames(self):
        possession = np.array([1, 1, 2, 2, 1, 1])
        ball_status = np.array([1, 1, 1, 0, 1, 1])
        expected = np.array([False, False, True, False, False, False])
        np.testing.assert_array_equal(
            possession_switch_mask(possession, ball_status), expected
        )

    def test_future_target_excludes_origin_and_includes_horizon_end(self):
        possession = np.array([1, 2, 2, 1, 1, 2])
        ball_status = np.ones(6, dtype=int)
        targets = build_switch_targets(
            possession,
            ball_status,
            match_id="M1",
            half="firstHalf",
            framerate=1,
            horizon_seconds=3,
            stride_seconds=1,
        )
        self.assertEqual(targets["switches_future"].tolist(), [2, 1, 2])

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            possession_switch_mask(np.array([1, 2]), np.array([1]))

    def test_dead_ball_origins_are_excluded_by_default(self):
        possession = np.array([1, 1, 2, 2, 1])
        ball_status = np.array([1, 0, 1, 1, 1])
        targets = build_switch_targets(
            possession,
            ball_status,
            match_id="M1",
            half="firstHalf",
            framerate=1,
            horizon_seconds=1,
            stride_seconds=1,
        )
        self.assertEqual(targets["frame"].tolist(), [0, 2, 3])

    def test_features_use_only_current_and_past_frames(self):
        possession = np.array([1, 1, 2, 2, 1, 1, 2, 2])
        ball_status = np.ones(8, dtype=int)
        ball_xy = np.column_stack([np.arange(8), np.zeros(8)])
        original = build_switch_feature_table(
            possession,
            ball_status,
            ball_xy,
            match_id="M1",
            half="firstHalf",
            framerate=1,
            horizon_seconds=2,
            stride_seconds=1,
        )

        changed_future = possession.copy()
        changed_future[5:] = 3 - changed_future[5:]
        modified = build_switch_feature_table(
            changed_future,
            ball_status,
            ball_xy,
            match_id="M1",
            half="firstHalf",
            framerate=1,
            horizon_seconds=2,
            stride_seconds=1,
        )
        feature_columns = [column for column in original.columns if column != "switches_future"]
        pd.testing.assert_frame_equal(
            original.loc[original["frame"] <= 4, feature_columns].reset_index(drop=True),
            modified.loc[modified["frame"] <= 4, feature_columns].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
