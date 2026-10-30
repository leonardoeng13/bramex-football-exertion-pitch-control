import unittest

import numpy as np
import pandas as pd

from bramex.workstream_c import _workstream_b_origin_features


def sample_table(n=10):
    possession = np.array(["Home"] * (n // 2) + ["Away"] * (n - n // 2))
    return pd.DataFrame(
        {
            "sample": np.arange(n),
            "inplay": True,
            "poss_side": possession,
            "ball_x": np.arange(n, dtype=float),
            "home_dir": 1.0,
            "home_voronoi": 0.6,
            "home_voronoi_ball": 0.7,
        }
    )


class WorkstreamBOriginFeatureTests(unittest.TestCase):
    def test_features_are_expressed_for_current_possessing_side(self):
        table = sample_table(8)
        features = _workstream_b_origin_features(
            table,
            origins=np.array([3, 7]),
            current_home=np.array([True, False]),
        )

        np.testing.assert_allclose(features["B_current_pitch_share"], [0.6, 0.4])
        np.testing.assert_allclose(features["B_current_zone_share"], [0.7, 0.3])
        np.testing.assert_allclose(features["B_possession_share_60s"], [1.0, 0.5])
        np.testing.assert_allclose(features["B_progression_m_60s"], [3.0, -3.0])

    def test_future_samples_cannot_change_origin_features(self):
        table = sample_table(10)
        original = _workstream_b_origin_features(
            table,
            origins=np.array([4]),
            current_home=np.array([True]),
        )

        changed = table.copy()
        changed.loc[5:, ["poss_side"]] = "Home"
        changed.loc[5:, ["ball_x"]] = 10_000.0
        changed.loc[5:, ["home_voronoi", "home_voronoi_ball"]] = 0.01
        modified = _workstream_b_origin_features(
            changed,
            origins=np.array([4]),
            current_home=np.array([True]),
        )

        pd.testing.assert_frame_equal(original, modified)

    def test_tti_is_preferred_when_available(self):
        table = sample_table(6)
        table["home_tti"] = 0.8
        table["home_tti_ball"] = 0.9
        features = _workstream_b_origin_features(
            table,
            origins=np.array([2, 5]),
            current_home=np.array([True, False]),
        )

        np.testing.assert_allclose(features["B_current_pitch_share"], [0.8, 0.2])
        np.testing.assert_allclose(features["B_current_zone_share"], [0.9, 0.1])


if __name__ == "__main__":
    unittest.main()
