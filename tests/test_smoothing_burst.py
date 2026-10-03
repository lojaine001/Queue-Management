"""
Unit tests for prediction smoothing and burst bypass logic.
Asserts that arrival bursts are preserved when BURST_BYPASS=True,
and smoothed when BURST_BYPASS=False.
"""
import os
import unittest
import pandas as pd
import numpy as np


def apply_outlier_cap_and_smoothing(df_input, burst_threshold=50.0, cap=30.0,
                                     smooth_window=5, burst_bypass=True):
    """
    Encapsulates the exact smoothing and outlier cap logic used in ensemble_predict.py.
    """
    df = df_input.copy()
    non_zero = df.loc[df["y"] > 0, "y"]
    burst_days = set()

    if not non_zero.empty and cap is not None:
        df["date_only"] = pd.to_datetime(df["ds"]).dt.date
        burst_days = set(df.loc[df["y"] >= burst_threshold, "date_only"].unique())
        if burst_days and burst_bypass:
            mask_no_burst = ~df["date_only"].isin(burst_days)
            df.loc[mask_no_burst, "y"] = df.loc[mask_no_burst, "y"].clip(upper=cap)
        else:
            df["y"] = df["y"].clip(upper=cap)

    if smooth_window > 1:
        smoothed = (
            df["y"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .median()
            .round(2)
        )
        if burst_bypass and "date_only" in df.columns and burst_days:
            mask_no_burst = ~df["date_only"].isin(burst_days)
            df.loc[mask_no_burst, "y"] = smoothed.loc[mask_no_burst]
        elif burst_bypass and (df["y"] >= burst_threshold).any():
            burst_mask = df["y"] >= burst_threshold
            df.loc[~burst_mask, "y"] = smoothed.loc[~burst_mask]
        else:
            df["y"] = smoothed

    return df


class TestSmoothingBurst(unittest.TestCase):
    def setUp(self):
        # Create a series with a clear burst day (e.g. y = 120 during peak)
        dates = pd.date_range("2026-09-20 09:00", periods=20, freq="3min")
        # Base values around 10-15, with a burst of 120 at index 10
        y = [10.0] * 20
        y[10] = 120.0
        self.df = pd.DataFrame({"ds": dates, "y": y})

    def test_burst_preserved_when_burst_bypass_true(self):
        """When BURST_BYPASS=True, peak burst should not be clipped to 30 or smoothed to median."""
        result = apply_outlier_cap_and_smoothing(
            self.df,
            burst_threshold=50.0,
            cap=30.0,
            smooth_window=5,
            burst_bypass=True,
        )
        peak_val = result.iloc[10]["y"]
        self.assertEqual(peak_val, 120.0, "Burst peak must be preserved when BURST_BYPASS is True")

    def test_burst_suppressed_when_burst_bypass_false(self):
        """When BURST_BYPASS=False, peak burst should be capped at 30 and smoothed by rolling median."""
        result = apply_outlier_cap_and_smoothing(
            self.df,
            burst_threshold=50.0,
            cap=30.0,
            smooth_window=5,
            burst_bypass=False,
        )
        peak_val = result.iloc[10]["y"]
        self.assertLessEqual(peak_val, 30.0, "Burst peak must be capped when BURST_BYPASS is False")
        # Rolling median of [10, 10, 30, 10, 10] is 10.0
        self.assertEqual(peak_val, 10.0, "Rolling median should smooth out isolated spike when bypass is False")

    def test_configurable_smooth_window_one_disables_smoothing(self):
        """When smooth_window <= 1, no rolling median is applied."""
        y = [10.0, 15.0, 25.0, 12.0, 8.0]
        dates = pd.date_range("2026-09-20 09:00", periods=5, freq="3min")
        df = pd.DataFrame({"ds": dates, "y": y})

        result = apply_outlier_cap_and_smoothing(
            df,
            burst_threshold=50.0,
            cap=None,
            smooth_window=1,
            burst_bypass=True,
        )
        np.testing.assert_array_equal(result["y"].values, df["y"].values)


if __name__ == "__main__":
    unittest.main()
