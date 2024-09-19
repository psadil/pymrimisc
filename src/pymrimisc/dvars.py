import numpy as np
import polars as pl
from scipy import stats


def sd_hIQR(x, d=1):
    w = x ** (1 / d)  # Power trans.: w~N(mu_w, sigma_w^2)
    sd = (np.quantile(w, 0.5) - np.quantile(w, 0.25)) / (1.349 / 2)  # hIQR
    out = (d * np.median(w) ** (d - 1)) * sd  # Delta method
    # In the paper, the above formula incorrectly has d^2 instead of d.
    # The code on github correctly uses d.
    return out


def get_zd(D: pl.Series) -> np.ndarray:
    DV2 = D.drop_nans().drop_nulls().to_numpy() * 4
    mu_0 = np.median(DV2)  # pg 305
    sigma_0 = sd_hIQR(DV2, d=3)  # pg 305: cube root power trans
    v = 2 * mu_0**2 / sigma_0**2
    X = v / mu_0 * DV2  # pg 298: ~X^2(v=2*mu_0^2/sigma_0^2)
    P = stats.chi2.cdf(X, v)
    return np.concatenate(
        [
            [np.nan],
            np.where(
                np.abs(P - 0.5) < 0.49999,
                stats.norm.ppf(1 - stats.chi2.cdf(X, v)),
                (DV2 - mu_0) / sigma_0,
            ),
        ]
    )


def get_dvars(x: np.ndarray) -> pl.DataFrame:

    # normalize
    X = x.copy()
    X = X / np.median(np.mean(X, axis=0)) * 100
    avgs = np.mean(X, 0)
    X = X - avgs[:, None].T

    out = (
        pl.from_numpy(X)
        .rename(lambda col: col.removeprefix("column_"))
        .with_row_index(name="t", offset=0)
        .unpivot(index="t")
        .with_columns(
            A=pl.col("value") ** 2,
            D=pl.col("value").diff().over("variable", order_by="t"),
            S=pl.col("value")
            .rolling_mean(window_size=2)
            .over("variable", order_by="t"),
        )
        .with_columns((pl.selectors.by_name("D", "S") ** 2) / 4)
        .group_by("t")
        .agg((~pl.selectors.by_name("variable")).mean())
        .sort("t")
        .with_columns(
            DPD=(pl.col("D") - pl.col("D").median())
            / pl.col("A").mean()
            * 100,
            ZD=pl.col("D").map_batches(get_zd),
            DVARS=pl.col("D").sqrt() * 2,
        )
        .fill_nan(None)
        .fill_null(0)
    )
    return out
