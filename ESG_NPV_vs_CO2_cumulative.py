from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pv_annuity(annual: float, rate: float, years: int) -> float:
    """Present value of an annuity paid at end of each year for `years` periods."""
    if rate == 0:
        return annual * years
    return annual * (1 - (1 + rate) ** (-years)) / rate


def build_results(
    scenarios: List[Dict[str, Any]],
    base_noi: float,
    yields: List[float],
    years: int,
    upgrade_share: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in scenarios:
        for y in yields:
            cash_no = float(s["Annual_Savings"])
            cash_with = cash_no * upgrade_share

            annual_no = base_noi + cash_no
            annual_with = base_noi + cash_with

            pv_10_no = pv_annuity(annual_no, y, years)
            pv_10_with = pv_annuity(annual_with, y, years)

            exit_no = annual_no / y if y != 0 else annual_no * years
            exit_with = annual_with / y if y != 0 else annual_with * years

            pv_exit_no = exit_no / ((1 + y) ** years)
            pv_exit_with = exit_with / ((1 + y) ** years)

            total_no = pv_10_no + pv_exit_no - float(s["CAPEX"])
            total_with = pv_10_with + pv_exit_with - float(s["CAPEX"])

            payback_no = float(s["CAPEX"]) / cash_no if cash_no > 0 else np.nan
            payback_with = float(s["CAPEX"]) / cash_with if cash_with > 0 else np.nan

            rows.append(
                {
                    "Scenario": s["Scenario"],
                    "Yield": f"{int(y*100)}%",
                    "CAPEX": float(s["CAPEX"]),
                    "Base_NOI": base_noi,
                    "Annual_Savings_NoUpgrade": cash_no,
                    "Annual_Savings_WithUpgrade": cash_with,
                    "Annual_NOI_NoUpgrade": annual_no,
                    "Annual_NOI_WithUpgrade": annual_with,
                    "PV_10yr_NoUpgrade": pv_10_no,
                    "PV_10yr_WithUpgrade": pv_10_with,
                    "ExitValue_NoUpgrade": exit_no,
                    "ExitValue_WithUpgrade": exit_with,
                    "PV_Exit_NoUpgrade": pv_exit_no,
                    "PV_Exit_WithUpgrade": pv_exit_with,
                    "TotalValue_NoUpgrade": total_no,
                    "TotalValue_WithUpgrade": total_with,
                    "SimplePayback_NoUpgrade_yrs": payback_no,
                    "SimplePayback_WithUpgrade_yrs": payback_with,
                    "CO2_10yr_tons": float(s["CO2_10yr"]),
                }
            )
    df = pd.DataFrame(rows)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)
    return df

def plot_cumulative_package(sub_df: pd.DataFrame, out_dir: Path, rep_yield_label: str = "7%") -> Path:
    sub = sub_df[sub_df["Yield"] == rep_yield_label].copy()
    sub = sub[["Scenario", "CO2_10yr_tons", "TotalValue_NoUpgrade"]].rename(
        columns={"TotalValue_NoUpgrade": "PackageNPV"}
    )

    # avoid divide-by-zero and ensure numeric types
    sub["CO2_10yr_tons"] = pd.to_numeric(sub["CO2_10yr_tons"], errors="coerce").fillna(0.0)
    sub["PackageNPV"] = pd.to_numeric(sub["PackageNPV"], errors="coerce").fillna(0.0)

    sub["eff"] = sub["PackageNPV"] / sub["CO2_10yr_tons"].replace({0: np.nan})
    sub = sub.sort_values("eff", ascending=False).reset_index(drop=True)

    sub["Cumulative_CO2"] = sub["CO2_10yr_tons"].cumsum()
    sub["Cumulative_NPV_m€"] = sub["PackageNPV"].cumsum() / 1e6

    # safety: if empty, return quickly
    if sub.empty:
        raise ValueError("No data for selected yield label")

    # NPV max point
    idx_max = sub["Cumulative_NPV_m€"].idxmax()
    cumu_max = (float(sub.loc[idx_max, "Cumulative_CO2"]), float(sub.loc[idx_max, "Cumulative_NPV_m€"]))

    # Mid CO2 point (nearest)
    total_co2 = float(sub["Cumulative_CO2"].iloc[-1])
    mid_co2 = total_co2 / 2.0
    idx_mid = (sub["Cumulative_CO2"] - mid_co2).abs().idxmin()
    cumu_mid = (float(sub.loc[idx_mid, "Cumulative_CO2"]), float(sub.loc[idx_mid, "Cumulative_NPV_m€"]))

    # Improved NPV Neutral: find first crossing of cumulative NPV through zero (positive -> non-positive)
    cum_npv = sub["Cumulative_NPV_m€"].to_numpy(dtype=float)
    cum_co2 = sub["Cumulative_CO2"].to_numpy(dtype=float)
    cumu_neutral = None

    # find first index where cumulative NPV <= 0
    nonpos_indices = np.where(cum_npv <= 0)[0]
    if nonpos_indices.size > 0:
        i = int(nonpos_indices[0])
        if i == 0:
            # first point is non-positive -> neutral at first CO2 point (or zero)
            cumu_neutral = (cum_co2[0], 0.0)
        else:
            # interpolate between i-1 (positive) and i (non-positive) to get more precise CO2 where NPV==0
            y1, y2 = cum_npv[i - 1], cum_npv[i]
            x1, x2 = cum_co2[i - 1], cum_co2[i]
            if y2 != y1:
                t = (0.0 - y1) / (y2 - y1)
                x_zero = x1 + t * (x2 - x1)
            else:
                x_zero = x1
            cumu_neutral = (float(x_zero), 0.0)
    else:
        # no non-positive value found -> no neutral crossing within cumulative range
        cumu_neutral = None

    cumu_co2_max = (float(sub["Cumulative_CO2"].iloc[-1]), float(sub["Cumulative_NPV_m€"].iloc[-1]))

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sub["Cumulative_CO2"], sub["Cumulative_NPV_m€"], linestyle="--", color="black", marker="+", markersize=8)
    ax.fill_between(sub["Cumulative_CO2"], sub["Cumulative_NPV_m€"], alpha=0.05, color="black")

    ax.scatter([cumu_max[0]], [cumu_max[1]], color="black", s=70, marker="P")
    ax.annotate("NPV Max", xy=cumu_max, xytext=(6, 6), textcoords="offset points")

    ax.scatter([cumu_mid[0]], [cumu_mid[1]], color="black", s=70, marker="P")
    ax.annotate("NPV Mid", xy=cumu_mid, xytext=(6, -12), textcoords="offset points")

    # Plot interpolated neutral point (if found) and annotate (ensures it can appear before mid)
    if cumu_neutral is not None:
        ax.scatter([cumu_neutral[0]], [cumu_neutral[1]], color="black", s=70, marker="P")
        ax.annotate("NPV Neutral", xy=cumu_neutral, xytext=(6, 6), textcoords="offset points")

    ax.scatter([cumu_co2_max[0]], [cumu_co2_max[1]], color="black", s=70, marker="P")
    ax.annotate("CO2 Max", xy=cumu_co2_max, xytext=(6, -14), textcoords="offset points")

    ax.set_xlabel("Cumulative Metric Tons of CO₂ Saved (10 Years)")
    ax.set_ylabel("Net Present Value of Packages (millions €)")
    ax.set_title(f"NPV of Packages versus CO₂ Saved — yields 6%, 7%, 8% (highlighting {rep_yield_label})")

    # also plot the 6% and 8% cumulative curves in the background for comparison
    for yl in ("6%", "8%"):
        if yl == rep_yield_label:
            continue
        tmp = sub_df[sub_df["Yield"] == yl].copy()
        if tmp.empty:
            continue
        tmp["CO2_10yr_tons"] = pd.to_numeric(tmp["CO2_10yr_tons"], errors="coerce").fillna(0.0)
        tmp["TotalValue_NoUpgrade"] = pd.to_numeric(tmp["TotalValue_NoUpgrade"], errors="coerce").fillna(0.0)
        tmp["eff"] = tmp["TotalValue_NoUpgrade"] / tmp["CO2_10yr_tons"].replace({0: np.nan})
        tmp = tmp.sort_values("eff", ascending=False).reset_index(drop=True)
        tmp["Cumulative_CO2"] = tmp["CO2_10yr_tons"].cumsum()
        tmp["Cumulative_NPV_m€"] = tmp["TotalValue_NoUpgrade"].cumsum() / 1e6
        ax.plot(
            tmp["Cumulative_CO2"],
            tmp["Cumulative_NPV_m€"],
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
            zorder=0,
            label=f"{yl} (compare)",
        )

    # show legend if comparison lines were added
    if any(sub_df["Yield"].isin(["6%", "8%"])):
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.ticklabel_format(axis="y", style="plain")
    plt.tight_layout()

    out = out_dir / "NPV_vs_CO2_cumulative.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

def plot_exit_vs_co2(df: pd.DataFrame, out_dir: Path, yields: List[float]) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"6%": "#1f77b4", "7%": "#ff7f0e", "8%": "#2ca02c"}
    for y in yields:
        y_label = f"{int(y*100)}%"
        sub = df[df["Yield"] == y_label].sort_values("CO2_10yr_tons")
        ax.plot(sub["CO2_10yr_tons"], sub["ExitValue_NoUpgrade"] / 1e6, linestyle="--",
                color=colors.get(y_label, "gray"), marker="o", label=f"Exit @{y_label} (no upgrade)")
        ax.plot(sub["CO2_10yr_tons"], sub["ExitValue_WithUpgrade"] / 1e6, linestyle=":",
                color=colors.get(y_label, "gray"), marker="s", label=f"Exit @{y_label} (with upgrade)")

    for _, r in df[df["Yield"] == "6%"].iterrows():
        ax.annotate(r["Scenario"], (r["CO2_10yr_tons"], r["ExitValue_NoUpgrade"] / 1e6),
                    xytext=(6, -6), textcoords="offset points", fontsize=9)

    ax.set_title("Exit Value (millions €) vs 10yr CO₂ Saved — with Base NOI")
    ax.set_xlabel("Metric Tons of CO₂ Saved (10 Years)")
    ax.set_ylabel("Exit Value (millions €)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = out_dir / "ExitValue_vs_CO2_with_rent.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    # Inputs
    scenarios = [
        {"Scenario": "Scenario 1", "CAPEX": 2_924_128.00, "Annual_Savings": 244_988.00, "CO2_10yr": 2449.88},
        {"Scenario": "Scenario 2", "CAPEX": 2_507_181.00, "Annual_Savings": 223_328.00, "CO2_10yr": 2233.28},
        {"Scenario": "Scenario 3", "CAPEX": 1_591_506.00, "Annual_Savings": 167_866.00, "CO2_10yr": 1678.66},
        {"Scenario": "Scenario 4", "CAPEX": 1_536_464.00, "Annual_Savings": 121_682.00, "CO2_10yr": 1216.82},
        {"Scenario": "Scenario 5", "CAPEX": 1_123_939.00, "Annual_Savings": 128_763.00, "CO2_10yr": 1287.63},
    ]

    base_NOI = 1_393_200.00
    yields = [0.06, 0.07, 0.08]
    years = 10
    upgrade_share = 0.5

    df = build_results(scenarios, base_NOI, yields, years, upgrade_share)

    out_dir = Path.home() / "Desktop"
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = out_dir / "ESG_business_case_10yr_with_rent.xlsx"
    df.to_excel(excel_path, index=False, sheet_name="10yr Business Case")

    chart1 = plot_cumulative_package(df, out_dir, rep_yield_label="7%")
    chart2 = plot_exit_vs_co2(df, out_dir, yields)

    print(f"Excel saved to: {excel_path}")
    print(f"Chart saved to: {chart1}")
    print(f"Chart saved to: {chart2}")


if __name__ == "__main__":
    main()
