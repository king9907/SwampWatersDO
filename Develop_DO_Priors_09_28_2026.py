import os
import io
import json
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

today_str = datetime.now().strftime("%m_%d_%Y")

# =============================================================================
# 1. FILE PATHS & CONFIGURATION
# =============================================================================
USGS_SITE_ID = "02091500"  # Contentnea Creek at Hookerton, NC (Continuous Thermistor)
START_DATE = "2002-04-04"
END_DATE = "2004-08-31"
AIR_TEMP_FILE = r"NC_Air_Temp_1996-2026.xlsx"

DO_DATA_FILE = "regional_do_raw_data.csv"
TEMP_DATA_FILE = "regional_temp_raw_data.csv"
SCREENING_FILE = "Final_Stations_Minimally_Impacted_Screening.csv"
EXPORT_JSON_PARAMS = "Updated_Unimpaired_Sw_DO_Regression_Params.json"

N_SIMS = 10000
RANDOM_SEED = 42

# Verified NHDPlus COMIDs for Cohort 1C: 7 Strictly Unimpaired Class Sw Reaches (0 NC00 permits)
UNIMPAIRED_SW_COMIDS = {
    11235563,  # Creeping Swamp at NC 43 nr Vanceboro
    11573918,  # Nahunta Swamp nr Shine
    8811323,   # Black River nr Dunn
    10976681,  # Trent River at SR 1130 nr Pleasant Hill
    3350685,   # Briery Swamp beside SR 1545 nr Stokes
    11573958,  # Moccasin Run at SR 1543 nr Pikeville
    3349747,   # Haw Branch at Voice of America nr Leggetts Crossroads
}

MINIMALLY_IMPACTED_EXTRA_COMIDS = {
    10525101,  # Goshen Swamp at NC 403 nr Faison
    3350707,   # Tranters Creek at SR 1556 nr Washington
    10524689,  # NE Cape Fear River at NC 11 at Kornegay
}

FALLBACK_STATION_TO_COMID = {
    "21NC03WQ-J8150000": 11235563, "21NC02WQ-J8150000": 11235563, "21NC01WQ-J8150000": 11235563,
    "21NCMONITORING-J6890000": 11571552, "21NCCOALITIONS-J6890000": 11571552, "21NC01WQ-J6890000": 11571552,
    "21NCMONITORING-O7100000": 3350569, "21NCCOALITIONS-O7100000": 3350569, "21NC01WQ-O7100000": 3350569,
    "21NCMONITORING-O7000000": 3348989, "21NCCOALITIONS-O7000000": 3348989, "21NC03WQ-O7000000": 3348989, "21NC01WQ-O7000000": 3348989,
    "21NC03WQ-J7810000": 11574664, "21NC02WQ-J7810000": 11574664, "21NC01WQ-J7810000": 11574664, "21NC01WQ-J7770000": 11574664,
    "21NC03WQ-O7116000": 3350685,
    "21NC03WQ-O7300000": 3350985, "21NC02WQ-O7300000": 3350985, "21NC01WQ-O7300000": 3350985,
    "21NCMONITORING-J6055000": 11235989, "21NCCOALITIONS-J6055000": 11235989,
    "21NC03WQ-O7119000": 3349747,
    "21NC03WQ-J7301000": 11573958,
    "21NC02WQ-J8670000": 10976681, "21NC01WQ-J8670000": 10976681,
    "21NC01WQ-J7320000": 11573918,
    "21NC01WQ-B8820000": 8811323,
    "21NC01WQ-B9100000": 10524689,
    "21NC01WQ-O7120000": 3350707,
    "21NC01WQ-O7400000": 1969301,
    "21NC01WQ-B9180000": 10525101,
}


# =============================================================================
# 2. PHYSICAL THERMODYNAMIC DO SATURATION FUNCTION (BENSON & KRAUSE, 1984)
# =============================================================================
def do_saturation_benson_krause(temp_c):
    """
    Standard freshwater atmospheric DO saturation concentration (mg/L)
    as a function of water temperature (deg C) per Benson & Krause (1984).
    """
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    ln_do_sat = (
        -139.34411
        + (1.575701e5 / t_k)
        - (6.642308e7 / (t_k ** 2))
        + (1.243800e10 / (t_k ** 3))
        - (8.621949e11 / (t_k ** 4))
    )
    return np.exp(ln_do_sat)


# =============================================================================
# 3. PART A: AIR-TO-WATER TEMPERATURE REGRESSION (USGS 02091500 HOOKERTON)
# =============================================================================
def fit_air_water_temp_regression(usgs_site: str, start_dt: str, end_dt: str, air_excel_path: str) -> dict:
    print("=" * 65)
    print(f"  PART 1: USGS {usgs_site} AIR-TO-WATER TEMPERATURE REGRESSION")
    print("=" * 65)
    url = (
        f"https://waterservices.usgs.gov/nwis/dv/?format=rdb"
        f"&sites={usgs_site}&startDT={start_dt}&endDT={end_dt}&parameterCd=00010"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    data_lines = [line for line in response.text.splitlines() if not line.startswith("#")]
    df_usgs = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t").drop(0)

    col_max = next((c for c in df_usgs.columns if "00010" in c and c.endswith("00001")), None)
    col_min = next((c for c in df_usgs.columns if "00010" in c and c.endswith("00002")), None)
    col_mean = next((c for c in df_usgs.columns if "00010" in c and c.endswith("00003")), None)

    df_water = pd.DataFrame()
    df_water["Date"] = pd.to_datetime(df_usgs["datetime"], errors="coerce").dt.normalize()
    df_water["Water_Temp_Mean"] = pd.to_numeric(df_usgs[col_mean], errors="coerce")
    df_water["Water_Daily_Range"] = (
        pd.to_numeric(df_usgs[col_max], errors="coerce") - pd.to_numeric(df_usgs[col_min], errors="coerce")
        if (col_max and col_min) else np.nan
    )
    df_water = df_water.dropna(subset=["Date", "Water_Temp_Mean"])

    df_air = pd.read_excel(
        air_excel_path, sheet_name="Worksheet", skiprows=12,
        names=["Date", "Max_Air", "Avg_Air", "Min_Air"]
    )
    df_air["Date"] = pd.to_datetime(df_air["Date"], errors="coerce").dt.normalize()
    df_air = df_air.dropna(subset=["Date"])
    df_air[["Max_Air", "Avg_Air", "Min_Air"]] = df_air[["Max_Air", "Avg_Air", "Min_Air"]].apply(pd.to_numeric, errors="coerce")
    df_air["Air_Daily_Range"] = df_air["Max_Air"] - df_air["Min_Air"]

    df_merged = pd.merge(df_air, df_water, on="Date", how="inner")
    df_summer = df_merged[df_merged["Date"].dt.month.isin([6, 7, 8])].dropna(subset=["Avg_Air", "Water_Temp_Mean"]).copy()

    slope, intercept, r_val, p_val, _ = stats.linregress(df_summer["Avg_Air"], df_summer["Water_Temp_Mean"])
    df_summer["Residuals"] = df_summer["Water_Temp_Mean"] - ((slope * df_summer["Avg_Air"]) + intercept)
    sigma_eps = float(df_summer["Residuals"].std())

    air_range_mean = float(df_summer["Air_Daily_Range"].mean())
    water_range_mean = float(df_summer["Water_Daily_Range"].mean())

    # Sine-wave half-amplitudes A = (Max - Min) / 2.0
    air_half_amp = air_range_mean / 2.0
    water_half_amp = water_range_mean / 2.0

    air_mean_summer = float(df_summer["Avg_Air"].mean())
    air_std_summer = float(df_summer["Avg_Air"].std())
    air_p90 = float(df_summer["Avg_Air"].quantile(0.90))
    air_max = float(df_summer["Avg_Air"].max())

    print(f"  WATER_TEMP_SLOPE:            {slope:.4f}")
    print(f"  WATER_TEMP_INTERCEPT:        {intercept:.4f}")
    print(f"  WATER_TEMP_RMSE:             {sigma_eps:.4f} (R² = {r_val**2:.4f}, n = {len(df_summer)})")
    print(f"  AIR_TEMP_DAILY_RANGE (2A):   {air_range_mean:.2f} °C -> SINE HALF-AMP (A): {air_half_amp:.2f} °C")
    print(f"  WATER_TEMP_DAILY_RANGE (2A): {water_range_mean:.2f} °C -> SINE HALF-AMP (A): {water_half_amp:.2f} °C")
    print(f"  SUMMER AIR PRIORS:           Mean = {air_mean_summer:.2f} °C, SD = {air_std_summer:.2f} °C, P90 = {air_p90:.2f} °C, Max = {air_max:.2f} °C")

    return {
        "WATER_TEMP_SLOPE": round(float(slope), 4),
        "WATER_TEMP_INTERCEPT": round(float(intercept), 4),
        "WATER_TEMP_RMSE": round(sigma_eps, 4),
        "AIR_TEMP_RANGE": round(air_range_mean, 4),
        "WATER_TEMP_RANGE": round(water_range_mean, 4),
        "AIR_TEMP_HALF_AMP": round(air_half_amp, 4),
        "WATER_TEMP_HALF_AMP": round(water_half_amp, 4),
        "AIR_SUMMER_MEAN": round(air_mean_summer, 4),
        "AIR_SUMMER_STD": round(air_std_summer, 4),
        "AIR_P90": round(air_p90, 4),
        "AIR_MAX": round(air_max, 4),
    }


# =============================================================================
# 4. PART B: COMID-POOLED UNIMPAIRED SW DO-TEMPERATURE REGRESSION & DIURNAL FIT
# =============================================================================
def load_station_cohorts(screening_csv: str) -> pd.DataFrame:
    if os.path.exists(screening_csv):
        df_s = pd.read_csv(screening_csv)
        if "Screening_Group" in df_s.columns:
            df_s = df_s[df_s["Screening_Group"] == "Original 33 stations"].copy()
        df_s["NHDPlus_COMID"] = pd.to_numeric(df_s["NHDPlus_COMID"], errors="coerce").astype("Int64")
        return df_s[["MonitoringLocationIdentifier", "NHDPlus_COMID"]].dropna()
    return pd.DataFrame([
        {"MonitoringLocationIdentifier": k, "NHDPlus_COMID": v}
        for k, v in FALLBACK_STATION_TO_COMID.items()
    ])


def assemble_comid_pooled_do_temp(do_file: str, temp_file: str, station_map: pd.DataFrame):
    valid_sids = set(station_map["MonitoringLocationIdentifier"].unique())

    do_cols = ["MonitoringLocationIdentifier", "ActivityStartDate", "ActivityStartTime/Time", "CharacteristicName", "ResultMeasureValue"]
    df_do = pd.read_csv(do_file, usecols=lambda c: c in do_cols, low_memory=False)
    df_do = df_do[df_do["MonitoringLocationIdentifier"].isin(valid_sids)].copy()
    df_do = df_do[df_do["CharacteristicName"].astype(str).str.contains("oxygen", case=False, na=False)].copy()

    df_do["DO_mgL"] = pd.to_numeric(df_do["ResultMeasureValue"], errors="coerce")
    df_do["Date"] = pd.to_datetime(df_do["ActivityStartDate"], format="mixed", errors="coerce").dt.normalize()
    df_do = df_do.dropna(subset=["DO_mgL", "Date"])
    df_do = df_do[df_do["DO_mgL"] >= 0.0]
    df_do = df_do.merge(station_map, on="MonitoringLocationIdentifier", how="inner")

    df_do_dedup = df_do.drop_duplicates(subset=["NHDPlus_COMID", "Date", "DO_mgL"])
    df_do_daily = (
        df_do_dedup.groupby(["NHDPlus_COMID", "Date"], as_index=False)["DO_mgL"]
        .mean()
        .rename(columns={"DO_mgL": "DO"})
    )

    temp_cols = ["MonitoringLocationIdentifier", "ActivityStartDate", "ActivityStartTime/Time", "CharacteristicName", "ResultMeasureValue", "ResultMeasure/MeasureUnitCode"]
    df_t = pd.read_csv(temp_file, usecols=lambda c: c in temp_cols, low_memory=False)
    df_t = df_t[df_t["MonitoringLocationIdentifier"].isin(valid_sids)].copy()

    if "CharacteristicName" in df_t.columns:
        t_mask = df_t["CharacteristicName"].astype(str).str.contains("Temperature, water|temp", case=False, na=False)
        if t_mask.any():
            df_t = df_t[t_mask].copy()

    df_t["Temp_C"] = pd.to_numeric(df_t["ResultMeasureValue"], errors="coerce")
    if "ResultMeasure/MeasureUnitCode" in df_t.columns:
        is_f = df_t["ResultMeasure/MeasureUnitCode"].astype(str).str.upper().str.contains("F", na=False)
        df_t.loc[is_f, "Temp_C"] = (df_t.loc[is_f, "Temp_C"] - 32.0) * (5.0 / 9.0)

    df_t["Date"] = pd.to_datetime(df_t["ActivityStartDate"], format="mixed", errors="coerce").dt.normalize()
    df_t = df_t.dropna(subset=["Temp_C", "Date"])
    df_t = df_t[df_t["Temp_C"] >= 0.0]
    df_t = df_t.merge(station_map, on="MonitoringLocationIdentifier", how="inner")

    df_t_dedup = df_t.drop_duplicates(subset=["NHDPlus_COMID", "Date", "Temp_C"])
    df_t_daily = (
        df_t_dedup.groupby(["NHDPlus_COMID", "Date"], as_index=False)["Temp_C"]
        .mean()
        .rename(columns={"Temp_C": "Temp"})
    )

    df_paired = pd.merge(df_do_daily, df_t_daily, on=["NHDPlus_COMID", "Date"], how="inner")
    return df_paired, df_do


def bootstrap_diurnal_do_amplitude(df_do_all: pd.DataFrame, n_boot=1000, seed=42):
    np.random.seed(seed)
    df_hr = df_do_all.copy()
    dt_strings = df_hr["ActivityStartDate"].astype(str) + " " + df_hr["ActivityStartTime/Time"].astype(str)
    df_hr["Parsed_DT"] = pd.to_datetime(dt_strings, format="mixed", errors="coerce")
    df_hr["Hour"] = df_hr["Parsed_DT"].dt.hour + (df_hr["Parsed_DT"].dt.minute / 60.0)
    df_hr = df_hr.dropna(subset=["DO_mgL", "Hour", "Parsed_DT"])
    df_hr = df_hr[df_hr["Hour"] != 0]
    df_hr = df_hr.drop_duplicates(subset=["NHDPlus_COMID", "Parsed_DT", "DO_mgL"])

    summer_drought = df_hr[
        (df_hr["Parsed_DT"].dt.year.isin([2007, 2008])) &
        (df_hr["Parsed_DT"].dt.month.isin([4, 5, 6, 7, 8, 9, 10]))
    ].copy()
    if len(summer_drought) < 30:
        summer_drought = df_hr[df_hr["Parsed_DT"].dt.month.isin([4, 5, 6, 7, 8, 9, 10])].copy()

    local_means = summer_drought.groupby("NHDPlus_COMID")["DO_mgL"].transform("mean")
    summer_drought["delta_o"] = summer_drought["DO_mgL"] - local_means

    def diurnal_anomaly_model(t, amplitude, phase):
        return amplitude * np.sin((2.0 * np.pi / 24.0) * t + phase)

    max_possible_amp = (summer_drought["delta_o"].max() - summer_drought["delta_o"].min()) / 2.0
    amp_guess = max_possible_amp / 2.0
    initial_guess = [amp_guess, 0.0]
    lower_bounds = [0.0, -np.pi]
    upper_bounds = [max_possible_amp, np.pi]

    boot_half_amps = []
    for _ in range(n_boot):
        sample = summer_drought.sample(frac=1.0, replace=True)
        try:
            popt, _ = curve_fit(
                diurnal_anomaly_model, sample["Hour"], sample["delta_o"],
                p0=initial_guess, bounds=(lower_bounds, upper_bounds),
                method="trf", maxfev=3000
            )
            calc_amp = abs(popt[0])
            if calc_amp < max_possible_amp:
                boot_half_amps.append(calc_amp)
        except RuntimeError:
            continue

    half_amp_mean = float(np.mean(boot_half_amps))
    half_amp_std = float(np.std(boot_half_amps))
    return half_amp_mean, half_amp_std


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
def main():
    # 1. Run Hookerton Air-to-Water Temperature Regression directly
    temp_params = fit_air_water_temp_regression(USGS_SITE_ID, START_DATE, END_DATE, AIR_TEMP_FILE)

    # 2. Assemble COMID-Pooled Unimpaired Class Sw DO and Water Temperature Data
    print("\n" + "=" * 65)
    print("  PART 2: UNIMPAIRED CLASS SW DO-TEMPERATURE REGRESSION (J=7)")
    print("=" * 65)
    station_map = load_station_cohorts(SCREENING_FILE)
    df_paired, df_do_all = assemble_comid_pooled_do_temp(DO_DATA_FILE, TEMP_DATA_FILE, station_map)

    df_unimp_paired = df_paired[df_paired["NHDPlus_COMID"].isin(UNIMPAIRED_SW_COMIDS)].copy()

    # Fit Hierarchical Mixed-Effects Model (1 | COMID) so Creeping Swamp (n=341)
    # does not dominate 48.5% of the intercept over the other 6 reaches
    import statsmodels.api as sm

    df_unimp_paired["grp"] = df_unimp_paired["NHDPlus_COMID"].astype(str)
    m_mixed = sm.MixedLM.from_formula("DO ~ Temp", df_unimp_paired, groups=df_unimp_paired["grp"]).fit()
    do_intercept = float(m_mixed.fe_params["Intercept"])
    do_coef_temp = float(m_mixed.fe_params["Temp"])
    # Total predictive SD across a swamp reach = sqrt(within-reach residual variance + between-reach variance)
    do_rmse = float(np.sqrt(m_mixed.scale + m_mixed.cov_re.iloc[0, 0]))

    # Marginal R² of the fixed thermal effect
    y_primary = df_unimp_paired["DO"].values
    y_pred_fixed = do_intercept + do_coef_temp * df_unimp_paired["Temp"].values
    var_fixed = float(np.var(y_pred_fixed, ddof=1))
    do_r2 = var_fixed / (var_fixed + float(m_mixed.scale) + float(m_mixed.cov_re.iloc[0, 0]))

    # Empirical minimum non-zero DO (Practical Quantitation Limit in WQP dataset)
    empirical_min_do = float(df_unimp_paired["DO"].min())

    print(
        f"  Paired Unimpaired Sw Observations: N = {len(df_unimp_paired)} across J = {df_unimp_paired['NHDPlus_COMID'].nunique()} reaches")
    print(f"  DO_INTERCEPT (MixedLM):            {do_intercept:.4f}")
    print(f"  DO_COEF_TEMP (MixedLM):            {do_coef_temp:.4f}")
    print(f"  DO_RMSE (Total Predictive SD):     {do_rmse:.4f} (Marginal R² = {do_r2:.4f})")
    print(f"  EMPIRICAL_MIN_DO (Dataset Min):    {empirical_min_do:.4f} mg/L")

    # Fit Primary Model on Strictly Unimpaired Class Sw Reaches (J=7)
    X_primary = df_unimp_paired[["Temp"]].values
    y_primary = df_unimp_paired["DO"].values
    primary_model = LinearRegression().fit(X_primary, y_primary)
    do_intercept = float(primary_model.intercept_)
    do_coef_temp = float(primary_model.coef_[0])
    do_rmse = float(np.sqrt(mean_squared_error(y_primary, primary_model.predict(X_primary))))
    do_r2 = float(r2_score(y_primary, primary_model.predict(X_primary)))

    # Empirical minimum non-zero DO (Practical Quantitation Limit in WQP dataset)
    empirical_min_do = float(df_unimp_paired["DO"].min())

    print(f"  Paired Unimpaired Sw Observations: N = {len(df_unimp_paired)} across J = {df_unimp_paired['NHDPlus_COMID'].nunique()} reaches")
    print(f"  DO_INTERCEPT:                      {do_intercept:.4f}")
    print(f"  DO_COEF_TEMP:                      {do_coef_temp:.4f}")
    print(f"  DO_RMSE:                           {do_rmse:.4f} (R² = {do_r2:.4f})")
    print(f"  EMPIRICAL_MIN_DO (Dataset Min):    {empirical_min_do:.4f} mg/L")

    # 3. Bootstrap Regional Diurnal DO Half-Amplitude
    half_amp_mean, half_amp_std = bootstrap_diurnal_do_amplitude(df_do_all, n_boot=1000, seed=RANDOM_SEED)
    full_swing_mean = half_amp_mean * 2.0
    full_swing_std = half_amp_std * 2.0

    # 4. Simulate LFHT Priors Using the Exact Coupled Air -> Water -> DO Pipeline
    #    Bounded by Physical Zero (0.0 mg/L) and Benson & Krause (1984) 100% Saturation DO_sat(Tw)
    np.random.seed(RANDOM_SEED)
    a_air = (temp_params["AIR_P90"] - temp_params["AIR_SUMMER_MEAN"]) / temp_params["AIR_SUMMER_STD"]
    b_air = (temp_params["AIR_MAX"] - temp_params["AIR_SUMMER_MEAN"]) / temp_params["AIR_SUMMER_STD"]
    sim_air_temp = truncnorm.rvs(
        a_air, b_air, loc=temp_params["AIR_SUMMER_MEAN"], scale=temp_params["AIR_SUMMER_STD"], size=N_SIMS
    )
    sim_water_temp = (temp_params["WATER_TEMP_SLOPE"] * sim_air_temp) + temp_params["WATER_TEMP_INTERCEPT"]
    sim_expected_do = do_intercept + (do_coef_temp * sim_water_temp)

    # Physical upper bound: 100% atmospheric DO saturation at each simulated water temperature
    sim_do_sat_upper = do_saturation_benson_krause(sim_water_temp)

    # Draw baseline daily mean DO from Truncated Normal bounded strictly on [0.0, DO_sat(Tw)]
    a_do = (0.0 - sim_expected_do) / do_rmse
    b_do = (sim_do_sat_upper - sim_expected_do) / do_rmse
    sim_means = truncnorm.rvs(a_do, b_do, loc=sim_expected_do, scale=do_rmse)

    # Draw diurnal half-amplitude from Truncated Normal bounded strictly on [0.0, +inf)
    a_amp = (0.0 - half_amp_mean) / half_amp_std
    sim_half_amps = truncnorm.rvs(a_amp, np.inf, loc=half_amp_mean, scale=half_amp_std, size=N_SIMS)
    sim_swings = sim_half_amps * 2.0

    sim_mins = np.maximum(sim_means - sim_half_amps, 0.0)
    sim_maxs = np.minimum(sim_means + sim_half_amps, sim_do_sat_upper)

    priors = pd.DataFrame({
        "Parameter": ["DO_Min", "DO_Max", "DO_Avg", "Diurnal_Half_Amp_A", "Diurnal_Full_Swing_2A"],
        "Mean_mgL": [sim_mins.mean(), sim_maxs.mean(), sim_means.mean(), sim_half_amps.mean(), sim_swings.mean()],
        "StdDev_mgL": [sim_mins.std(), sim_maxs.std(), sim_means.std(), sim_half_amps.std(), sim_swings.std()],
        "P5_Lower": [
            np.percentile(sim_mins, 5), np.percentile(sim_maxs, 5), np.percentile(sim_means, 5),
            np.percentile(sim_half_amps, 5), np.percentile(sim_swings, 5)
        ],
        "P95_Upper": [
            np.percentile(sim_mins, 95), np.percentile(sim_maxs, 95), np.percentile(sim_means, 95),
            np.percentile(sim_half_amps, 95), np.percentile(sim_swings, 95)
        ]
    })

    # 5. Export Unified JSON Parameters for QUAL2K MCMC Script
    unified_params = {
        **temp_params,
        "DO_INTERCEPT": round(do_intercept, 4),
        "DO_COEF_LOGQ": 0.0,
        "DO_COEF_TEMP": round(do_coef_temp, 4),
        "DO_RMSE": round(do_rmse, 4),
        "DO_R2": round(do_r2, 4),
        "EMPIRICAL_MIN_DO": round(empirical_min_do, 4),
        "DO_HALF_AMP_MEAN": round(half_amp_mean, 4),
        "DO_HALF_AMP_STD": round(half_amp_std, 4),
        "DO_FULL_SWING_MEAN": round(full_swing_mean, 4),
        "DO_FULL_SWING_STD": round(full_swing_std, 4),
        "N_PAIRED_UNIMPAIRED_OBS": int(len(df_unimp_paired)),
        "J_UNIMPAIRED_REACHES": int(df_unimp_paired["NHDPlus_COMID"].nunique()),
    }
    with open(EXPORT_JSON_PARAMS, "w") as f:
        json.dump(unified_params, f, indent=4)

    print("\n--- FINAL COUPLED LFHT QUAL2K SUMMER PRIORS TABLE ---")
    print(priors.round(3).to_string(index=False))

    out_file_do = f"QUAL2K_Conditioned_Baseline_DO_Priors_{today_str}.csv"
    priors.to_csv(out_file_do, index=False)
    print(f"\n[EXPORTED] Conditioned DO priors saved to: {out_file_do}")
    print(f"[EXPORTED] Unified Thermal + DO JSON params saved to: {EXPORT_JSON_PARAMS}")

if __name__ == "__main__":
    main()