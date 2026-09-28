import os
import time
import warnings
from io import StringIO
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, kruskal, invgamma
from pynhd import NLDI, WaterData
from pynhd.nhdplus_derived import streamcat

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. SETTINGS & FILEPATHS (ZERO HEURISTIC MULTIPLIERS)
# ============================================================
INPUT_FILE = "Final_33_Golden_Stations.csv"
LOCAL_WWTP_CSV = "NC_NPDES_Wastewater_Discharge_Permits.csv"
RAW_DO_FILE = "regional_do_raw_data.csv"

OUTPUT_STATION_FILE = "Final_Stations_Minimally_Impacted_Screening.csv"
OUTPUT_NPDES_FILE = "Upstream_NPDES_Facilities.csv"
OUTPUT_STATS_TABLE = "Cohort_Statistical_Comparison_Table.csv"
OUTPUT_HARMONIC_TABLE = "Bayesian_Harmonic_Convergence_Comparison.csv"
OUTPUT_BOXPLOT_FIG = "Cohort_Disturbance_Comparison_Boxplots.png"
OUTPUT_HYDRO_DO_FIG = "Cohort_Hydrology_and_Harmonic_Priors.png"

DEVELOPED_THRESHOLD = 20.0
DISTANCE_BANDS_KM = [5, 10, 25, 50, 100, 200, 500, 1000]


# ============================================================
# 2. AUTO-DISCOVER CONTENTNEA CREEK USGS GAUGES
# ============================================================
def discover_contentnea_usgs_stations(state="NC") -> pd.DataFrame:
    """Query USGS NWIS for mainstem stream gauges on Contentnea Creek."""
    print("Querying USGS NWIS API to discover Contentnea Creek reference gauges...")
    url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={state}&siteType=ST&siteOutput=expanded"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    lines = [line for line in r.text.split("\n") if not line.startswith("#")]
    df_nwis = pd.read_csv(StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)

    mask = (
        df_nwis["station_nm"].str.contains("CONTENTNEA CR", case=False, na=False) &
        ~df_nwis["station_nm"].str.contains("LITTLE CONTENTNEA|TRIB|TRIP|UNMD TR|SWAMP|RUN|BR", case=False, na=False)
    )
    df_cont = df_nwis[mask].copy()

    records = []
    for _, row in df_cont.iterrows():
        records.append({
            "MonitoringLocationIdentifier": f"USGS-{row['site_no'].strip()}",
            "MonitoringLocationName": row["station_nm"].strip(),
            "LatitudeMeasure": pd.to_numeric(row.get("dec_lat_va"), errors="coerce"),
            "LongitudeMeasure": pd.to_numeric(row.get("dec_long_va"), errors="coerce"),
            "Screening_Group": "Contentnea longitudinal reference",
            "Station_Source": "USGS_NWIS_AutoDiscovered"
        })
    print(f"  -> Discovered {len(records)} mainstem USGS Contentnea Creek gauges.")
    return pd.DataFrame(records)


# ============================================================
# 3. NLDI COMID RESOLUTION & EXACT NHDPLUS EROM HYDROLOGY
# ============================================================
def classify_nc_permit(base_id: str) -> str:
    p = str(base_id).upper().strip()
    if p.startswith("NC00"):
        return "Individual_WWTP_or_Industrial (NC00)"
    if p.startswith("NCG55"):
        return "General_Small_Domestic (NCG55)"
    if p.startswith("NCG"):
        return "General_Stormwater_or_Cooling (NCG)"
    if p.startswith("NCS"):
        return "Stormwater_MS4_or_Industrial (NCS)"
    if p.startswith("NCL"):
        return "Land_Application_NonDischarge (NCL)"
    if p.startswith("NCA"):
        return "Animal_Operation (NCA)"
    return f"Other_or_Geocoding_Error ({p[:3]})"


def resolve_station_comids(df: pd.DataFrame, nldi: NLDI) -> pd.DataFrame:
    coord_cache = {}
    comids, statuses = [], []

    for _, row in df.iterrows():
        sid = str(row["MonitoringLocationIdentifier"]).strip()
        comid, status = pd.NA, "Unresolved"

        if sid.startswith("USGS-"):
            try:
                feat = nldi.getfeature_byid("nwissite", sid)
                if feat is not None and len(feat) > 0 and pd.notna(feat.iloc[0].get("comid")):
                    comid = int(feat.iloc[0]["comid"])
                    status = "USGS station lookup"
            except Exception as e:
                status = f"NLDI USGS lookup error: {e}"

        if pd.isna(comid):
            lat, lon = row["LatitudeMeasure"], row["LongitudeMeasure"]
            if pd.notna(lat) and pd.notna(lon):
                key = (round(float(lat), 5), round(float(lon), 5))
                if key in coord_cache:
                    comid, status = coord_cache[key][0], "Found (cached)"
                else:
                    try:
                        feat = nldi.feature_byloc((float(lon), float(lat)))
                        if feat is not None and len(feat) > 0 and pd.notna(feat.iloc[0].get("comid")):
                            comid = int(feat.iloc[0]["comid"])
                            status = "Coordinate lookup"
                        else:
                            status = "No flowline found"
                    except Exception as e:
                        status = f"Coordinate lookup error: {e}"
                    coord_cache[key] = (comid, status)
                    time.sleep(0.15)

        comids.append(comid)
        statuses.append(status)

    df["NHDPlus_COMID"] = pd.Series(comids, dtype="Int64")
    df["NLDI_Status"] = statuses
    return df


def fetch_exact_nhdplus_hydrology(valid_comids: list) -> pd.DataFrame:
    """
    Retrieve exact USGS NHDPlus flowline attributes via WaterData('nhdflowline_network'):
      - gnis_name: Official USGS hydrography stream name (for exact name-mismatch verification)
      - totdasqkm: Exact cumulative watershed drainage area (sq km)
      - qe_ma: Gage-adjusted mean annual flow (CFS, EROM)
      - qe_08: Gage-adjusted August late-summer flow (CFS, EROM)
      - ve_08: Gage-adjusted August late-summer velocity (ft/s, EROM)
    Zero heuristic multipliers or fallback guesses are used.
    """
    print("Querying USGS NHDPlus Flowline Network for exact EROM monthly flows & drainage areas...")
    records = []
    try:
        wd = WaterData("nhdflowline_network")
        flines = wd.byid("comid", [int(c) for c in valid_comids])
        flines.columns = [c.lower() for c in flines.columns]

        for cid in valid_comids:
            sub = flines[flines["comid"].astype(int) == int(cid)]
            if not sub.empty:
                row = sub.iloc[0]
                records.append({
                    "NHDPlus_COMID": int(cid),
                    "NHDPlus_GNIS_Name": str(row.get("gnis_name", "")).strip(),
                    "NHDPlus_StreamOrder": pd.to_numeric(row.get("streamorde"), errors="coerce"),
                    "watershed_area_sqkm": pd.to_numeric(row.get("totdasqkm"), errors="coerce"),
                    "NHDPlus_Mean_Annual_Flow_CFS": pd.to_numeric(row.get("qe_ma"), errors="coerce"),
                    "NHDPlus_August_Flow_CFS": pd.to_numeric(row.get("qe_08"), errors="coerce"),
                    "NHDPlus_September_Flow_CFS": pd.to_numeric(row.get("qe_09"), errors="coerce"),
                    "NHDPlus_August_Velocity_fps": pd.to_numeric(row.get("ve_08"), errors="coerce"),
                })
    except Exception as e:
        print(f"  -> WARNING: WaterData query failed ({e}).")

    df_h = pd.DataFrame(records) if records else pd.DataFrame({"NHDPlus_COMID": valid_comids})
    df_h["NHDPlus_COMID"] = pd.to_numeric(df_h["NHDPlus_COMID"], errors="coerce").astype("Int64")

    for col in ["watershed_area_sqkm", "NHDPlus_Mean_Annual_Flow_CFS", "NHDPlus_August_Flow_CFS",
                "NHDPlus_September_Flow_CFS", "NHDPlus_August_Velocity_fps"]:
        df_h.loc[df_h[col] < 0, col] = np.nan

    return df_h


# ============================================================
# 4. UPSTREAM NPDES NAVIGATION
# ============================================================
def trace_upstream_npdes_by_comid(unique_comids: list, nldi: NLDI) -> pd.DataFrame:
    records = []
    max_dist = max(DISTANCE_BANDS_KM)

    for i, comid in enumerate(unique_comids, 1):
        print(f"  [{i}/{len(unique_comids)}] Navigating upstream NPDES for COMID {comid}...")
        try:
            npdes_all = nldi.navigate_byid(
                fsource="comid", fid=str(int(comid)),
                navigation="upstreamTributaries", source="npdes", distance=max_dist
            )
        except Exception:
            continue

        if npdes_all is None or len(npdes_all) == 0:
            continue

        id_col = next((c for c in ["identifier", "id", "sourceid"] if c in npdes_all.columns), None)
        if not id_col:
            continue

        first_band = {}
        for d_km in DISTANCE_BANDS_KM:
            try:
                sub = nldi.navigate_byid(
                    fsource="comid", fid=str(int(comid)),
                    navigation="upstreamTributaries", source="npdes", distance=d_km
                )
                if sub is not None and len(sub) > 0 and id_col in sub.columns:
                    for pid in sub[id_col].dropna().astype(str):
                        if pid.strip() not in first_band:
                            first_band[pid.strip()] = d_km
            except Exception:
                pass
            time.sleep(0.10)

        for _, fac in npdes_all.iterrows():
            raw_id = str(fac[id_col]).strip()
            base_id = raw_id[:9]
            upper_km = first_band.get(raw_id, max_dist)
            lower_cands = [d for d in DISTANCE_BANDS_KM if d < upper_km]
            lower_km = max(lower_cands) if lower_cands else 0
            bracket = f"<= {upper_km} km" if lower_km == 0 else f">{lower_km} to <= {upper_km} km"
            geom = getattr(fac, "geometry", None)

            records.append({
                "Station_COMID": int(comid),
                "NPDES_ID": raw_id,
                "Base_Permit_ID": base_id,
                "NC_Permit_Category": classify_nc_permit(base_id),
                "Is_Individual_NC00": base_id.startswith("NC00"),
                "NPDES_COMID": int(fac["comid"]) if "comid" in npdes_all.columns and pd.notna(fac["comid"]) else pd.NA,
                "NPDES_Latitude": geom.y if geom is not None else pd.NA,
                "NPDES_Longitude": geom.x if geom is not None else pd.NA,
                "NLDI_Network_Distance_Lower_km": lower_km,
                "NLDI_Network_Distance_Upper_km": upper_km,
                "NLDI_Network_Distance_Bracket": bracket,
            })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates(subset=["Station_COMID", "Base_Permit_ID"])


# ============================================================
# 5. BAYESIAN & MIXED-EFFECTS HARMONIC CONVERGENCE DIAGNOSTICS
# ============================================================
def run_bayesian_harmonic_gibbs(df: pd.DataFrame, group_col: str, n_chains=4, n_iter=3000, warmup=1000, seed=42):
    """
    4-Chain Conjugate Gibbs Sampler for Hierarchical Harmonic DO Prior:
    DO_ij = (beta_0 + u_0j) + beta_sin * sin(2*pi*DOY/365.25) + beta_cos * cos(2*pi*DOY/365.25) + eps_ij
    Computes exact Gelman-Rubin R-hat and 95% Credible Intervals.
    """
    np.random.seed(seed)
    y = df["DO_mgL"].values
    X = np.column_stack([np.ones(len(df)), df["sin_t"].values, df["cos_t"].values])
    groups, group_idx = np.unique(df[group_col].astype(str), return_inverse=True)
    J, N = len(groups), len(y)
    n_keep = n_iter - warmup
    chains = np.zeros((n_chains, n_keep, 6))

    mu_0 = np.array([6.5, 0.0, 0.0])
    Lambda_0_inv = np.diag([1 / 25.0, 1 / 25.0, 1 / 25.0])
    a_e, b_e, a_u, b_u = 1.0, 1.0, 1.0, 1.0

    for c in range(n_chains):
        beta = np.array([6.0 + np.random.normal(0, 1), np.random.normal(0, 0.5), np.random.normal(0, 0.5)])
        u = np.random.normal(0, 1.0, size=J)
        sigma2_e = np.random.uniform(2.0, 6.0)
        sigma2_u = np.random.uniform(0.5, 3.0)

        for it in range(n_iter):
            resid_fixed = y - X @ beta
            for j in range(J):
                mask = (group_idx == j)
                prec_j = (np.sum(mask) / sigma2_e) + (1.0 / sigma2_u)
                var_j = 1.0 / prec_j
                u[j] = np.random.normal(var_j * (np.sum(resid_fixed[mask]) / sigma2_e), np.sqrt(var_j))

            y_star = y - u[group_idx]
            cov_beta = np.linalg.inv((X.T @ X / sigma2_e) + Lambda_0_inv)
            mean_beta = cov_beta @ ((X.T @ y_star) / sigma2_e + Lambda_0_inv @ mu_0)
            beta = np.random.multivariate_normal(mean_beta, cov_beta)

            resid_total = y - (X @ beta + u[group_idx])
            sigma2_e = invgamma.rvs(a=a_e + N / 2.0, scale=b_e + 0.5 * np.sum(resid_total ** 2))
            sigma2_u = invgamma.rvs(a=a_u + J / 2.0, scale=b_u + 0.5 * np.sum(u ** 2))

            if it >= warmup:
                amp = np.sqrt(beta[1] ** 2 + beta[2] ** 2)
                chains[c, it - warmup, :] = [beta[0], beta[1], beta[2], np.sqrt(sigma2_e), np.sqrt(sigma2_u), amp]

    # Gelman-Rubin R-hat for Mean_DO (idx 0) and Amplitude (idx 5)
    def calc_rhat(idx):
        s = chains[:, :, idx]
        W = np.mean(np.var(s, axis=1, ddof=1))
        B = n_keep * np.var(np.mean(s, axis=1), ddof=1)
        var_hat = ((n_keep - 1.0) / n_keep) * W + (1.0 / n_keep) * B
        return float(np.sqrt(var_hat / W)) if W > 0 else np.nan

    return {
        "Post_Mean_DO": float(np.mean(chains[:, :, 0])),
        "Post_SD_DO": float(np.std(chains[:, :, 0])),
        "Post_Amp": float(np.mean(chains[:, :, 5])),
        "Post_Reach_SD": float(np.mean(chains[:, :, 4])),
        "Post_Resid_SD": float(np.mean(chains[:, :, 3])),
        "Rhat_Mean_DO": round(calc_rhat(0), 4),
        "Rhat_Amp": round(calc_rhat(5), 4),
    }, chains


def evaluate_pooling_and_harmonic_convergence(results: pd.DataFrame) -> pd.DataFrame:
    """
    Test pooling vs. single-station vs. unpooled cohorts on regional_do_raw_data.csv
    and generate Figure 2 showing empirical hydrology + fitted harmonic DO priors.
    """
    if not os.path.exists(RAW_DO_FILE):
        print(f"Skipping harmonic DO convergence test ({RAW_DO_FILE} not found).")
        return pd.DataFrame()

    print("\n" + "=" * 100)
    print("TESTING POOLING STRATEGIES & BAYESIAN HARMONIC CONVERGENCE ON REGIONAL DO DATA")
    print("=" * 100)

    df_do = pd.read_csv(RAW_DO_FILE, low_memory=False)
    df_do["DO_mgL"] = pd.to_numeric(df_do["ResultMeasureValue"], errors="coerce")
    df_do["Date"] = pd.to_datetime(df_do["ActivityStartDate"], errors="coerce")
    df_do = df_do.dropna(subset=["DO_mgL", "Date"])
    df_do = df_do[(df_do["DO_mgL"] >= 0.0) & (df_do["DO_mgL"] <= 20.0)]

    orig_stations = results[results["Screening_Group"] == "Original 33 stations"].copy()
    df_do_33 = df_do.merge(
        orig_stations[["MonitoringLocationIdentifier", "MonitoringLocationName", "NHDPlus_COMID", "Cohort", "N_NC00_Individual_Unique", "N_NC00_Within_10km"]],
        on="MonitoringLocationIdentifier", how="inner"
    )
    df_do_33["DOY"] = df_do_33["Date"].dt.dayofyear
    df_do_33["sin_t"] = np.sin(2 * np.pi * df_do_33["DOY"] / 365.25)
    df_do_33["cos_t"] = np.cos(2 * np.pi * df_do_33["DOY"] / 365.25)

    # Deduplicate exact same-day duplicate uploads on the same COMID
    df_dedup = df_do_33.drop_duplicates(subset=["NHDPlus_COMID", "Date", "DO_mgL"]).copy()

    # Define candidate cohorts
    c_unimp_unpooled = df_do_33[df_do_33["Cohort"] == "Unimpaired Sw (0 NC00)"].copy()
    c_unimp_pooled = df_dedup[df_dedup["Cohort"] == "Unimpaired Sw (0 NC00)"].copy()

    longest_sids = (
        c_unimp_pooled.groupby(["NHDPlus_COMID", "MonitoringLocationIdentifier"])
        .size().reset_index(name="n").sort_values("n", ascending=False)
        .drop_duplicates("NHDPlus_COMID")["MonitoringLocationIdentifier"].tolist()
    )
    c_unimp_single = c_unimp_pooled[c_unimp_pooled["MonitoringLocationIdentifier"].isin(longest_sids)].copy()

    # Minimally Impacted: 0 NC00 within 10 km OR Goshen/Kornegay non-POTW reaches
    c_min_imp = df_dedup[
        (df_dedup["Cohort"] == "Unimpaired Sw (0 NC00)") |
        (df_dedup["NHDPlus_COMID"].isin([10525101, 3350707, 10524689]))
    ].copy()

    c_cont = df_dedup[df_dedup["Cohort"] == "Contentnea Creek"].copy()

    cohorts_to_test = [
        ("1A. Unimpaired Sw (0 NC00) - Unpooled Station IDs", c_unimp_unpooled, "MonitoringLocationIdentifier"),
        ("1B. Unimpaired Sw (0 NC00) - Single Station ID per Reach", c_unimp_single, "NHDPlus_COMID"),
        ("1C. Unimpaired Sw (0 NC00) - Pooled by COMID [PRIMARY]", c_unimp_pooled, "NHDPlus_COMID"),
        ("2. Minimally Impacted Sw (No POTW <=10km) - Pooled by COMID", c_min_imp, "NHDPlus_COMID"),
        ("3. All Original 33 Reaches - Pooled by COMID", df_dedup, "NHDPlus_COMID"),
        ("4. Contentnea Creek WQP Reaches - Pooled by COMID", c_cont, "NHDPlus_COMID"),
    ]

    summary_rows = []
    fitted_curves = {}

    for label, sub_df, g_col in cohorts_to_test:
        clean_df = sub_df.dropna(subset=["DO_mgL", "sin_t", "cos_t", g_col]).copy().reset_index(drop=True)
        clean_df["grp"] = clean_df[g_col].astype(str)
        n_groups = clean_df["grp"].nunique()
        n_obs = len(clean_df)
        min_n = int(clean_df.groupby("grp").size().min())

        # Test Diagonal Random-Slopes MixedLM
        with warnings.catch_warnings(record=True) as w_diag:
            warnings.simplefilter("always")
            vc = {"sin_slope": "0 + sin_t", "cos_slope": "0 + cos_t"}
            m_diag = sm.MixedLM.from_formula("DO_mgL ~ sin_t + cos_t", clean_df, groups=clean_df["grp"], re_formula="1", vc_formula=vc).fit()

        # Run 4-Chain Bayesian Gibbs Sampler
        bayes_res, _ = run_bayesian_harmonic_gibbs(clean_df, "grp")

        summary_rows.append({
            "Cohort_and_Grouping": label,
            "Groups_J": n_groups,
            "Total_Obs_N": n_obs,
            "Min_Obs_Per_Group": min_n,
            "Diag_RandomSlopes_Converged": m_diag.converged and (len(w_diag) == 0),
            "Bayesian_Mean_DO_mgL": f"{bayes_res['Post_Mean_DO']:.2f} ± {bayes_res['Post_SD_DO']:.2f}",
            "Seasonal_Amplitude_mgL": round(bayes_res["Post_Amp"], 2),
            "Reach_SD_mgL": round(bayes_res["Post_Reach_SD"], 2),
            "Residual_SD_mgL": round(bayes_res["Post_Resid_SD"], 2),
            "Gelman_Rubin_Rhat_Mean": bayes_res["Rhat_Mean_DO"],
            "Gelman_Rubin_Rhat_Amp": bayes_res["Rhat_Amp"],
        })

        # Store seasonal curve for plotting
        doy_grid = np.linspace(1, 365, 365)
        b0 = m_diag.fe_params["Intercept"]
        bsin = m_diag.fe_params["sin_t"]
        bcos = m_diag.fe_params["cos_t"]
        fitted_curves[label] = (
            doy_grid,
            b0 + bsin * np.sin(2 * np.pi * doy_grid / 365.25) + bcos * np.cos(2 * np.pi * doy_grid / 365.25)
        )

    df_harm = pd.DataFrame(summary_rows)
    print(df_harm.to_string(index=False))
    df_harm.to_csv(OUTPUT_HARMONIC_TABLE, index=False)
    print(f"\n[SAVED] Harmonic convergence diagnostic table -> {OUTPUT_HARMONIC_TABLE}")
    return df_harm, fitted_curves, df_dedup


# ============================================================
# 6. STATISTICAL COMPARISONS & FIGURES (ZERO HEURISTICS)
# ============================================================
def safe_kruskal(*groups):
    valid = [g.dropna().values for g in groups if len(g.dropna()) > 0]
    if len(valid) < 2 or len(np.unique(np.concatenate(valid))) <= 1:
        return np.nan
    try:
        return float(kruskal(*valid)[1])
    except Exception:
        return np.nan


def safe_mw(v1, v2):
    a, b = v1.dropna().values, v2.dropna().values
    if len(a) == 0 or len(b) == 0 or len(np.unique(np.concatenate([a, b]))) <= 1:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided")[1])
    except Exception:
        return np.nan


def plot_safe_box_and_strip(ax, df, col, cohort_order, palette, title, ylabel, fmt=".2f"):
    sub_df = df.dropna(subset=[col, "Cohort"]).copy()
    if sub_df.empty:
        ax.text(0.5, 0.5, f"No valid data for {col}", ha="center", va="center", transform=ax.transAxes)
        return

    sns.boxplot(
        data=sub_df, x="Cohort", y=col, hue="Cohort",
        order=cohort_order, palette=palette, width=0.45,
        ax=ax, boxprops=dict(alpha=0.75), showfliers=False, legend=False
    )
    sns.stripplot(
        data=sub_df, x="Cohort", y=col, hue="Cohort",
        order=cohort_order, palette=palette, size=8, jitter=0.15,
        edgecolor="black", linewidth=1.0, alpha=0.9, ax=ax, legend=False
    )

    for c_idx, c_name in enumerate(cohort_order):
        vals = sub_df.loc[sub_df["Cohort"] == c_name, col].dropna()
        if not vals.empty:
            ax.text(
                c_idx, vals.median(), f"Med: {vals.median():{fmt}}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8)
            )

    g1 = sub_df.loc[sub_df["Cohort"] == cohort_order[0], col]
    g2 = sub_df.loc[sub_df["Cohort"] == cohort_order[1], col]
    g3 = sub_df.loc[sub_df["Cohort"] == cohort_order[2], col]
    p_mw = safe_mw(g1, g3)
    p_kw = safe_kruskal(g1, g2, g3)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        f"Unimpaired Sw\n(n={len(g1)})",
        f"Impacted Sw\n(n={len(g2)})",
        f"Contentnea Crk\n(n={len(g3)})"
    ], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.text(
        0.04, 0.94, f"MW p (Unimp vs Cont) = {p_mw:.4f}\nKruskal-Wallis p = {p_kw:.4f}",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )


def generate_tables_and_figures(results: pd.DataFrame, fitted_curves: dict, df_dedup: pd.DataFrame):
    u_reaches = results.dropna(subset=["NHDPlus_COMID"]).copy()
    u_reaches = u_reaches[~u_reaches["Mainstem_GNIS_Mismatch"]].drop_duplicates(subset=["NHDPlus_COMID"]).copy()

    cohort_order = ["Unimpaired Sw (0 NC00)", "WWTP-Impacted Sw (>0 NC00)", "Contentnea Creek"]
    palette = {"Unimpaired Sw (0 NC00)": "#2b8cbe", "WWTP-Impacted Sw (>0 NC00)": "#fdae61", "Contentnea Creek": "#d73027"}

    metrics = [
        ("developed_pct", "Developed Land (%)"),
        ("developed_med_high_pct", "Med+High Impervious Urban (%)"),
        ("population_density_2010_per_sqkm", "Population Density (people/km²)"),
        ("N_NC00_Individual_Unique", "Upstream Individual Permits (NC00)"),
        ("road_density_km_per_sqkm", "Road Density (km/km²)"),
        ("agriculture_pct", "Agricultural Land (%)"),
        ("NC00_Per_100_sqkm", "Area-Normalized NC00 Permits (per 100 km²)"),
        ("NHDPlus_August_Flow_CFS", "USGS NHDPlus August Mean Flow (qe_08, CFS)"),
        ("NHDPlus_August_Velocity_fps", "USGS NHDPlus August Velocity (ve_08, ft/s)"),
        ("baseflow_index_pct", "USGS Baseflow Index (BFI %)"),
    ]

    g_unimp = u_reaches[u_reaches["Cohort"] == "Unimpaired Sw (0 NC00)"]
    g_imp = u_reaches[u_reaches["Cohort"] == "WWTP-Impacted Sw (>0 NC00)"]
    g_cont = u_reaches[u_reaches["Cohort"] == "Contentnea Creek"]

    stat_rows = []
    for col, label in metrics:
        if col not in u_reaches.columns or u_reaches[col].notna().sum() == 0:
            continue
        v_unimp, v_imp, v_cont = g_unimp[col].dropna(), g_imp[col].dropna(), g_cont[col].dropna()
        stat_rows.append({
            "Metric": label,
            "Unimpaired_Sw_Mean_SD": f"{v_unimp.mean():.2f} ± {v_unimp.std():.2f}",
            "Unimpaired_Sw_Median_IQR": f"{v_unimp.median():.2f} ({v_unimp.quantile(0.25):.2f}-{v_unimp.quantile(0.75):.2f})",
            "Impacted_Sw_Mean_SD": f"{v_imp.mean():.2f} ± {v_imp.std():.2f}",
            "Impacted_Sw_Median_IQR": f"{v_imp.median():.2f} ({v_imp.quantile(0.25):.2f}-{v_imp.quantile(0.75):.2f})",
            "Contentnea_Mean_SD": f"{v_cont.mean():.2f} ± {v_cont.std():.2f}",
            "Contentnea_Median_IQR": f"{v_cont.median():.2f} ({v_cont.quantile(0.25):.2f}-{v_cont.quantile(0.75):.2f})",
            "MW_p_Unimp_vs_Contentnea": round(safe_mw(v_unimp, v_cont), 4),
            "MW_p_Unimp_vs_ImpactedSw": round(safe_mw(v_unimp, v_imp), 4),
            "Kruskal_Wallis_p": round(safe_kruskal(v_unimp, v_imp, v_cont), 4),
        })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(OUTPUT_STATS_TABLE, index=False)

    # FIGURE 1: 6-Panel Watershed Disturbance Boxplots (Keep untouched)
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9.5), dpi=300)
    axes1 = axes1.flatten()
    for idx, (col, label) in enumerate(metrics[:6]):
        plot_safe_box_and_strip(axes1[idx], u_reaches, col, cohort_order, palette, f"{chr(65+idx)}. {label}", label)
    plt.suptitle("Watershed Disturbance & Point-Source Comparison Across Unique Stream Reaches (COMIDs)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_BOXPLOT_FIG, dpi=300)
    plt.close()

    # FIGURE 2: Empirical NHDPlus EROM Hydrology & Fitted Seasonal Harmonic DO Priors (Zero Heuristics)
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes2 = axes2.flatten()

    plot_safe_box_and_strip(
        axes2[0], u_reaches, "NC00_Per_100_sqkm", cohort_order, palette,
        "A. Area-Normalized Point-Source Density (NC00 / 100 km²)", "Individual NC00 Permits per 100 km²"
    )
    plot_safe_box_and_strip(
        axes2[1], u_reaches, "NHDPlus_August_Flow_CFS", cohort_order, palette,
        "B. USGS NHDPlus EROM August Flow (qe_08, CFS)", "August Gage-Adjusted Flow (CFS)"
    )
    plot_safe_box_and_strip(
        axes2[2], u_reaches, "NHDPlus_August_Velocity_fps", cohort_order, palette,
        "C. USGS NHDPlus EROM August Velocity (ve_08, ft/s)", "August Gage-Adjusted Velocity (ft/s)"
    )

    # Panel D: Fitted Hierarchical Harmonic DO Curves across Pooling Options
    ax = axes2[3]
    if fitted_curves:
        curve_styles = [
            ("1C. Unimpaired Sw (0 NC00) - Pooled by COMID [PRIMARY]", "#2b8cbe", "-", "Unimpaired Sw Pooled (J=7, N=704)"),
            ("1B. Unimpaired Sw (0 NC00) - Single Station ID per Reach", "#2b8cbe", "--", "Unimpaired Sw Single ID (J=7, N=497)"),
            ("2. Minimally Impacted Sw (No POTW <=10km) - Pooled by COMID", "#fdae61", "-", "Minimally Impacted Pooled (J=10, N=887)"),
            ("4. Contentnea Creek WQP Reaches - Pooled by COMID", "#d73027", "-", "Contentnea Creek Pooled (J=2, N=942)"),
        ]
        for key, color, ls, lbl in curve_styles:
            if key in fitted_curves:
                doy_g, y_pred = fitted_curves[key]
                ax.plot(doy_g, y_pred, color=color, linestyle=ls, linewidth=2.3, label=lbl)
        ax.axhline(5.0, color="black", linestyle=":", linewidth=1.1, label="NC Class C Standard (5.0 mg/L)")
        ax.set_title("D. Fitted Hierarchical Harmonic DO Priors by Cohort", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Day of Year (DOY)", fontsize=10)
        ax.set_ylabel("Predicted Baseline DO (mg/L)", fontsize=10)
        ax.set_xlim(1, 365)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Empirical NHDPlus Summer Hydrology (EROM) & Harmonic Headwater DO Priors", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_HYDRO_DO_FIG, dpi=300)
    plt.close()
    print(f"[SAVED] Figure 2 (Empirical Hydrology & Harmonic Priors) -> {OUTPUT_HYDRO_DO_FIG}")


# ============================================================
# 7. MAIN EXECUTION
# ============================================================
def main():
    orig = pd.read_csv(INPUT_FILE)
    orig["Screening_Group"] = "Original 33 stations"
    orig["Station_Source"] = "NC WQP"
    orig["LatitudeMeasure"] = pd.to_numeric(orig["LatitudeMeasure"], errors="coerce")
    orig["LongitudeMeasure"] = pd.to_numeric(orig["LongitudeMeasure"], errors="coerce")

    contentnea_usgs = discover_contentnea_usgs_stations(state="NC")
    stations = pd.concat([orig, contentnea_usgs], ignore_index=True)

    nldi = NLDI()
    stations = resolve_station_comids(stations, nldi)
    valid_comids = stations["NHDPlus_COMID"].dropna().astype(int).unique().tolist()

    metrics = [
        "pcturbop2019", "pcturblo2019", "pcturbmd2019", "pcturbhi2019",
        "pcthay2019", "pctcrop2019", "pctdecid2019", "pctconif2019", "pctmxfst2019",
        "pcthbwet2019", "pctwdwet2019", "rddens", "damdens", "npdesdens", "popden2010", "bfi"
    ]
    sc_df = streamcat(metric_names=metrics, metric_areas="ws", comids=valid_comids)
    sc_df.columns = [c.strip().lower() for c in sc_df.columns]
    sc_df["NHDPlus_COMID"] = pd.to_numeric(sc_df["comid"], errors="coerce").astype("Int64")

    sc_df["developed_pct"] = sc_df[["pcturbop2019ws", "pcturblo2019ws", "pcturbmd2019ws", "pcturbhi2019ws"]].sum(axis=1)
    sc_df["developed_med_high_pct"] = sc_df[["pcturbmd2019ws", "pcturbhi2019ws"]].sum(axis=1)
    sc_df["agriculture_pct"] = sc_df[["pcthay2019ws", "pctcrop2019ws"]].sum(axis=1)
    sc_df["forest_pct"] = sc_df[["pctdecid2019ws", "pctconif2019ws", "pctmxfst2019ws"]].sum(axis=1)
    sc_df["wetland_pct"] = sc_df[["pcthbwet2019ws", "pctwdwet2019ws"]].sum(axis=1)
    sc_df["natural_cover_pct"] = sc_df["forest_pct"] + sc_df["wetland_pct"]

    sc_df = sc_df.rename(columns={
        "rddensws": "road_density_km_per_sqkm",
        "damdensws": "dam_density_per_sqkm",
        "npdesdensws": "npdes_density_per_sqkm",
        "popden2010ws": "population_density_2010_per_sqkm",
        "bfiws": "baseflow_index_pct"
    })

    df_hydro = fetch_exact_nhdplus_hydrology(valid_comids)
    npdes_comid_df = trace_upstream_npdes_by_comid(valid_comids, nldi)

    if not npdes_comid_df.empty:
        cid_summary = []
        for cid in valid_comids:
            sub = npdes_comid_df[npdes_comid_df["Station_COMID"] == cid]
            nc00 = sub[sub["Is_Individual_NC00"]]
            cid_summary.append({
                "NHDPlus_COMID": cid,
                "N_All_Permits_Unique": sub["Base_Permit_ID"].nunique(),
                "N_NC00_Individual_Unique": nc00["Base_Permit_ID"].nunique(),
                "N_NC00_Within_10km": nc00[nc00["NLDI_Network_Distance_Upper_km"] <= 10]["Base_Permit_ID"].nunique(),
                "Nearest_NC00_Upper_km": nc00["NLDI_Network_Distance_Upper_km"].min() if not nc00.empty else np.nan,
            })
        df_cid_sum = pd.DataFrame(cid_summary)
    else:
        df_cid_sum = pd.DataFrame({"NHDPlus_COMID": valid_comids})

    results = (
        stations.merge(sc_df, on="NHDPlus_COMID", how="left")
        .merge(df_hydro, on="NHDPlus_COMID", how="left")
        .merge(df_cid_sum, on="NHDPlus_COMID", how="left")
    )
    for col in ["N_All_Permits_Unique", "N_NC00_Individual_Unique", "N_NC00_Within_10km"]:
        results[col] = results[col].fillna(0).astype(int)

    results["NC00_Per_100_sqkm"] = np.where(
        results["watershed_area_sqkm"] > 0,
        (results["N_NC00_Individual_Unique"] / results["watershed_area_sqkm"]) * 100.0,
        np.nan
    )

    # Exact GNIS Name Verification: Flag if a tributary creek/swamp snaps to a different named River
    def detect_gnis_name_mismatch(row):
        st_name = str(row.get("MonitoringLocationName", "")).upper()
        gnis = str(row.get("NHDPlus_GNIS_Name", "")).upper()
        if "RIVER" in gnis and not any(r in st_name for r in ["RIV", "RIVER"]):
            return True
        return False

    results["Mainstem_GNIS_Mismatch"] = results.apply(detect_gnis_name_mismatch, axis=1)

    contentnea_comids = set(
        results.loc[
            results["MonitoringLocationName"].astype(str).str.contains("CONTENTNEA", case=False, na=False) |
            (results["Screening_Group"] == "Contentnea longitudinal reference"),
            "NHDPlus_COMID"
        ].dropna().astype(int)
    )

    def assign_cohort(row):
        cid = row["NHDPlus_COMID"]
        if pd.isna(cid):
            return "Unresolved"
        if row["Mainstem_GNIS_Mismatch"]:
            return "Mainstem GNIS Mismatch"
        if int(cid) in contentnea_comids or "CONTENTNEA" in str(row["MonitoringLocationName"]).upper():
            return "Contentnea Creek"
        if row["N_NC00_Individual_Unique"] == 0 and row["developed_pct"] < DEVELOPED_THRESHOLD:
            return "Unimpaired Sw (0 NC00)"
        return "WWTP-Impacted Sw (>0 NC00)"

    results["Cohort"] = results.apply(assign_cohort, axis=1)
    results.to_csv(OUTPUT_STATION_FILE, index=False)

    _, fitted_curves, df_dedup = evaluate_pooling_and_harmonic_convergence(results)
    generate_tables_and_figures(results, fitted_curves, df_dedup)


if __name__ == "__main__":
    main()