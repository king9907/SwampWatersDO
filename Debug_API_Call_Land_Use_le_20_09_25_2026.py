# ============================================================
# MINIMALLY IMPACTED STREAM SCREENING
#
# Original 33 stations + Contentnea Creek reference stations
#
# Method:
#   1. Map stations to NHDPlus COMIDs using USGS NLDI
#   2. Retrieve StreamCat watershed metrics
#   3. Calculate:
#        - developed land use
#        - agricultural land use
#        - total anthropogenic land-use proxy
#        - forest cover
#        - wetland cover
#        - road density
#        - road-stream crossing density
#        - dam density
#        - anthropogenic barrier density
#        - NPDES density
#        - mine density
#        - population density
#        - canal/ditch/pipeline density
#   4. Apply the <20% developed and <20% anthropogenic
#      land-use screening criteria
#   5. Save one combined results CSV
# ============================================================


import time
import pandas as pd

from pynhd import NLDI
from pynhd.nhdplus_derived import streamcat, StreamCat


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = r"Final_33_Golden_Stations.csv"

OUTPUT_FILE = (
    r"Final_Stations_Minimally_Impacted_Screening.csv"
)

NLCD_YEAR = 2019

# Your screening thresholds
DEVELOPED_THRESHOLD = 20.0
ANTHROPOGENIC_THRESHOLD = 20.0

REQUEST_DELAY = 0.20


# ============================================================
# 1. READ ORIGINAL STATIONS
# ============================================================

print("=" * 80)
print("READING ORIGINAL STATION FILE")
print("=" * 80)

stations = pd.read_csv(INPUT_FILE)

print(f"Found {len(stations)} original stations.")


# ============================================================
# 2. STANDARDIZE COORDINATES
# ============================================================

stations["LatitudeMeasure"] = pd.to_numeric(
    stations["LatitudeMeasure"],
    errors="coerce"
)

stations["LongitudeMeasure"] = pd.to_numeric(
    stations["LongitudeMeasure"],
    errors="coerce"
)


# ============================================================
# 3. ADD CONTENTNEA CREEK REFERENCE STATIONS
#
# These are USGS stations and will be resolved through NLDI
# using the official USGS station identifier rather than
# relying on manually entered coordinates.
#
# 02091788 is a historical USGS station identified by USGS
# documentation as Contentnea Creek near Quinley, at the mouth.
# ============================================================

contentnea_stations = pd.DataFrame([
    {
        "MonitoringLocationIdentifier":
            "USGS-02090380",

        "MonitoringLocationName":
            "CONTENTNEA CREEK NEAR LUCAMA, NC",

        "LatitudeMeasure":
            pd.NA,

        "LongitudeMeasure":
            pd.NA,

        "Screening_Group":
            "Contentnea longitudinal reference",

        "Station_Source":
            "USGS"
    },

    {
        "MonitoringLocationIdentifier":
            "USGS-02090500",

        "MonitoringLocationName":
            "CONTENTNEA CREEK NEAR WILSON, NC",

        "LatitudeMeasure":
            pd.NA,

        "LongitudeMeasure":
            pd.NA,

        "Screening_Group":
            "Contentnea longitudinal reference",

        "Station_Source":
            "USGS"
    },

    {
        "MonitoringLocationIdentifier":
            "USGS-02090519",

        "MonitoringLocationName":
            "CONTENTNEA CREEK NR EVANSDALE, NC",

        "LatitudeMeasure":
            pd.NA,

        "LongitudeMeasure":
            pd.NA,

        "Screening_Group":
            "Contentnea longitudinal reference",

        "Station_Source":
            "USGS"
    },

    {
        "MonitoringLocationIdentifier":
            "USGS-02091500",

        "MonitoringLocationName":
            "CONTENTNEA CREEK AT HOOKERTON, NC",

        "LatitudeMeasure":
            pd.NA,

        "LongitudeMeasure":
            pd.NA,

        "Screening_Group":
            "Contentnea longitudinal reference",

        "Station_Source":
            "USGS"
    },

    {
        "MonitoringLocationIdentifier":
            "USGS-02091764",

        "MonitoringLocationName":
            "CONTENTNEA CREEK AT GRIFTON, NC",

        "LatitudeMeasure":
            pd.NA,

        "LongitudeMeasure":
            pd.NA,

        "Screening_Group":
            "Contentnea longitudinal reference",

        "Station_Source":
            "USGS"
    },

    {
        "MonitoringLocationIdentifier":
            "USGS-02091788",

        "MonitoringLocationName":
            "CONTENTNEA CREEK NEAR QUINLEY, NC (MOUTH)",

        "LatitudeMeasure":
            pd.NA,

        "LongitudeMeasure":
            pd.NA,

        "Screening_Group":
            "Contentnea longitudinal reference",

        "Station_Source":
            "USGS"
    }
])


# ============================================================
# 4. MAKE ORIGINAL STATIONS COMPATIBLE WITH EXTRA STATIONS
# ============================================================

if "Screening_Group" not in stations.columns:
    stations["Screening_Group"] = "Original 33 stations"

else:
    stations["Screening_Group"] = (
        stations["Screening_Group"]
        .fillna("Original 33 stations")
    )


if "Station_Source" not in stations.columns:
    stations["Station_Source"] = "NC WQP"

else:
    stations["Station_Source"] = (
        stations["Station_Source"]
        .fillna("NC WQP")
    )


# Add any columns that exist only in one table
for col in contentnea_stations.columns:

    if col not in stations.columns:

        stations[col] = pd.NA

for col in stations.columns:

    if col not in contentnea_stations.columns:

        contentnea_stations[col] = pd.NA


# Force identical column order
contentnea_stations = contentnea_stations[
    stations.columns
]


# Combine
stations = pd.concat(
    [
        stations,
        contentnea_stations
    ],
    ignore_index=True
)


print(
    f"Total stations to evaluate: "
    f"{len(stations)}"
)


# ============================================================
# 5. INITIALIZE NLDI
# ============================================================

print("\n" + "=" * 80)
print("CONNECTING TO USGS NLDI")
print("=" * 80)

nldi = NLDI()


# ============================================================
# 6. FUNCTION TO FIND COMID
# ============================================================

def get_comid(row):
    """
    Resolve an NHDPlus COMID.

    For USGS stations:
        use the official NWIS station identifier.

    For NC WQP stations:
        use latitude/longitude.
    """

    site_id = str(
        row["MonitoringLocationIdentifier"]
    )

    # --------------------------------------------------------
    # USGS station
    # --------------------------------------------------------

    if site_id.startswith("USGS-"):

        usgs_id = site_id.replace(
            "USGS-",
            "",
            1
        )

        try:

            feature = nldi.getfeature_byid(
                "nwissite",
                f"USGS-{usgs_id}"
            )

            if (
                feature is not None
                and len(feature) > 0
            ):

                comid = feature.iloc[0]["comid"]

                if pd.notna(comid):

                    return (
                        int(comid),
                        "USGS station lookup"
                    )

            return (
                pd.NA,
                "No NLDI feature returned"
            )

        except Exception as e:

            return (
                pd.NA,
                f"NLDI error: {e}"
            )


    # --------------------------------------------------------
    # NC WQP station
    # --------------------------------------------------------

    lat = row["LatitudeMeasure"]
    lon = row["LongitudeMeasure"]

    if pd.isna(lat) or pd.isna(lon):

        return (
            pd.NA,
            "Missing coordinates"
        )

    try:

        # NLDI uses (longitude, latitude)
        feature = nldi.feature_byloc(
            (
                float(lon),
                float(lat)
            )
        )

        if (
            feature is None
            or len(feature) == 0
        ):

            return (
                pd.NA,
                "No flowline found"
            )

        comid = feature.iloc[0]["comid"]

        if pd.isna(comid):

            return (
                pd.NA,
                "No COMID returned"
            )

        return (
            int(comid),
            "Coordinate lookup"
        )

    except Exception as e:

        return (
            pd.NA,
            f"NLDI error: {e}"
        )


# ============================================================
# 7. FIND NHDPLUS COMIDs
# ============================================================

print("\n" + "=" * 80)
print("FINDING NHDPLUS COMIDs")
print("=" * 80)


comid_values = []
comid_status = []

comid_cache = {}


for i, row in stations.iterrows():

    site_id = row[
        "MonitoringLocationIdentifier"
    ]

    print(
        f"[{i + 1}/{len(stations)}] "
        f"{site_id}"
    )


    # --------------------------------------------------------
    # Coordinate cache for NC WQP stations
    # --------------------------------------------------------

    site_string = str(site_id)

    if not site_string.startswith("USGS-"):

        lat = row["LatitudeMeasure"]
        lon = row["LongitudeMeasure"]

        if (
            pd.notna(lat)
            and pd.notna(lon)
        ):

            coordinate_key = (
                round(float(lat), 5),
                round(float(lon), 5)
            )

            if coordinate_key in comid_cache:

                comid, status = (
                    comid_cache[coordinate_key]
                )

                comid_values.append(comid)

                comid_status.append(
                    "Found (cached)"
                )

                print(
                    f"    COMID = {comid}"
                )

                continue


            comid, status = get_comid(row)

            comid_values.append(comid)
            comid_status.append(status)

            if pd.notna(comid):

                comid_cache[
                    coordinate_key
                ] = (
                    comid,
                    status
                )

                print(
                    f"    COMID = {comid}"
                )

            else:

                print(
                    f"    {status}"
                )

            time.sleep(REQUEST_DELAY)

            continue


    # --------------------------------------------------------
    # USGS station lookup
    # --------------------------------------------------------

    comid, status = get_comid(row)

    comid_values.append(comid)
    comid_status.append(status)

    if pd.notna(comid):

        print(
            f"    COMID = {comid}"
        )

    else:

        print(
            f"    {status}"
        )

    time.sleep(REQUEST_DELAY)


stations["NHDPlus_COMID"] = comid_values

stations["NLDI_Status"] = comid_status


# ============================================================
# 8. REPORT NLDI RESULTS
# ============================================================

valid_comids = (
    stations["NHDPlus_COMID"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)


print("\n" + "=" * 80)

print(
    f"Successfully identified "
    f"{len(valid_comids)} unique NHDPlus COMIDs."
)

failed_nldi = stations[
    stations["NHDPlus_COMID"].isna()
]

if len(failed_nldi) > 0:

    print(
        f"\nWARNING: "
        f"{len(failed_nldi)} stations could not be "
        f"resolved through NLDI:"
    )

    print(
        failed_nldi[
            [
                "MonitoringLocationIdentifier",
                "MonitoringLocationName",
                "NLDI_Status"
            ]
        ].to_string(index=False)
    )


if len(valid_comids) == 0:

    raise RuntimeError(
        "No valid NHDPlus COMIDs were found."
    )


# ============================================================
# 9. INITIALIZE STREAMCAT
# ============================================================

print("\n" + "=" * 80)
print("CHECKING STREAMCAT METRICS")
print("=" * 80)

sc = StreamCat()

valid_names = {
    str(name).lower()
    for name in sc.valid_names
}


# ============================================================
# 10. DEFINE METRICS
#
# We request:
#
# REQUIRED:
#   2019 developed classes
#   2019 hay/pasture
#   2019 cultivated crops
#
# CONTEXT:
#   forest
#   wetlands
#
# ANTHROPOGENIC DISTURBANCE:
#   roads
#   road-stream crossings
#   dams
#   anthropogenic barriers
#   NPDES sites
#   mines
#   population
#   canals/ditches/pipelines
# ============================================================

required_metrics = [

    "pcturbop2019",
    "pcturblo2019",
    "pcturbmd2019",
    "pcturbhi2019",

    "pcthay2019",
    "pctcrop2019",
]


optional_metrics = [

    # Forest
    "pctdecid2019",
    "pctconif2019",
    "pctmxfst2019",

    # Wetlands
    "pcthbwet2019",
    "pctwdwet2019",

    # Open water
    "pctow2019",

    # Infrastructure / development
    "rddens",
    "rdcrs",

    # Dams and barriers
    "damdens",
    "nabd_dens",

    # Regulated / industrial activity
    "npdesdens",
    "tridens",
    "minedens",

    # Human population
    "popden2010",

    # Hydrologic modification
    "canaldens",
]


# ============================================================
# 11. VERIFY REQUIRED METRICS
# ============================================================

missing_required = [
    metric
    for metric in required_metrics
    if metric.lower() not in valid_names
]

if missing_required:

    print(
        "\nAVAILABLE STREAMCAT METRICS WERE CHECKED, "
        "BUT THESE REQUIRED METRICS WERE NOT FOUND:"
    )

    print(
        "\n".join(missing_required)
    )

    raise RuntimeError(
        "Required StreamCat metrics are missing."
    )


# ============================================================
# 12. SELECT OPTIONAL METRICS THAT EXIST
# ============================================================

available_optional = [
    metric
    for metric in optional_metrics
    if metric.lower() in valid_names
]

missing_optional = [
    metric
    for metric in optional_metrics
    if metric.lower() not in valid_names
]


print("\nRequired metrics:")
for metric in required_metrics:
    print(f"    {metric}")


print("\nOptional metrics available:")
for metric in available_optional:
    print(f"    {metric}")


if missing_optional:

    print("\nOptional metrics unavailable:")
    for metric in missing_optional:
        print(f"    {metric}")


metrics_to_request = (
    required_metrics
    + available_optional
)


# ============================================================
# 13. REQUEST STREAMCAT WATERSHED DATA
# ============================================================

print("\n" + "=" * 80)
print("REQUESTING STREAMCAT WATERSHED DATA")
print("=" * 80)

print(
    f"Requesting {len(metrics_to_request)} metrics "
    f"for {len(valid_comids)} COMIDs..."
)


try:

    streamcat_data = streamcat(
        metric_names=metrics_to_request,
        metric_areas="ws",
        comids=valid_comids
    )

except Exception as e:

    raise RuntimeError(
        "\nStreamCat request failed.\n"
        f"Original error: {e}"
    ) from e


print(
    f"Received StreamCat data for "
    f"{len(streamcat_data)} COMIDs."
)


# ============================================================
# 14. STANDARDIZE STREAMCAT COLUMN NAMES
# ============================================================

streamcat_data.columns = [
    str(col).strip().lower()
    for col in streamcat_data.columns
]


streamcat_data["comid"] = pd.to_numeric(
    streamcat_data["comid"],
    errors="coerce"
).astype("Int64")


# ============================================================
# 15. CHECK THE FOUR DEVELOPED CATEGORIES
# ============================================================

developed_columns = [
    "pcturbop2019ws",
    "pcturblo2019ws",
    "pcturbmd2019ws",
    "pcturbhi2019ws",
]


missing_developed = [
    col
    for col in developed_columns
    if col not in streamcat_data.columns
]


if missing_developed:

    print(
        "\nSTREAMCAT RETURNED THESE COLUMNS:"
    )

    print(
        streamcat_data.columns.tolist()
    )

    raise RuntimeError(
        "\nExpected developed-land-use columns "
        "were not returned:\n"
        + "\n".join(missing_developed)
    )


# ============================================================
# 16. CALCULATE DEVELOPED LAND USE
# ============================================================

streamcat_data["developed_open_space_pct"] = (
    streamcat_data["pcturbop2019ws"]
)

streamcat_data["developed_low_intensity_pct"] = (
    streamcat_data["pcturblo2019ws"]
)

streamcat_data["developed_medium_intensity_pct"] = (
    streamcat_data["pcturbmd2019ws"]
)

streamcat_data["developed_high_intensity_pct"] = (
    streamcat_data["pcturbhi2019ws"]
)


streamcat_data["developed_pct"] = (
    streamcat_data[developed_columns]
    .sum(axis=1, min_count=4)
)


# ============================================================
# 17. CALCULATE AGRICULTURAL LAND USE
# ============================================================

streamcat_data["hay_pasture_pct"] = (
    streamcat_data["pcthay2019ws"]
)

streamcat_data["cultivated_crop_pct"] = (
    streamcat_data["pctcrop2019ws"]
)


streamcat_data["agriculture_pct"] = (
    streamcat_data["hay_pasture_pct"]
    + streamcat_data["cultivated_crop_pct"]
)


# ============================================================
# 18. CALCULATE TOTAL ANTHROPOGENIC LAND-USE PROXY
#
# Defined here as:
#
#   developed
#   +
#   hay/pasture
#   +
#   cultivated crops
#
# This is an operational screening definition, not an EPA
# definition of "minimally impacted."
# ============================================================

streamcat_data["anthropogenic_land_use_pct"] = (
    streamcat_data["developed_pct"]
    + streamcat_data["agriculture_pct"]
)


# ============================================================
# 19. FOREST COVER
# ============================================================

forest_columns = [
    "pctdecid2019ws",
    "pctconif2019ws",
    "pctmxfst2019ws",
]


if all(
    col in streamcat_data.columns
    for col in forest_columns
):

    streamcat_data["forest_pct"] = (
        streamcat_data[forest_columns]
        .sum(
            axis=1,
            min_count=3
        )
    )

else:

    streamcat_data["forest_pct"] = pd.NA


# ============================================================
# 20. WETLAND COVER
# ============================================================

wetland_columns = [
    "pcthbwet2019ws",
    "pctwdwet2019ws",
]


if all(
    col in streamcat_data.columns
    for col in wetland_columns
):

    streamcat_data["wetland_pct"] = (
        streamcat_data[wetland_columns]
        .sum(
            axis=1,
            min_count=2
        )
    )

else:

    streamcat_data["wetland_pct"] = pd.NA


# ============================================================
# 21. SCREENING FLAGS
# ============================================================

streamcat_data[
    "developed_less_than_20pct"
] = (
    streamcat_data["developed_pct"]
    < DEVELOPED_THRESHOLD
)


streamcat_data[
    "anthropogenic_land_use_less_than_20pct"
] = (
    streamcat_data[
        "anthropogenic_land_use_pct"
    ]
    < ANTHROPOGENIC_THRESHOLD
)


# Both land-use criteria must be met
# for this particular screen.

streamcat_data[
    "passes_land_use_screen"
] = (
    streamcat_data[
        "developed_less_than_20pct"
    ]
    &
    streamcat_data[
        "anthropogenic_land_use_less_than_20pct"
    ]
)


streamcat_data[
    "land_use_screen_result"
] = (
    streamcat_data[
        "passes_land_use_screen"
    ]
    .map(
        {
            True:
                "MEETS <20% LAND-USE SCREEN",

            False:
                "DOES NOT MEET <20% LAND-USE SCREEN"
        }
    )
)


# ============================================================
# 22. RENAME OPTIONAL DISTURBANCE METRICS
# ============================================================

rename_map = {

    "rddensws":
        "road_density_km_per_sqkm",

    "rdcrsws":
        "road_stream_crossing_density_per_sqkm",

    "damdensws":
        "dam_density_per_sqkm",

    "nabd_densws":
        "anthropogenic_barrier_density_per_sqkm",

    "npdesdensws":
        "npdes_density_per_sqkm",

    "tridensws":
        "tri_site_density_per_sqkm",

    "minedensws":
        "mine_density_per_sqkm",

    "popden2010ws":
        "population_density_2010_per_sqkm",

    "canaldensws":
        "canal_ditch_pipeline_density_km_per_sqkm",

    "pctow2019ws":
        "open_water_pct",
}


streamcat_data = streamcat_data.rename(
    columns=rename_map
)


# ============================================================
# 23. KEEP USEFUL COLUMNS
# ============================================================

keep_columns = [

    "comid",

    # Developed
    "developed_open_space_pct",
    "developed_low_intensity_pct",
    "developed_medium_intensity_pct",
    "developed_high_intensity_pct",
    "developed_pct",

    # Agriculture
    "hay_pasture_pct",
    "cultivated_crop_pct",
    "agriculture_pct",

    # Combined screening metric
    "anthropogenic_land_use_pct",

    # Natural cover
    "forest_pct",
    "wetland_pct",

    # Screening results
    "developed_less_than_20pct",
    "anthropogenic_land_use_less_than_20pct",
    "passes_land_use_screen",
    "land_use_screen_result",

    # Optional disturbance indicators
    "road_density_km_per_sqkm",
    "road_stream_crossing_density_per_sqkm",
    "dam_density_per_sqkm",
    "anthropogenic_barrier_density_per_sqkm",
    "npdes_density_per_sqkm",
    "tri_site_density_per_sqkm",
    "mine_density_per_sqkm",
    "population_density_2010_per_sqkm",
    "canal_ditch_pipeline_density_km_per_sqkm",

    # Open water
    "open_water_pct",
]


# Keep only columns that actually exist
keep_columns = [
    col
    for col in keep_columns
    if col in streamcat_data.columns
]


# ============================================================
# 24. MERGE BACK TO STATIONS
# ============================================================

stations["NHDPlus_COMID"] = pd.to_numeric(
    stations["NHDPlus_COMID"],
    errors="coerce"
).astype("Int64")


results = stations.merge(
    streamcat_data[keep_columns],
    left_on="NHDPlus_COMID",
    right_on="comid",
    how="left"
)


results = results.drop(
    columns=["comid"],
    errors="ignore"
)


# ============================================================
# 25. ADD A SIMPLE STATUS DESCRIPTION
# ============================================================

def determine_screen_status(row):

    developed = row[
        "developed_pct"
    ]

    anthropogenic = row[
        "anthropogenic_land_use_pct"
    ]


    if (
        pd.isna(developed)
        or pd.isna(anthropogenic)
    ):

        return "No land-use data"


    developed_ok = (
        developed < DEVELOPED_THRESHOLD
    )

    anthropogenic_ok = (
        anthropogenic
        < ANTHROPOGENIC_THRESHOLD
    )


    if (
        developed_ok
        and anthropogenic_ok
    ):

        return (
            "Meets both <20% developed "
            "and <20% anthropogenic land-use screens"
        )


    if not developed_ok:

        return (
            "Fails developed-land-use screen"
        )


    if not anthropogenic_ok:

        return (
            "Fails anthropogenic "
            "land-use screen"
        )


    return "Check"


results[
    "Land_Use_Screening_Interpretation"
] = results.apply(
    determine_screen_status,
    axis=1
)


# ============================================================
# 26. SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 27. PRINT MAIN RESULTS
# ============================================================

print("\n" + "=" * 120)
print("LAND-USE SCREENING RESULTS")
print("=" * 120)


summary_columns = [

    "MonitoringLocationIdentifier",
    "MonitoringLocationName",
    "NHDPlus_COMID",

    "developed_pct",

    "hay_pasture_pct",
    "cultivated_crop_pct",

    "agriculture_pct",

    "anthropogenic_land_use_pct",

    "forest_pct",
    "wetland_pct",

    "developed_less_than_20pct",
    "anthropogenic_land_use_less_than_20pct",
    "passes_land_use_screen",

    "Land_Use_Screening_Interpretation"
]


summary_columns = [
    col
    for col in summary_columns
    if col in results.columns
]


print(
    results[
        summary_columns
    ].to_string(index=False)
)


# ============================================================
# 28. PRINT DISTURBANCE INDICATORS
# ============================================================

disturbance_columns = [

    "MonitoringLocationIdentifier",
    "MonitoringLocationName",

    "road_density_km_per_sqkm",
    "road_stream_crossing_density_per_sqkm",

    "dam_density_per_sqkm",
    "anthropogenic_barrier_density_per_sqkm",

    "npdes_density_per_sqkm",
    "tri_site_density_per_sqkm",

    "mine_density_per_sqkm",

    "population_density_2010_per_sqkm",

    "canal_ditch_pipeline_density_km_per_sqkm"
]


disturbance_columns = [
    col
    for col in disturbance_columns
    if col in results.columns
]


print("\n" + "=" * 120)
print("ADDITIONAL DISTURBANCE INDICATORS")
print("=" * 120)

print(
    results[
        disturbance_columns
    ].to_string(index=False)
)


# ============================================================
# 29. CONTENTNEA-ONLY TABLE
# ============================================================

contentnea_results = results[
    results["Screening_Group"]
    == "Contentnea longitudinal reference"
].copy()


contentnea_results = contentnea_results.sort_values(
    by="anthropogenic_land_use_pct",
    na_position="last"
)


print("\n" + "=" * 120)
print("CONTENTNEA CREEK LONGITUDINAL SCREEN")
print("=" * 120)


contentnea_columns = [

    "MonitoringLocationIdentifier",
    "MonitoringLocationName",

    "NHDPlus_COMID",

    "developed_pct",

    "hay_pasture_pct",
    "cultivated_crop_pct",

    "agriculture_pct",
    "anthropogenic_land_use_pct",

    "forest_pct",
    "wetland_pct",

    "road_density_km_per_sqkm",
    "road_stream_crossing_density_per_sqkm",

    "dam_density_per_sqkm",
    "anthropogenic_barrier_density_per_sqkm",

    "npdes_density_per_sqkm",

    "canal_ditch_pipeline_density_km_per_sqkm",

    "passes_land_use_screen",
    "Land_Use_Screening_Interpretation"
]


contentnea_columns = [
    col
    for col in contentnea_columns
    if col in contentnea_results.columns
]


print(
    contentnea_results[
        contentnea_columns
    ].to_string(index=False)
)


# ============================================================
# 30. OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 120)
print("OVERALL SUMMARY")
print("=" * 120)


n_total = len(results)

n_landuse = (
    results["anthropogenic_land_use_pct"]
    .notna()
    .sum()
)

n_developed_pass = (
    results["developed_pct"]
    < DEVELOPED_THRESHOLD
).sum()

n_anthropogenic_pass = (
    results["anthropogenic_land_use_pct"]
    < ANTHROPOGENIC_THRESHOLD
).sum()

n_both_pass = (
    results["passes_land_use_screen"]
).sum()


print(
    f"Total stations evaluated: "
    f"{n_total}"
)

print(
    f"Stations with land-use data: "
    f"{n_landuse}/{n_total}"
)

print(
    f"Stations with < "
    f"{DEVELOPED_THRESHOLD}% developed land: "
    f"{n_developed_pass}/{n_landuse}"
)

print(
    f"Stations with < "
    f"{ANTHROPOGENIC_THRESHOLD}% anthropogenic "
    f"land use: "
    f"{n_anthropogenic_pass}/{n_landuse}"
)

print(
    f"Stations meeting BOTH land-use screens: "
    f"{n_both_pass}/{n_landuse}"
)


print("\n" + "=" * 120)
print(
    f"Results saved to:\n{OUTPUT_FILE}"
)
print("=" * 120)