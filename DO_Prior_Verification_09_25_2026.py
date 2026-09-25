import pandas as pd
import geopandas as gpd
import requests
import time
from shapely.geometry import Point, shape


def check_stations_huc12(stations_csv="Final_33_Golden_Stations.csv",
                         wwtp_csv="NC_NPDES_Wastewater_Discharge_Permits.csv"):
    print("\n======================================================")
    print("  HUC-12 WATERSHED WWTP CHECK FOR GOLDEN STATIONS")
    print("======================================================")

    # 1. Load WWTPs and reproject from Web Mercator to standard Lat/Lon (Run once)
    print("Loading WWTP data...")
    df_wwtp = pd.read_csv(wwtp_csv)
    gdf_wwtp = gpd.GeoDataFrame(
        df_wwtp,
        geometry=[Point(xy) for xy in zip(df_wwtp['x'], df_wwtp['y'])],
        crs="EPSG:3857"
    ).to_crs("EPSG:4326")

    # 2. Load Stations
    df_stations = pd.read_csv(stations_csv)
    print(f"Loaded {len(df_stations)} stations to check.\n")

    results = []

    # 3. Loop through stations
    for _, row in df_stations.iterrows():
        sid = row['MonitoringLocationIdentifier']
        lon = float(row['LongitudeMeasure'])
        lat = float(row['LatitudeMeasure'])

        print(f"Checking Station: {sid} ({lat}, {lon})")

        # Fetch the highly stable HUC-12 Watershed boundary for this coordinate
        url = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query"
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "huc12,name",
            "returnGeometry": "true",
            "f": "geojson"
        }

        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            if not data.get("features"):
                print("  -> Could not locate a local watershed for this coordinate.\n")
                continue

            feature = data["features"][0]
            huc12_name = feature["properties"]["name"]
            basin_geom = shape(feature["geometry"])

            # Spatial filter: Check if any WWTPs physically sit in this local watershed
            wwtps_in_basin = gdf_wwtp[gdf_wwtp.geometry.within(basin_geom)]
            wwtp_count = len(wwtps_in_basin)

            if wwtp_count > 0:
                print(f"  -> [FAIL] Inside Watershed '{huc12_name}'. Found {wwtp_count} WWTP(s):")
                for _, wwtp_row in wwtps_in_basin.iterrows():
                    print(
                        f"     - {wwtp_row.get('FACILITY', 'Unknown Facility')} (Permit: {wwtp_row.get('PERMITNUMBER', 'N/A')})")
            else:
                print(f"  -> [PASS] Inside Watershed '{huc12_name}'. Clean. 0 WWTPs found.")

            results.append({
                'Station_ID': sid,
                'HUC12_Name': huc12_name,
                'Has_WWTP': wwtp_count > 0,
                'WWTP_Count': wwtp_count
            })

        except requests.exceptions.RequestException as e:
            print(f"  -> [ERROR] API Request failed: {e}")

        # Sleep to prevent USGS from blocking the connection for rapid requests
        time.sleep(0.5)
        print("-" * 60)

    # Summary
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        clean_stations = df_results[df_results['Has_WWTP'] == False]
        fails = len(df_results) - len(clean_stations)
        print(f"\nSUMMARY: {len(clean_stations)} stations kept. {fails} stations flagged with WWTPs in their HUC-12.")

    return df_results


# Execute
df_clean = check_stations_huc12()