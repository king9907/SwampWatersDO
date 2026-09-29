import pandas as pd
import numpy as np
import requests
import json
from io import StringIO
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, truncnorm
from scipy.optimize import curve_fit
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import geopandas as gpd
from shapely.geometry import Point, shape

today_str = datetime.now().strftime("%m_%d_%Y")


def get_station_coords(site_ids, batch_size=50):
    print(f"Fetching metadata for {len(site_ids)} DO stations in batches...")
    url = "https://www.waterqualitydata.us/data/Station/search"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_stations = []

    for i in range(0, len(site_ids), batch_size):
        batch = site_ids[i:i + batch_size]
        params = {"siteid": batch, "mimeType": "csv"}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.text.strip()) > 50:
                df_batch = pd.read_csv(StringIO(response.text), low_memory=False)
                usecols = ['MonitoringLocationIdentifier', 'MonitoringLocationName', 'LatitudeMeasure',
                           'LongitudeMeasure']
                cols_to_keep = [c for c in usecols if c in df_batch.columns]
                all_stations.append(df_batch[cols_to_keep])
        except Exception:
            pass
        time.sleep(0.5)

    if all_stations:
        return pd.concat(all_stations, ignore_index=True).drop_duplicates(subset=['MonitoringLocationIdentifier'])
    return pd.DataFrame()


def get_active_usgs_gauges(state='NC'):
    print(f"Fetching active USGS gauges for {state}...")
    url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={state}&siteType=ST&siteOutput=expanded"
    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text), comment='#', sep='\t').iloc[1:]
    df = df.rename(columns={'site_no': 'USGS_ID', 'dec_lat_va': 'Lat', 'dec_long_va': 'Lon'})
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
    return df.dropna(subset=['Lat', 'Lon'])


def get_sw_hydraulics_robust(df_usgs_master, swamp_vertices, target_da=234.81):
    print(f"\n--- Identifying Verified 'Sw' Peer Gauges ---")
    u_coords = np.radians(df_usgs_master[['Lat', 'Lon']].values)
    sw_coords = np.radians(swamp_vertices)
    tree = cKDTree(sw_coords)
    dist, _ = tree.query(u_coords, k=1)
    df_usgs_master['Is_Swamp'] = dist * 6371 <= 0.5

    da_min, da_max = target_da * 0.5, target_da * 2.0
    df_usgs_master['drain_area_va'] = pd.to_numeric(df_usgs_master['drain_area_va'], errors='coerce')

    peers = df_usgs_master[
        (df_usgs_master['Is_Swamp']) &
        (df_usgs_master['drain_area_va'] >= da_min) &
        (df_usgs_master['drain_area_va'] <= da_max)
        ].copy()

    print(f"Found {len(peers)} Sw-classified gauges within the peer DA range.")
    if peers.empty: return pd.DataFrame()

    a, b, c, d = 0.35, 0.45, 0.15, 0.25
    hydraulics = []

    for sid in peers['USGS_ID'].unique():
        url = (f"https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites={sid}"
               f"&statReportType=monthly&statType=mean&parameterCd=00060")
        try:
            r = requests.get(url, timeout=15)
            if "agency_cd" not in r.text: continue

            lines = [l for l in r.text.split('\n') if not l.startswith('#')]
            df = pd.read_csv(StringIO("\n".join([lines[0]] + lines[2:])), sep='\t')

            df.columns = [col.lower() for col in df.columns]
            month_col = [c for c in df.columns if 'month_nu' in c][0]
            val_col = [c for c in df.columns if 'mean_va' in c][0]

            df[month_col] = pd.to_numeric(df[month_col], errors='coerce')
            df[val_col] = pd.to_numeric(df[val_col], errors='coerce')

            q_cfs = df[df[month_col].isin([6, 7, 8, 9])][val_col].median()

            if np.isnan(q_cfs) or q_cfs <= 0: continue

            q_cms = q_cfs * 0.0283168
            h_m = a * (q_cms ** b)
            u_ms = c * (q_cms ** d)
            ka = 5.33 * (u_ms ** 0.67 / h_m ** 1.85)

            hydraulics.append({'USGS_ID': sid, 'Q_cfs': q_cfs, 'U_ms': u_ms, 'H_m': h_m, 'ka_d': ka})
            print(f"  {sid}: Q={q_cfs:.1f} cfs | ka={ka:.2f} d^-1")
        except Exception:
            continue

    return pd.DataFrame(hydraulics)


def filter_by_epa_streamcat(df_stations, max_impact_pct=20.0):
    print("\n======================================================")
    print("  PHASE 3.25: EPA STREAMCAT ANTHROPOGENIC FILTERING")
    print("======================================================")
    pristine_stations = []
    missing_data = []

    for _, row in df_stations.iterrows():
        sid = row['MonitoringLocationIdentifier']
        lat, lon = row['LatitudeMeasure'], row['LongitudeMeasure']

        waters_url = (f"https://ofmpub.epa.gov/waters10/PointIndexing.Service?"
                      f"pGeometry=POINT({lon}+{lat})&pGeometryMod=WGS84&"
                      f"pPointIndexingMethod=DISTANCE&pPointIndexingMaxDist=5&"
                      f"pResolution=3&optOutFormat=JSON")

        success = False
        for attempt in range(3):
            try:
                r_comid = requests.get(waters_url, timeout=10).json()

                # Check for the COMID in either 'ary_flowlines' or 'ary_locations'
                if 'output' in r_comid and 'ary_flowlines' in r_comid['output']:
                    comid = r_comid['output']['ary_flowlines'][0]['comid']
                elif 'output' in r_comid and 'ary_locations' in r_comid['output']:
                    comid = r_comid['output']['ary_locations'][0]['comid']
                else:
                    break  # API worked, but no stream snapped

                sc_url = f"https://v22.epa.gov/enviro/efservice/streamcat/comid/{comid}/json"
                r_sc = requests.get(sc_url, timeout=10).json()

                if r_sc:
                    sc_data = r_sc[0]
                    urban = float(sc_data.get('pcturbws2011', 0))
                    ag = float(sc_data.get('pctagws2011', 0))
                    total_impact = urban + ag

                    if total_impact <= max_impact_pct:
                        pristine_stations.append(sid)
                        print(f"  -> [PASS] {sid}: {total_impact:.1f}% impacted.")
                    else:
                        print(f"  -> [FAIL] {sid}: {total_impact:.1f}% impacted.")

                success = True
                break

            except Exception:
                time.sleep(1)

        if not success:
            print(f"  -> [ERROR] {sid}: API failed to verify land use. Dropping station.")
            missing_data.append(sid)

    df_pristine = df_stations[df_stations['MonitoringLocationIdentifier'].isin(pristine_stations)].copy()
    print(f"\nRetained {len(df_pristine)} stations. Dropped {len(missing_data)} due to API failures.")
    return df_pristine


import pandas as pd
import geopandas as gpd
import requests
import time
from shapely.geometry import Point, shape


def filter_by_huc12_wwtp(df_stations, wwtp_csv="NC_NPDES_Wastewater_Discharge_Permits.csv"):
    print("\n======================================================")
    print("  PHASE 3.3: HUC-12 WATERSHED WWTP FILTERING")
    print("======================================================")

    print("Loading WWTP data...")
    try:
        df_wwtp = pd.read_csv(wwtp_csv)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find '{wwtp_csv}'. Please ensure it is in the same directory.")
        return pd.DataFrame()

    # Create WWTP GeoDataFrame in Web Mercator (meters) for accurate distance calculations
    gdf_wwtp_meters = gpd.GeoDataFrame(
        df_wwtp,
        geometry=[Point(xy) for xy in zip(df_wwtp['x'], df_wwtp['y'])],
        crs="EPSG:3857"
    )

    # We also need a WGS84 version to intersect with the USGS GeoJSON polygons
    gdf_wwtp = gdf_wwtp_meters.to_crs("EPSG:4326")

    results = []

    for _, row in df_stations.iterrows():
        sid = row['MonitoringLocationIdentifier']
        lon = float(row['LongitudeMeasure'])
        lat = float(row['LatitudeMeasure'])

        print(f"Checking Station: {sid} ({lat}, {lon})")

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

        has_wwtp = False
        min_dist_km = None

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()

                if not data.get("features"):
                    print("  -> Could not locate watershed. Defaulting to keep.")
                    break

                feature = data["features"][0]
                huc12_name = feature["properties"]["name"]
                basin_geom = shape(feature["geometry"])

                # Intersect WWTPs with the basin polygon using WGS84
                wwtps_in_basin = gdf_wwtp[gdf_wwtp.geometry.within(basin_geom)]
                wwtp_count = len(wwtps_in_basin)
                has_wwtp = wwtp_count > 0

                if has_wwtp:
                    # Calculate distance to the nearest WWTP in meters using the projected CRS
                    station_pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

                    # Filter the projected WWTPs to just the ones in this basin
                    local_wwtps_meters = gdf_wwtp_meters.loc[wwtps_in_basin.index]

                    # Calculate distances and find the minimum
                    distances = local_wwtps_meters.geometry.distance(station_pt)
                    min_dist_km = distances.min() / 1000.0  # Convert meters to kilometers

                    print(f"  -> [FAIL] {wwtp_count} WWTP(s) in '{huc12_name}'. Nearest is {min_dist_km:.2f} km away.")
                else:
                    print(f"  -> [PASS] Clean. 0 WWTPs found in '{huc12_name}'.")

                break

            except Exception as e:
                if attempt == 2:
                    print(f"  -> [ERROR] API failed after 3 attempts: {e}. Defaulting to keep.")
                else:
                    time.sleep(2)

        results.append({
            'MonitoringLocationIdentifier': sid,
            'Has_WWTP': has_wwtp,
            'Nearest_WWTP_km': min_dist_km
        })
        time.sleep(0.5)

    df_results = pd.DataFrame(results)
    df_merged = df_stations.merge(df_results, on='MonitoringLocationIdentifier', how='left')
    return df_merged


def optimize_reference_clusters(df_golden_stations, df_hyd_priors, df_do_raw, df_usgs_master):
    print("\n======================================================")
    print("  PHASE 3.75: DEVELOPING HYDROGEOMORPHIC CLUSTERS")
    print("======================================================")
    df_hyd_locs = df_hyd_priors.merge(df_usgs_master[['USGS_ID', 'Lat', 'Lon']], on='USGS_ID', how='inner')

    if df_hyd_locs.empty: return df_golden_stations

    hyd_coords = np.radians(df_hyd_locs[['Lat', 'Lon']].values)
    do_coords = np.radians(df_golden_stations[['LatitudeMeasure', 'LongitudeMeasure']].values)

    tree = cKDTree(hyd_coords)
    distances, indices = tree.query(do_coords, k=1)

    station_features = []
    for i, row in enumerate(df_golden_stations.itertuples()):
        sid = row.MonitoringLocationIdentifier
        station_do = df_do_raw[df_do_raw['MonitoringLocationIdentifier'] == sid]
        do_std = pd.to_numeric(station_do['ResultMeasureValue'], errors='coerce').std()

        if not np.isnan(do_std):
            nearest_hyd = df_hyd_locs.iloc[indices[i]]
            station_features.append({
                'Station_ID': sid,
                'USGS_ID': nearest_hyd['USGS_ID'],
                'ka': nearest_hyd['ka_d'],
                'H_m': nearest_hyd['H_m'],
                'U_ms': nearest_hyd['U_ms'],
                'DO_Std': do_std
            })

    df_features = pd.DataFrame(station_features)
    if df_features.empty or len(df_features) < 4: return df_golden_stations

    features_to_scale = ['ka', 'H_m', 'U_ms']
    X_scaled = StandardScaler().fit_transform(df_features[features_to_scale])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_features['Cluster'] = kmeans.fit_predict(X_scaled)

    cluster_stats = df_features.groupby('Cluster').agg(
        Avg_Within_Station_DO_Variance=('DO_Std', 'mean')
    ).reset_index()

    optimal_cluster_idx = cluster_stats.loc[cluster_stats['Avg_Within_Station_DO_Variance'].idxmin(), 'Cluster']
    final_station_ids = df_features[df_features['Cluster'] == optimal_cluster_idx]['Station_ID'].values

    filtered_golden_stations = df_golden_stations[
        df_golden_stations['MonitoringLocationIdentifier'].isin(final_station_ids)
    ].copy()

    return filtered_golden_stations


def main():
    do_data_file = "regional_do_raw_data.csv"
    benthos_file = "ncdeq_benthos_master.csv"
    geojson_file = "Surface_Water_Classifications.geojson"

    print("======================================================")
    print("  PHASE 1 & 2: LOADING DATA & SWAMP VERIFICATION")
    print("======================================================")
    try:
        df_do_raw = pd.read_csv(do_data_file, low_memory=False)
        df_benthos = pd.read_csv(benthos_file)
        with open(geojson_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"Missing required file: {e}")
        return

    unique_do_ids = df_do_raw['MonitoringLocationIdentifier'].unique().tolist()
    df_stations = get_station_coords(unique_do_ids)
    df_stations['LatitudeMeasure'] = pd.to_numeric(df_stations['LatitudeMeasure'], errors='coerce')
    df_stations['LongitudeMeasure'] = pd.to_numeric(df_stations['LongitudeMeasure'], errors='coerce')
    df_stations = df_stations.dropna(subset=['LatitudeMeasure', 'LongitudeMeasure'])

    swamp_vertices = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        bims = str(props.get('BIMS_Class') or "")
        sup = str(props.get('Sup_Class') or "")
        pri = str(props.get('Pri_Class') or "")
        if 'Sw' not in f"{bims} | {sup} | {pri}": continue

        geom = feature.get('geometry')
        if not geom: continue
        coords = geom.get('coordinates', [])
        geom_type = geom.get('type')
        if geom_type == 'LineString':
            for pt in coords: swamp_vertices.append([pt[1], pt[0]])
        elif geom_type == 'MultiLineString':
            for line in coords:
                for pt in line: swamp_vertices.append([pt[1], pt[0]])

    swamp_coords = np.radians(swamp_vertices)
    station_coords = np.radians(df_stations[['LatitudeMeasure', 'LongitudeMeasure']].values)
    tree_swamp = cKDTree(swamp_coords)
    distances_sw, _ = tree_swamp.query(station_coords, k=1)

    df_stations['Distance_to_Swamp_km'] = distances_sw * 6371
    swamp_do_stations = df_stations[df_stations['Distance_to_Swamp_km'] <= 0.5].copy()

    print("\n======================================================")
    print("  PHASE 3: FILTERING FOR BIOLOGICAL INTEGRITY")
    print("======================================================")
    df_benthos = df_benthos.dropna(subset=['Latitude', 'Longitude'])
    benthos_coords = np.radians(df_benthos[['Latitude', 'Longitude']].values)
    swamp_station_coords = np.radians(swamp_do_stations[['LatitudeMeasure', 'LongitudeMeasure']].values)

    tree_bugs = cKDTree(benthos_coords)
    dist_bugs, idx_bugs = tree_bugs.query(swamp_station_coords, k=1)
    swamp_do_stations['Bug_Rating'] = df_benthos.iloc[idx_bugs]['Latest_Rating'].values
    swamp_do_stations['Distance_to_Bugs_km'] = dist_bugs * 6371

    final_golden_stations = swamp_do_stations[
        (swamp_do_stations['Distance_to_Bugs_km'] <= 3.0) &
        (swamp_do_stations['Bug_Rating'].isin(['Good-Fair', 'Moderate']))
        ].copy()

    # PHASE 3.25: StreamCat filter (ensures all downstream groups are <= 20% impacted)
    final_golden_stations = filter_by_epa_streamcat(final_golden_stations, max_impact_pct=20.0)
    if final_golden_stations.empty: return

    # -------------------------------------------------------------
    # NEW INTEGRATED PHASE 3.3: HUC-12 WWTP FILTERING & SPLIT
    # -------------------------------------------------------------
    df_stations_with_wwtp_flag = filter_by_huc12_wwtp(final_golden_stations)

    if df_stations_with_wwtp_flag.empty:
        print("CRITICAL: 0 stations returned from HUC-12 WWTP filtering.")
        return

    # Split the cohort into clean and impacted
    df_clean_stations = df_stations_with_wwtp_flag[df_stations_with_wwtp_flag['Has_WWTP'] == False].copy()
    df_impacted_stations = df_stations_with_wwtp_flag[df_stations_with_wwtp_flag['Has_WWTP'] == True].copy()

    export_csv = f"Final_{len(df_clean_stations)}_Pristine_Stations.csv"
    df_clean_stations.to_csv(export_csv, index=False)
    print(f"\n[EXPORTED] Verified clean station list saved to {export_csv}")

    # -------------------------------------------------------------
    # PHASE 3.4: COMPARATIVE DO STATISTICS FOR CLEAN VS IMPACTED
    # -------------------------------------------------------------
    print("\n======================================================")
    print("  DISSOLVED OXYGEN (DO) COMPARATIVE SUMMARY")
    print("======================================================")

    def print_cohort_stats(cohort_name, df_cohort, raw_do_df):
        ids = df_cohort['MonitoringLocationIdentifier'].tolist()
        df_filt = raw_do_df[raw_do_df['MonitoringLocationIdentifier'].isin(ids)].copy()
        df_filt['Value'] = pd.to_numeric(df_filt['ResultMeasureValue'], errors='coerce')
        df_filt = df_filt.dropna(subset=['Value'])

        stats = df_filt['Value'].agg(['count', 'mean', 'median', 'min', 'max', 'std'])

        print(f"\n--- {cohort_name} (N={len(ids)} stations) ---")
        print(f"Total Observations: {int(stats['count']):,}")
        print(f"  - Mean DO:    {stats['mean']:.2f} mg/L")
        print(f"  - Median DO:  {stats['median']:.2f} mg/L")
        print(f"  - Min DO:     {stats['min']:.2f} mg/L")
        print(f"  - Max DO:     {stats['max']:.2f} mg/L")
        print(f"  - Std Dev:    {stats['std']:.2f} mg/L")

    print_cohort_stats("Pristine Cohort (No Local WWTPs)", df_clean_stations, df_do_raw)
    print_cohort_stats("Impacted Cohort (Local WWTPs Present)", df_impacted_stations, df_do_raw)

    # REASSIGN final_golden_stations to strictly the clean ones so Phase 3.5+ uses the pristine group
    final_golden_stations = df_clean_stations

    # -------------------------------------------------------------
    print(f"\nProceeding to Phase 3.5 with {len(final_golden_stations)} verified natural stations.")

    print("\n======================================================")
    print("  PHASE 3.5 & 3.75: PEER HYDRAULICS & CLUSTERING")
    print("======================================================")
    df_usgs_site_master = get_active_usgs_gauges()
    df_hyd_priors = get_sw_hydraulics_robust(df_usgs_site_master, swamp_vertices, target_da=234.81)
    final_golden_stations = optimize_reference_clusters(final_golden_stations, df_hyd_priors, df_do_raw,
                                                        df_usgs_site_master)
    final_ids = final_golden_stations['MonitoringLocationIdentifier'].unique()

    print("\n======================================================")
    print("  PHASE 4: HYDRO-THERMAL REGRESSION DATA ASSEMBLY")
    print("======================================================")
    df_wqp = df_do_raw[df_do_raw['MonitoringLocationIdentifier'].isin(final_ids)].copy()

    do_mask = df_wqp['CharacteristicName'].str.contains('oxygen', case=False, na=False)
    df_hourly = df_wqp[do_mask].copy()
    df_hourly['Value'] = pd.to_numeric(df_hourly['ResultMeasureValue'], errors='coerce')
    dt_strings = df_hourly['ActivityStartDate'].astype(str) + ' ' + df_hourly['ActivityStartTime/Time'].astype(str)
    df_hourly['Parsed_DT'] = pd.to_datetime(dt_strings, errors='coerce')
    df_hourly['Hour'] = df_hourly['Parsed_DT'].dt.hour + (df_hourly['Parsed_DT'].dt.minute / 60.0)
    df_hourly = df_hourly.dropna(subset=['Value', 'Hour', 'Parsed_DT'])
    df_hourly = df_hourly[df_hourly['Hour'] != 0]

    df_wqp['Date'] = pd.to_datetime(df_wqp['ActivityStartDate'], errors='coerce').dt.normalize()
    df_wqp['Value'] = pd.to_numeric(df_wqp['ResultMeasureValue'], errors='coerce')
    df_do = df_wqp[do_mask].groupby(['MonitoringLocationIdentifier', 'Date'])['Value'].mean().reset_index()
    df_do = df_do.rename(columns={'Value': 'DO'})

    df_temp_csv = pd.read_csv("regional_temp_raw_data.csv", low_memory=False)
    df_temp_csv = df_temp_csv[df_temp_csv['MonitoringLocationIdentifier'].isin(final_ids)].copy()
    df_temp_csv['Date'] = pd.to_datetime(df_temp_csv['ActivityStartDate'], errors='coerce').dt.normalize()
    df_temp_csv['Value'] = pd.to_numeric(df_temp_csv['ResultMeasureValue'], errors='coerce')
    df_temp_raw = df_temp_csv.groupby(['MonitoringLocationIdentifier', 'Date'])['Value'].mean().reset_index()
    df_temp_raw = df_temp_raw.rename(columns={'Value': 'Temp'})

    df_paired = pd.merge(df_do, df_temp_raw, on=['MonitoringLocationIdentifier', 'Date'], how='inner')
    df_paired = df_paired[(df_paired['DO'] > 0) & (df_paired['Temp'] > 0)]

    peer_usgs_ids = df_hyd_priors['USGS_ID'].unique().tolist()
    flow_data = []
    current_year = datetime.now().year

    for sid in peer_usgs_ids:
        url = (f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={sid}"
               f"&parameterCd=00060&startDT=1980-01-01&endDT={current_year}-01-01")
        try:
            r = requests.get(url, timeout=15)
            if "agency_cd" not in r.text: continue
            lines = [l for l in r.text.split('\n') if not l.startswith('#')]
            if len(lines) < 3: continue

            df_q = pd.read_csv(StringIO("\n".join([lines[0]] + lines[2:])), sep='\t')
            val_cols = [c for c in df_q.columns if c.endswith('_00060_00003')]
            if not val_cols: continue

            df_q['Date'] = pd.to_datetime(df_q['datetime'], errors='coerce').dt.normalize()
            df_q['Q_cfs'] = pd.to_numeric(df_q[val_cols[0]], errors='coerce')
            df_q['USGS_ID'] = sid
            flow_data.append(df_q[['USGS_ID', 'Date', 'Q_cfs']].dropna())
        except Exception:
            continue

    df_flow_master = pd.concat(flow_data, ignore_index=True) if flow_data else pd.DataFrame()
    df_hyd_locs = df_hyd_priors.merge(df_usgs_site_master[['USGS_ID', 'Lat', 'Lon']], on='USGS_ID', how='inner')

    if not df_hyd_locs.empty:
        hyd_coords = np.radians(df_hyd_locs[['Lat', 'Lon']].values)
        do_coords = np.radians(final_golden_stations[['LatitudeMeasure', 'LongitudeMeasure']].values)
        tree_mapping = cKDTree(hyd_coords)
        _, indices_mapping = tree_mapping.query(do_coords, k=1)
        station_to_usgs = {row.MonitoringLocationIdentifier: df_hyd_locs.iloc[indices_mapping[i]]['USGS_ID'] for i, row
                           in enumerate(final_golden_stations.itertuples())}
        df_paired['USGS_ID'] = df_paired['MonitoringLocationIdentifier'].map(station_to_usgs)

    df_regression = pd.merge(df_paired, df_flow_master, on=['USGS_ID', 'Date'], how='inner')
    df_regression = df_regression[df_regression['Q_cfs'] > 0].copy()
    df_regression['Log_Q'] = np.log(df_regression['Q_cfs'])

    if df_regression.empty:
        print("CRITICAL: 0 paired observations. Cannot train regression.")
        return

    print("\n======================================================")
    print("  PHASE 5: PREDICTIVE BOUNDARY CONDITION MODELING")
    print("======================================================")
    X = df_regression[['Log_Q', 'Temp']]
    y = df_regression['DO']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    TARGET_Q_CFS = 4.0
    TARGET_TEMP_C = 28.5
    predicted_mean_do = model.predict(pd.DataFrame({'Log_Q': [np.log(TARGET_Q_CFS)], 'Temp': [TARGET_TEMP_C]}))[0]

    n_sims = 10000
    lower_bound = (0 - predicted_mean_do) / rmse
    upper_bound = (15 - predicted_mean_do) / rmse
    sim_means = truncnorm.rvs(lower_bound, upper_bound, loc=predicted_mean_do, scale=rmse, size=n_sims)

    print("\n--- Bootstrapping Empirical DO Amplitude via Mean-Centered Anomalies ---")

    summer_drought = df_hourly[(df_hourly['Parsed_DT'].dt.year.isin([2007, 2008])) &
                               (df_hourly['Parsed_DT'].dt.month.isin([4, 5, 6, 7, 8, 9, 10]))].copy()

    if summer_drought.empty:
        summer_drought = df_hourly[df_hourly['Parsed_DT'].dt.month.isin([4, 5, 6, 7, 8, 9, 10])].copy()

    summer_drought['o'] = summer_drought['Value']
    local_means = summer_drought.groupby('MonitoringLocationIdentifier')['o'].transform('mean')
    summer_drought['delta_o'] = summer_drought['o'] - local_means

    def diurnal_anomaly_model(t, amplitude, phase):
        return amplitude * np.sin((2 * np.pi / 24) * t + phase)

    bootstrapped_amplitudes = []
    max_possible_amp = (summer_drought['delta_o'].max() - summer_drought['delta_o'].min()) / 2.0
    amp_guess = min(1.5, max_possible_amp / 2.0)

    initial_guess = [amp_guess, 0.0]
    lower_bounds = [0.0, -np.pi]
    upper_bounds = [max_possible_amp, np.pi]

    print(f"Applying robust bounds: Amplitude mathematically capped at {max_possible_amp:.2f} mg/L")

    for _ in range(1000):
        sample = summer_drought.sample(frac=1.0, replace=True)
        try:
            popt, _ = curve_fit(diurnal_anomaly_model, sample['Hour'], sample['delta_o'],
                                p0=initial_guess, bounds=(lower_bounds, upper_bounds),
                                method='trf', maxfev=3000)

            calculated_amp = abs(popt[0])
            if calculated_amp < (max_possible_amp * 0.95):
                bootstrapped_amplitudes.append(calculated_amp)
        except RuntimeError:
            continue

    if bootstrapped_amplitudes:
        empirical_amp_mean = np.mean(bootstrapped_amplitudes)
        empirical_amp_std = np.std(bootstrapped_amplitudes)
    else:
        print("CRITICAL WARNING: Optimizer hit boundaries on all passes. Falling back to biological constraints.")
        empirical_amp_mean = 0.75
        empirical_amp_std = 0.25

    empirical_swing_mean = empirical_amp_mean * 2
    empirical_swing_std = empirical_amp_std * 2

    print(f"Bootstrapped \u0394DO Swing Mean:   {empirical_swing_mean:.2f} mg/L")
    print(f"Bootstrapped \u0394DO Swing StdDev: {empirical_swing_std:.2f} mg/L")

    sim_swings = np.maximum(np.random.normal(empirical_swing_mean, empirical_swing_std, n_sims), 0.1)
    sim_mins = np.maximum(sim_means - (sim_swings / 2), 0)
    sim_maxs = np.maximum(sim_means + (sim_swings / 2), 0)

    priors = pd.DataFrame({
        'Parameter': ['DO_Min', 'DO_Max', 'DO_Avg', 'Diurnal_Swing'],
        'Mean_mgL': [sim_mins.mean(), sim_maxs.mean(), sim_means.mean(), sim_swings.mean()],
        'StdDev_mgL': [sim_mins.std(), sim_maxs.std(), sim_means.std(), sim_swings.std()],
        'P5_Lower': [np.percentile(sim_mins, 5), np.percentile(sim_maxs, 5), np.percentile(sim_means, 5),
                     np.percentile(sim_swings, 5)],
        'P95_Upper': [np.percentile(sim_mins, 95), np.percentile(sim_maxs, 95), np.percentile(sim_means, 95),
                      np.percentile(sim_swings, 95)]
    })

    print("\n--- FINAL QUAL2K SUMMER PRIORS ---")
    print(priors.to_string(index=False))

    out_file_do = f"QUAL2K_Conditioned_Baseline_DO_Priors_{today_str}.csv"
    priors.to_csv(out_file_do, index=False)
    print(f"[EXPORTED] Conditioned DO priors saved to: {out_file_do}")


if __name__ == "__main__":
    main()