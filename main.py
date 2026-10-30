import math
import tempfile
from datetime import datetime, timezone, timedelta

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pyproj import CRS, Transformer

app = FastAPI(title="Territoria GOES-19 Processor", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

BUCKET = "noaa-goes19"
FIRE_PRODUCT = "ABI-L2-FDCF"
GLM_PRODUCT = "GLM-L2-LCFA"
DEFAULT_SCENES = 6
MAX_SCENES = 12

# Caja geografica operativa para Argentina + margen minimo fronterizo.
ARG_LAT_MIN = -55.2
ARG_LAT_MAX = -21.5
ARG_LON_MIN = -73.8
ARG_LON_MAX = -53.0

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED, retries={"max_attempts": 3}),
)


def haversine(a, b, c, d):
    radius_earth = 6371.0
    p1 = math.radians(a)
    p2 = math.radians(c)
    dp = math.radians(c - a)
    dl = math.radians(d - b)
    q = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius_earth * 2 * math.atan2(math.sqrt(q), math.sqrt(1 - q))


def bearing(a, b, c, d):
    p1 = math.radians(a)
    p2 = math.radians(c)
    dl = math.radians(d - b)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    degrees = (math.degrees(math.atan2(y, x)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SO", "O", "NO"][round(degrees / 45) % 8]


def in_argentina_box(lat, lon):
    return ARG_LAT_MIN <= lat <= ARG_LAT_MAX and ARG_LON_MIN <= lon <= ARG_LON_MAX


def product_keys(product, limit, lookback_hours=8):
    limit = max(1, int(limit))
    now = datetime.now(timezone.utc)
    found = []
    for back in range(lookback_hours):
        moment = now - timedelta(hours=back)
        prefix = f"{product}/{moment.year}/{moment.timetuple().tm_yday:03d}/{moment.hour:02d}/"
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        found.extend(item["Key"] for item in response.get("Contents", []) if item["Key"].endswith(".nc"))
        if len(found) >= limit:
            break
    keys = sorted(set(found))[-limit:]
    if not keys:
        raise RuntimeError(f"No se encontro archivo reciente para {product}")
    return keys


def decoded_value(var, y, x):
    raw = var.values[y, x]
    fill = var.attrs.get("_FillValue")
    if fill is not None and raw == fill:
        return None
    value = float(raw) * float(var.attrs.get("scale_factor", 1.0)) + float(var.attrs.get("add_offset", 0.0))
    return value if math.isfinite(value) else None


def scalar(ds, names, y, x):
    for name in names:
        if name in ds:
            try:
                return decoded_value(ds[name], y, x)
            except Exception:
                pass
    return None


def extract_fire(key, slat, slon, radius):
    with tempfile.NamedTemporaryFile(suffix=".nc") as temp_file:
        s3.download_fileobj(BUCKET, key, temp_file)
        temp_file.flush()
        ds = xr.open_dataset(temp_file.name, engine="h5netcdf", mask_and_scale=False)
        try:
            mask = np.asarray(ds["Mask"].values)
            codes = np.array([10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35], dtype=mask.dtype)
            ys, xs = np.where(np.isin(mask, codes))
            candidates = int(len(ys))
            projection = ds["goes_imager_projection"]
            height = float(projection.perspective_point_height)
            crs = CRS.from_proj4(
                f"+proj=geos +h={height} +lon_0={float(projection.longitude_of_projection_origin)} "
                f"+sweep={str(projection.sweep_angle_axis)} +ellps=GRS80"
            )
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            x_var, y_var = ds["x"], ds["y"]
            xv = np.asarray(x_var.values, dtype="float64") * float(x_var.attrs.get("scale_factor", 1.0)) + float(x_var.attrs.get("add_offset", 0.0))
            yv = np.asarray(y_var.values, dtype="float64") * float(y_var.attrs.get("scale_factor", 1.0)) + float(y_var.attrs.get("add_offset", 0.0))
            lons, lats = transformer.transform(xv[xs] * height, yv[ys] * height)
            labels = {10:"processed",11:"saturated",12:"cloudy",13:"high",14:"medium",15:"low",30:"processed-temporal",31:"saturated-temporal",32:"cloudy-temporal",33:"high-temporal",34:"medium-temporal",35:"low-temporal"}
            scene_time = str(ds.attrs.get("time_coverage_start") or ds.attrs.get("date_created") or datetime.now(timezone.utc).isoformat())
            output = []
            for y, x, latitude, longitude in zip(ys, xs, lats, lons):
                if not np.isfinite(latitude) or not np.isfinite(longitude):
                    continue
                latitude, longitude = float(latitude), float(longitude)
                km = haversine(slat, slon, latitude, longitude)
                if km > float(radius):
                    continue
                code = int(mask[y, x])
                output.append({
                    "id": f"{key}-{int(y)}-{int(x)}", "sourceKey": key,
                    "lat": round(latitude, 6), "lon": round(longitude, 6),
                    "distanceKm": round(km, 2), "bearing": bearing(slat, slon, latitude, longitude),
                    "classification": labels.get(code, str(code)), "maskCode": code,
                    "frpMw": scalar(ds, ["Power", "FRP"], y, x),
                    "temperatureK": scalar(ds, ["Temp", "Temperature"], y, x),
                    "sizeKm2": scalar(ds, ["Area"], y, x), "time": scene_time,
                })
            return sorted(output, key=lambda x: x["distanceKm"]), candidates
        finally:
            ds.close()


def glm_time(ds, value):
    epoch = ds.attrs.get("time_coverage_start") or datetime.now(timezone.utc).isoformat()
    try:
        base = datetime.fromisoformat(str(epoch).replace("Z", "+00:00"))
    except Exception:
        base = datetime.now(timezone.utc)
    try:
        seconds = float(value) * 1e-9
        return (base + timedelta(seconds=seconds)).astimezone(timezone.utc).isoformat()
    except Exception:
        return base.astimezone(timezone.utc).isoformat()


def extract_glm(key, slat, slon, radius, only_argentina=True):
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        s3.download_fileobj(BUCKET, key, tf); tf.flush()
        ds = xr.open_dataset(tf.name, engine="h5netcdf", mask_and_scale=True)
        try:
            latv = np.asarray(ds["flash_lat"].values, dtype=float)
            lonv = np.asarray(ds["flash_lon"].values, dtype=float)
            energy = np.asarray(ds["flash_energy"].values, dtype=float) if "flash_energy" in ds else np.full(latv.shape, np.nan)
            offset = np.asarray(ds["flash_time_offset_of_first_event"].values) if "flash_time_offset_of_first_event" in ds else np.zeros(latv.shape)
            out = []
            for i, (la, lo) in enumerate(zip(latv, lonv)):
                if not np.isfinite(la) or not np.isfinite(lo): continue
                la, lo = float(la), float(lo)
                if only_argentina and not in_argentina_box(la, lo): continue
                km = haversine(slat, slon, la, lo)
                if km > radius: continue
                ej = float(energy[i]) if i < len(energy) and np.isfinite(energy[i]) else None
                out.append({
                    "id": f"{key}-flash-{i}", "sourceKey": key,
                    "lat": round(la, 6), "lon": round(lo, 6),
                    "distanceKm": round(km, 2), "bearing": bearing(slat, slon, la, lo),
                    "energyJ": ej, "time": glm_time(ds, offset[i] if i < len(offset) else 0),
                })
            return out
        finally:
            ds.close()


@app.get("/")
def root():
    return {"ok": True, "service": "Territoria GOES-19 Processor", "version": "6.0.0", "products": [FIRE_PRODUCT, GLM_PRODUCT]}


@app.get("/fires")
def fires(lat: float=Query(...,ge=-90,le=90), lon: float=Query(...,ge=-180,le=180), radius: float=Query(5000,ge=5,le=20000), scenes: int=Query(DEFAULT_SCENES,ge=1,le=MAX_SCENES)):
    try:
        keys=product_keys(FIRE_PRODUCT, scenes); results=[]; candidates=0; errors=[]
        for key in keys:
            try:
                data,n=extract_fire(key,lat,lon,radius);results.extend(data);candidates+=n
            except Exception as e: errors.append({"sourceKey":key,"error":str(e)})
        return {"ok":True,"satellite":"GOES-19","product":FIRE_PRODUCT,"sourceKeys":keys,"detections":sorted(results,key=lambda x:(x["distanceKm"],x["time"])),"total":len(results),"sceneFirePixels":candidates,"deduplicated":False,"processedScenes":len(keys)-len(errors),"requestedScenes":scenes,"radiusKm":radius,"errors":errors[:5]}
    except Exception as e:
        return {"ok":False,"error":str(e),"detections":[],"total":0}


@app.get("/lightning")
def lightning(lat: float=Query(...,ge=-90,le=90), lon: float=Query(...,ge=-180,le=180), radius: float=Query(5000,ge=5,le=20000), minutes: int=Query(15,ge=1,le=30), argentina: bool=True, max_flashes: int=Query(1500,ge=50,le=5000)):
    try:
        # Cada archivo LCFA cubre 20 s. Pedimos un pequeño margen extra y luego filtramos por hora real.
        wanted=min(120, max(6, math.ceil(minutes*60/20)+6))
        keys=product_keys(GLM_PRODUCT,wanted,lookback_hours=2)
        cutoff=datetime.now(timezone.utc)-timedelta(minutes=minutes)
        flashes=[];errors=[]
        for key in keys:
            try: flashes.extend(extract_glm(key,lat,lon,radius,argentina))
            except Exception as e: errors.append({"sourceKey":key,"error":str(e)})
        fresh=[]
        for f in flashes:
            try:
                t=datetime.fromisoformat(str(f["time"]).replace("Z","+00:00"))
                if t>=cutoff: fresh.append(f)
            except Exception: pass
        fresh.sort(key=lambda x:x["time"],reverse=True)
        fresh=fresh[:max_flashes]
        return {"ok":True,"satellite":"GOES-19","product":GLM_PRODUCT,"scope":"Argentina" if argentina else "radius","minutes":minutes,"flashes":fresh,"detections":fresh,"total":len(fresh),"returned":len(fresh),"processedFiles":len(keys)-len(errors),"errors":errors[:5]}
    except Exception as e:
        return {"ok":False,"error":str(e),"product":GLM_PRODUCT,"flashes":[],"detections":[],"total":0}
