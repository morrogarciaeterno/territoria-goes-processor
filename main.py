import os
import math
import tempfile
from datetime import datetime, timezone

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pyproj import CRS, Transformer

app = FastAPI(title="Territoria GOES-19 Processor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

BUCKET = "noaa-goes19"
PRODUCT = "ABI-L2-FDCF"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 3}))


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2-lon1)
    y = math.sin(dl)*math.cos(p2)
    x = math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    deg = (math.degrees(math.atan2(y,x))+360)%360
    names = ["N","NE","E","SE","S","SO","O","NO"]
    return names[round(deg/45)%8]


def latest_key():
    now = datetime.now(timezone.utc)
    for back in range(0, 4):
        t = now.timestamp() - back*3600
        d = datetime.fromtimestamp(t, timezone.utc)
        doy = d.timetuple().tm_yday
        prefix = f"{PRODUCT}/{d.year}/{doy:03d}/{d.hour:02d}/"
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        keys = [x["Key"] for x in response.get("Contents", []) if x["Key"].endswith(".nc")]
        if keys:
            return sorted(keys)[-1]
    raise RuntimeError("No se encontró un archivo GOES-19 reciente")


def scalar(ds, names, yi, xi):
    for name in names:
        if name in ds:
            try:
                return float(ds[name].values[yi, xi])
            except Exception:
                pass
    return None


def extract(key, station_lat, station_lon, radius_km):
    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        s3.download_fileobj(BUCKET, key, f)
        f.flush()
        ds = xr.open_dataset(f.name, engine="h5netcdf", mask_and_scale=True)
        mask_name = "Mask" if "Mask" in ds else "Power"
        mask = np.asarray(ds[mask_name].values)
        fire_codes = {10,11,12,13,14,15,30,31,32,33,34,35}
        ys, xs = np.where(np.isin(mask.astype("int32"), list(fire_codes)))

        proj = ds["goes_imager_projection"]
        h = float(proj.perspective_point_height)
        lon0 = float(proj.longitude_of_projection_origin)
        sweep = str(proj.sweep_angle_axis)
        crs_geos = CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} +ellps=GRS80")
        transformer = Transformer.from_crs(crs_geos, "EPSG:4326", always_xy=True)
        xv = np.asarray(ds["x"].values)[xs] * h
        yv = np.asarray(ds["y"].values)[ys] * h
        lons, lats = transformer.transform(xv, yv)

        labels = {10:"processed",11:"saturated",12:"cloudy",13:"high",14:"medium",15:"low",30:"processed-temporal",31:"saturated-temporal",32:"cloudy-temporal",33:"high-temporal",34:"medium-temporal",35:"low-temporal"}
        detections=[]
        for yi,xi,la,lo in zip(ys,xs,lats,lons):
            if not np.isfinite(la) or not np.isfinite(lo):
                continue
            km=haversine(station_lat,station_lon,float(la),float(lo))
            if km>radius_km:
                continue
            code=int(mask[yi,xi])
            detections.append({
                "id":f"{key}-{yi}-{xi}", "lat":round(float(la),6), "lon":round(float(lo),6),
                "distanceKm":round(km,2), "bearing":bearing(station_lat,station_lon,float(la),float(lo)),
                "classification":labels.get(code,str(code)), "maskCode":code,
                "frpMw":scalar(ds,["Power","FRP"],yi,xi),
                "temperatureK":scalar(ds,["Temp","Temperature"],yi,xi),
                "sizeKm2":scalar(ds,["Area"],yi,xi),
                "time":datetime.now(timezone.utc).isoformat()
            })
        ds.close()
        return sorted(detections,key=lambda x:x["distanceKm"])

@app.get("/")
def root():
    return {"ok": True, "service": "Territoria GOES-19 Processor", "product": PRODUCT}

@app.get("/fires")
def fires(lat: float=Query(...), lon: float=Query(...), radius: float=Query(50,ge=5,le=300), satellite: str="G19", product: str=PRODUCT):
    try:
        key=latest_key()
        detections=extract(key,lat,lon,radius)
        return {"ok":True,"satellite":"GOES-19","product":PRODUCT,"sourceKey":key,"detections":detections,"total":len(detections)}
    except Exception as e:
        return {"ok":False,"error":str(e),"detections":[]}
