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

app = FastAPI(title="Territoria GOES-19 Processor", version="5.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

BUCKET = "noaa-goes19"
PRODUCT = "ABI-L2-FDCF"
DEFAULT_SCENES = 6
MAX_SCENES = 12

s3 = boto3.client(
    "s3",
    config=Config(
        signature_version=UNSIGNED,
        retries={"max_attempts": 3},
    ),
)


def haversine(a, b, c, d):
    radius_earth = 6371.0
    p1 = math.radians(a)
    p2 = math.radians(c)
    dp = math.radians(c - a)
    dl = math.radians(d - b)
    q = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return radius_earth * 2 * math.atan2(math.sqrt(q), math.sqrt(1 - q))


def bearing(a, b, c, d):
    p1 = math.radians(a)
    p2 = math.radians(c)
    dl = math.radians(d - b)
    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )
    degrees = (math.degrees(math.atan2(y, x)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SO", "O", "NO"][round(degrees / 45) % 8]


def latest_keys(limit=DEFAULT_SCENES):
    """Obtiene varias escenas GOES-19 recientes, de más antigua a más nueva."""
    limit = max(1, min(int(limit), MAX_SCENES))
    now = datetime.now(timezone.utc)
    found = []

    # Revisa hasta 8 horas hacia atrás para tolerar demoras o huecos de publicación.
    for back in range(8):
        moment = datetime.fromtimestamp(now.timestamp() - back * 3600, timezone.utc)
        prefix = (
            f"{PRODUCT}/{moment.year}/"
            f"{moment.timetuple().tm_yday:03d}/{moment.hour:02d}/"
        )
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        found.extend(
            item["Key"]
            for item in response.get("Contents", [])
            if item["Key"].endswith(".nc")
        )
        if len(found) >= limit:
            break

    keys = sorted(set(found))[-limit:]
    if not keys:
        raise RuntimeError("No se encontró archivo GOES-19 reciente")
    return keys


def decoded_value(var, y, x):
    raw = var.values[y, x]
    fill = var.attrs.get("_FillValue")
    if fill is not None and raw == fill:
        return None
    value = (
        float(raw) * float(var.attrs.get("scale_factor", 1.0))
        + float(var.attrs.get("add_offset", 0.0))
    )
    return value if math.isfinite(value) else None


def scalar(ds, names, y, x):
    for name in names:
        if name in ds:
            try:
                return decoded_value(ds[name], y, x)
            except Exception:
                pass
    return None


def extract(key, slat, slon, radius):
    """Extrae los píxeles FDCF válidos de una escena dentro del radio indicado."""
    with tempfile.NamedTemporaryFile(suffix=".nc") as temp_file:
        s3.download_fileobj(BUCKET, key, temp_file)
        temp_file.flush()

        ds = xr.open_dataset(
            temp_file.name,
            engine="h5netcdf",
            mask_and_scale=False,
        )

        try:
            mask = np.asarray(ds["Mask"].values)
            codes = np.array(
                [10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35],
                dtype=mask.dtype,
            )
            ys, xs = np.where(np.isin(mask, codes))
            scene_candidates = int(len(ys))

            projection = ds["goes_imager_projection"]
            height = float(projection.perspective_point_height)
            crs = CRS.from_proj4(
                f"+proj=geos +h={height} "
                f"+lon_0={float(projection.longitude_of_projection_origin)} "
                f"+sweep={str(projection.sweep_angle_axis)} +ellps=GRS80"
            )
            transformer = Transformer.from_crs(
                crs,
                "EPSG:4326",
                always_xy=True,
            )

            x_var = ds["x"]
            y_var = ds["y"]
            x_values = (
                np.asarray(x_var.values, dtype="float64")
                * float(x_var.attrs.get("scale_factor", 1.0))
                + float(x_var.attrs.get("add_offset", 0.0))
            )
            y_values = (
                np.asarray(y_var.values, dtype="float64")
                * float(y_var.attrs.get("scale_factor", 1.0))
                + float(y_var.attrs.get("add_offset", 0.0))
            )

            lons, lats = transformer.transform(
                x_values[xs] * height,
                y_values[ys] * height,
            )

            labels = {
                10: "processed",
                11: "saturated",
                12: "cloudy",
                13: "high",
                14: "medium",
                15: "low",
                30: "processed-temporal",
                31: "saturated-temporal",
                32: "cloudy-temporal",
                33: "high-temporal",
                34: "medium-temporal",
                35: "low-temporal",
            }

            scene_time = str(
                ds.attrs.get("time_coverage_start")
                or ds.attrs.get("date_created")
                or datetime.now(timezone.utc).isoformat()
            )

            output = []
            for y, x, latitude, longitude in zip(ys, xs, lats, lons):
                if not np.isfinite(latitude) or not np.isfinite(longitude):
                    continue

                latitude = float(latitude)
                longitude = float(longitude)
                km = haversine(slat, slon, latitude, longitude)

                if km > float(radius):
                    continue

                code = int(mask[y, x])
                output.append(
                    {
                        "id": f"{key}-{int(y)}-{int(x)}",
                        "sourceKey": key,
                        "lat": round(latitude, 6),
                        "lon": round(longitude, 6),
                        "distanceKm": round(km, 2),
                        "bearing": bearing(slat, slon, latitude, longitude),
                        "classification": labels.get(code, str(code)),
                        "maskCode": code,
                        "frpMw": scalar(ds, ["Power", "FRP"], y, x),
                        "temperatureK": scalar(ds, ["Temp", "Temperature"], y, x),
                        "sizeKm2": scalar(ds, ["Area"], y, x),
                        "time": scene_time,
                    }
                )

            return sorted(output, key=lambda item: item["distanceKm"]), scene_candidates
        finally:
            ds.close()



@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Territoria GOES-19 Processor",
        "version": "5.1.0",
        "product": PRODUCT,
    }


@app.get("/fires")
def fires(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius: float = Query(5000, ge=5, le=20000),
    scenes: int = Query(DEFAULT_SCENES, ge=1, le=MAX_SCENES),
    satellite: str = "G19",
    product: str = PRODUCT,
):
    try:
        keys = latest_keys(scenes)
        scene_results = []
        scene_candidates = 0
        errors = []

        for key in keys:
            try:
                detections, candidates = extract(key, lat, lon, radius)
                scene_results.append(detections)
                scene_candidates += candidates
            except Exception as error:
                errors.append({"sourceKey": key, "error": str(error)})

        if not scene_results:
            detail = errors[0]["error"] if errors else "No se pudieron procesar escenas"
            raise RuntimeError(detail)

        # Sin deduplicación: devuelve absolutamente todos los registros
        # de todas las escenas procesadas. Un mismo foco puede aparecer
        # varias veces si persistió en escenas consecutivas.
        detections = sorted(
            [
                detection
                for scene_detections in scene_results
                for detection in scene_detections
            ],
            key=lambda item: (
                item.get("distanceKm", 999999),
                item.get("time", ""),
            ),
        )

        return {
            "ok": True,
            "satellite": "GOES-19",
            "product": PRODUCT,
            "sourceKey": keys[-1],
            "sourceKeys": keys,
            "detections": detections,
            "total": len(detections),
            "returned": len(detections),
            "sceneFirePixels": scene_candidates,
            "deduplicated": False,
            "processedScenes": len(scene_results),
            "requestedScenes": scenes,
            "radiusKm": radius,
            "coverage": "full-disk",
            "errors": errors[:5],
        }
    except Exception as error:
        return {
            "ok": False,
            "error": str(error),
            "detections": [],
            "total": 0,
        }
