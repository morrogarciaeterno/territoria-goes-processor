import io
import math
from datetime import datetime, timezone

import requests
from fastapi import Query
from PIL import Image

ZONES = [
    {"id": "gran_mendoza", "name": "NORTE / GRAN MENDOZA", "lat": -32.89, "lon": -68.84},
    {"id": "valle_uco", "name": "CENTRO / VALLE DE UCO", "lat": -33.58, "lon": -69.02},
    {"id": "sur", "name": "SUR / SAN RAFAEL - ALVEAR", "lat": -34.62, "lon": -68.33},
    {"id": "malargue", "name": "MALARGÜE", "lat": -35.47, "lon": -69.58},
]

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
RADAR_API = "https://radar-mendoza.elexpreso17.workers.dev/radar-mendoza?count=15"
RADAR_W = 500
RADAR_H = 700
RADAR_BOUNDS = {
    "south": -36.207795391439184,
    "west": -70.70489586150933,
    "north": -29.89716829540947,
    "east": -65.29510413849067,
}

DCC_DBZ_PALETTE = [
    (70, (0, 0, 0)), (65, (253, 52, 28)), (60, (254, 95, 5)),
    (57, (254, 154, 88)), (54, (255, 253, 1)), (51, (250, 196, 49)),
    (48, (210, 136, 59)), (45, (192, 100, 135)), (42, (200, 15, 134)),
    (39, (110, 13, 198)), (36, (28, 71, 232)), (33, (8, 127, 219)),
    (30, (0, 112, 0)), (20, (0, 90, 0)), (10, (72, 61, 139)),
]

RANK = {"none": 0, "blue": 1, "green": 2, "echo": 2, "yellow": 3, "orange": 4, "red": 5, "magenta": 6}


def _num(v, default=0.0):
    try:
        n = float(v)
        return n if math.isfinite(n) else default
    except Exception:
        return default


def _max(values, start, length):
    items = [_num(x, float("nan")) for x in (values or [])[start:start + length]]
    items = [x for x in items if math.isfinite(x)]
    return max(items) if items else 0.0


def _min(values, start, length):
    items = [_num(x, float("nan")) for x in (values or [])[start:start + length]]
    items = [x for x in items if math.isfinite(x)]
    return min(items) if items else None


def _idx(times, current):
    if not times:
        return 0
    try:
        target = datetime.fromisoformat(current).timestamp()
        return min(range(len(times)), key=lambda i: abs(datetime.fromisoformat(times[i]).timestamp() - target))
    except Exception:
        return 0


def _shear(hourly, index, length):
    s10 = (hourly.get("wind_speed_10m") or [])[index:index + length]
    d10 = (hourly.get("wind_direction_10m") or [])[index:index + length]
    s5 = (hourly.get("wind_speed_500hPa") or [])[index:index + length]
    d5 = (hourly.get("wind_direction_500hPa") or [])[index:index + length]
    output = []
    for a, da, b, db in zip(s10, d10, s5, d5):
        a, b, da, db = map(_num, (a, b, da, db))
        r1, r2 = math.radians(da), math.radians(db)
        u1, v1 = -a * math.sin(r1), -a * math.cos(r1)
        u2, v2 = -b * math.sin(r2), -b * math.cos(r2)
        output.append(math.hypot(u2 - u1, v2 - v1))
    return max(output) if output else 0.0


def _hail(cape, li, shear, freezing):
    if li is None:
        return 0, "SIN DATOS"
    if cape < 100 or li >= 2:
        return 2, "MUY BAJO"
    if cape < 300 or li >= 0:
        return 8, "BAJO"
    score = 35 if cape >= 2500 else 28 if cape >= 1500 else 20 if cape >= 800 else 10
    score += 25 if li <= -6 else 20 if li <= -4 else 14 if li <= -2 else 6 if li < 0 else 0
    score += 25 if shear >= 75 else 20 if shear >= 55 else 14 if shear >= 35 else 7 if shear >= 20 else 0
    if freezing is not None:
        score += 15 if 1800 <= freezing <= 3200 else 9 if 3200 < freezing <= 3800 else 8 if 1000 <= freezing < 1800 else 0
    percentage = max(0, min(100, round(score)))
    level = "MUY ALTO" if percentage >= 70 else "ALTO" if percentage >= 50 else "MODERADO" if percentage >= 30 else "BAJO" if percentage >= 10 else "MUY BAJO"
    return percentage, level


def _level(rank):
    return ["NORMAL", "VIGILANCIA", "ALERTA", "ALERTA_IMPORTANTE"][max(0, min(3, rank))]


def _weather(zone):
    current = "temperature_2m,relative_humidity_2m,dew_point_2m,cloud_cover,wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,weather_code,is_day,precipitation,rain,showers"
    hourly = "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,precipitation_probability,weather_code,cape,lifted_index,freezing_level_height,wind_speed_10m,wind_direction_10m,wind_gusts_10m,wind_speed_500hPa,wind_direction_500hPa"
    daily = "temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset"
    response = requests.get(
        OPEN_METEO,
        params={
            "latitude": zone["lat"], "longitude": zone["lon"], "current": current,
            "hourly": hourly, "daily": daily, "forecast_days": 7,
            "timezone": "America/Argentina/Mendoza", "wind_speed_unit": "kmh",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _radar_pixel(lat, lon):
    b = RADAR_BOUNDS
    x = round((lon - b["west"]) / (b["east"] - b["west"]) * (RADAR_W - 1))
    y = round((b["north"] - lat) / (b["north"] - b["south"]) * (RADAR_H - 1))
    return max(0, min(RADAR_W - 1, x)), max(0, min(RADAR_H - 1, y))


def _nearest_dbz(r, g, b, a):
    if a < 20:
        return None
    best_dbz, best_distance = None, float("inf")
    for dbz, rgb in DCC_DBZ_PALETTE:
        distance = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
        if distance < best_distance:
            best_dbz, best_distance = dbz, distance
    return best_dbz if best_distance <= 2500 else None


def _radar_class(r, g, b, a):
    if a < 20:
        return "none"
    maximum, minimum = max(r, g, b), min(r, g, b)
    if maximum - minimum < 45 or maximum < 70:
        return "none"
    if r > 140 and b > 100 and g < 130:
        return "magenta"
    if r > 155 and g < 110 and b < 130:
        return "red"
    if r > 180 and 55 < g < 190 and b < 115:
        return "orange"
    if r > 145 and g > 115 and b < 130:
        return "yellow"
    if g > r * 1.12 and g > b * 1.03:
        return "green"
    if b > r * 1.12 and b > g * 1.03:
        return "blue"
    return "echo"


def _sample_radar(image, pixel):
    px = image.load()
    strongest, covered, warm, max_dbz = "none", 0, 0, None
    cx, cy = pixel
    for y in range(cy - 4, cy + 5):
        for x in range(cx - 4, cx + 5):
            if not (0 <= x < RADAR_W and 0 <= y < RADAR_H):
                continue
            r, g, b, a = px[x, y]
            radar_class = _radar_class(r, g, b, a)
            dbz = _nearest_dbz(r, g, b, a)
            if radar_class != "none":
                covered += 1
            if RANK[radar_class] >= 3:
                warm += 1
            if RANK[radar_class] > RANK[strongest]:
                strongest = radar_class
            if dbz is not None and (max_dbz is None or dbz > max_dbz):
                max_dbz = dbz
    return {
        "strongest": strongest,
        "coverage": round(covered / 81 * 100),
        "warm": round(warm / 81 * 100),
        "maxDbz": max_dbz,
    }


def _radar_label(dbz):
    if dbz is None:
        return "SIN ECO CALIBRADO"
    if dbz >= 65:
        return "REFLECTIVIDAD EXTREMA"
    if dbz >= 57:
        return "ECO MUY INTENSO"
    if dbz >= 48:
        return "ECO INTENSO"
    if dbz >= 39:
        return "PRECIPITACIÓN MODERADA/FUERTE"
    if dbz >= 30:
        return "PRECIPITACIÓN DÉBIL/MODERADA"
    return "ECO DÉBIL"


def _load_radar_for_zones(zones):
    empty = {
        z["id"]: {
            "available": False, "frames": 0, "maxDbz": None, "trend": "SIN DATOS",
            "persistence": 0, "coverageFirst": 0, "coverageLast": 0,
            "coverageChange": 0, "coreDetected": False, "label": "RADAR NO DISPONIBLE",
        }
        for z in zones
    }
    try:
        session = requests.Session()
        metadata_response = session.get(RADAR_API, timeout=25, headers={"Accept": "application/json", "User-Agent": "Territoria-Alerts/1.0"})
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        frames = (metadata.get("frames") or [])[-15:]
        if not frames:
            raise RuntimeError("Radar DCC sin frames")

        history = {z["id"]: [] for z in zones}
        for frame in frames:
            proxy_url = frame.get("proxyUrl")
            if not proxy_url:
                continue
            image_response = session.get(proxy_url, timeout=25, headers={"User-Agent": "Territoria-Alerts/1.0"})
            image_response.raise_for_status()
            image = Image.open(io.BytesIO(image_response.content)).convert("RGBA").resize((RADAR_W, RADAR_H), Image.Resampling.BILINEAR)
            for zone in zones:
                sample = _sample_radar(image, _radar_pixel(zone["lat"], zone["lon"]))
                sample["frame"] = frame.get("name")
                sample["date"] = frame.get("date")
                history[zone["id"]].append(sample)

        output = {}
        for zone in zones:
            samples = history[zone["id"]]
            if not samples:
                output[zone["id"]] = empty[zone["id"]]
                continue
            first, last = samples[0]["coverage"], samples[-1]["coverage"]
            delta = last - first
            trend = "ESTABLE" if abs(delta) < 10 else "AUMENTANDO" if delta > 0 else "DISMINUYENDO"
            valid_dbz = [s["maxDbz"] for s in samples if s["maxDbz"] is not None]
            max_dbz = max(valid_dbz) if valid_dbz else None
            persistence = sum(1 for s in samples if s["coverage"] > 5)
            output[zone["id"]] = {
                "available": True,
                "frames": len(samples),
                "latestFrame": samples[-1].get("frame"),
                "maxDbz": max_dbz,
                "trend": trend,
                "persistence": persistence,
                "coverageFirst": first,
                "coverageLast": last,
                "coverageChange": delta,
                "coreDetected": max_dbz is not None and max_dbz >= 39 and persistence >= 2,
                "label": _radar_label(max_dbz),
            }
        return output, None
    except Exception as error:
        return empty, str(error)


def _zone_state(zone, radar):
    data = _weather(zone)
    current, hourly = data.get("current") or {}, data.get("hourly") or {}
    index = _idx(hourly.get("time"), current.get("time"))
    cape = _max(hourly.get("cape"), index, 12)
    li = _min(hourly.get("lifted_index"), index, 12)
    freezing = _min(hourly.get("freezing_level_height"), index, 12)
    shear = _shear(hourly, index, 12)
    rain = _max(hourly.get("precipitation_probability"), index, 12)
    gust = _max(hourly.get("wind_gusts_10m"), index, 12) or _num(current.get("wind_gusts_10m"))
    cloud = _num(current.get("cloud_cover"))
    humidity = _num(current.get("relative_humidity_2m"))
    dew = _num(current.get("dew_point_2m"))
    temperature = _num(current.get("temperature_2m"))
    spread = temperature - dew

    storm = min(100, round(
        (30 if cape >= 2000 else 22 if cape >= 1000 else 14 if cape >= 500 else 5 if cape >= 100 else 0)
        + (25 if li is not None and li <= -6 else 19 if li is not None and li <= -4 else 12 if li is not None and li <= -2 else 5 if li is not None and li < 0 else 0)
        + (15 if shear >= 55 else 10 if shear >= 35 else 5 if shear >= 20 else 0)
        + (12 if cloud >= 80 else 7 if cloud >= 60 else 0)
    ))
    hail, hail_label = _hail(cape, li, shear, freezing)

    future = [_num(x, float("nan")) for x in (hourly.get("temperature_2m") or [])[index:index + 72]]
    future = [x for x in future if math.isfinite(x)]
    frost = min(future) if future else None
    frost_state = "HELADA PREVISTA" if frost is not None and frost < 0 else "RIESGO ALTO" if frost is not None and frost <= 1 else "VIGILANCIA" if frost is not None and frost <= 3 else "SIN RIESGO"
    wind_state = "MUY FUERTES" if gust >= 80 else "FUERTES" if gust >= 60 else "PRECAUCIÓN" if gust >= 40 else "NORMALES"
    direction = _num(current.get("wind_direction_10m"))
    west = 225 <= direction <= 315
    zscore = (35 if west else 0) + (30 if humidity <= 25 else 15 if humidity <= 35 else 0) + (25 if gust >= 60 else 12 if gust >= 40 else 0) + (10 if spread >= 15 else 0)
    zonda = "INDICIO ALTO" if zscore >= 70 else "A VIGILAR" if zscore >= 45 else "SIN INDICIOS"

    forecast_rain_rank = 2 if rain >= 80 else 1 if rain >= 60 else 0
    radar_rain_rank = 0
    radar_label = None
    if radar.get("available") and radar.get("maxDbz") is not None:
        dbz = radar["maxDbz"]
        radar_rain_rank = 3 if dbz >= 60 else 2 if dbz >= 48 else 1 if dbz >= 30 else 0
        if radar.get("trend") == "AUMENTANDO" and dbz >= 39:
            radar_rain_rank = min(3, radar_rain_rank + 1)
        radar_label = f"RADAR DCC · {radar['label']} · {dbz} dBZ · {radar['trend']}"

    rain_rank = max(forecast_rain_rank, radar_rain_rank)
    rain_label = radar_label if radar_rain_rank >= forecast_rain_rank and radar_label else "POSIBLES PRECIPITACIONES" if rain >= 60 else "NORMAL"

    events = {
        "rain": {"level": _level(rain_rank), "value": round(rain), "label": rain_label},
        "storm": {"level": _level(3 if storm >= 75 else 2 if storm >= 55 else 1 if storm >= 35 else 0), "value": storm, "label": "POSIBLE DESARROLLO DE TORMENTAS" if storm >= 35 else "NORMAL"},
        "hail": {"level": _level(3 if hail >= 70 else 2 if hail >= 50 else 1 if hail >= 30 else 0), "value": hail, "label": hail_label},
        "frost": {"level": _level(2 if frost_state in ("HELADA PREVISTA", "RIESGO ALTO") else 1 if frost_state == "VIGILANCIA" else 0), "value": None if frost is None else round(frost, 1), "label": frost_state},
        "wind": {"level": _level(3 if gust >= 80 else 2 if gust >= 60 else 1 if gust >= 40 else 0), "value": round(gust), "label": wind_state},
        "zonda": {"level": _level(2 if zscore >= 70 else 1 if zscore >= 45 else 0), "value": zscore, "label": zonda},
    }

    recommendations = {
        "rain": "Precaución en zonas con acumulación de agua y seguimiento del núcleo observado por Territoria.",
        "storm": "Evitar actividades al aire libre si aumenta la actividad eléctrica y seguir las actualizaciones.",
        "hail": "Resguardar vehículos, cultivos y elementos expuestos si el potencial continúa aumentando.",
        "frost": "Proteger cultivos, animales e instalaciones sensibles al descenso térmico.",
        "wind": "Asegurar objetos exteriores y evitar árboles, postes y estructuras expuestas.",
        "zonda": "Reducir actividades al aire libre, asegurar objetos y extremar precauciones por viento y sequedad.",
    }
    active = []
    for name, event in events.items():
        event["recommendation"] = recommendations[name]
        if event["level"] != "NORMAL":
            active.append(name)

    return {
        "id": zone["id"], "zone": zone["name"], "lat": zone["lat"], "lon": zone["lon"],
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "cape": round(cape), "liftedIndex": None if li is None else round(li, 1),
            "shearKmh": round(shear), "freezingLevelM": None if freezing is None else round(freezing),
            "humidity": round(humidity), "gustKmh": round(gust),
        },
        "radar": radar,
        "events": events,
        "activeEvents": active,
    }


def register_alert_routes(app):
    @app.get("/alerts")
    def alerts(zone: str | None = Query(None)):
        selected = ZONES
        if zone:
            query = zone.strip().lower()
            selected = [z for z in ZONES if z["id"] == query or query in z["name"].lower()]
            if not selected:
                return {"ok": False, "error": "Zona no encontrada", "zones": [z["id"] for z in ZONES]}

        radar_by_zone, radar_error = _load_radar_for_zones(selected)
        output, errors = [], []
        for item in selected:
            try:
                output.append(_zone_state(item, radar_by_zone[item["id"]]))
            except Exception as error:
                errors.append({"zone": item["name"], "error": str(error)})

        return {
            "ok": bool(output),
            "service": "Territoria Alerts",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "radarSource": "Radar DCC Mendoza · 15 frames",
            "radarError": radar_error,
            "zones": output,
            "errors": errors,
        }
