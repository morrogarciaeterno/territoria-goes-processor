import math
from datetime import datetime, timezone

import requests
from fastapi import Query

ZONES = [
    {"id":"gran_mendoza","name":"NORTE / GRAN MENDOZA","lat":-32.89,"lon":-68.84},
    {"id":"valle_uco","name":"CENTRO / VALLE DE UCO","lat":-33.58,"lon":-69.02},
    {"id":"sur","name":"SUR / SAN RAFAEL - ALVEAR","lat":-34.62,"lon":-68.33},
    {"id":"malargue","name":"MALARGÜE","lat":-35.47,"lon":-69.58},
]

OPEN_METEO='https://api.open-meteo.com/v1/forecast'


def _num(v, default=0.0):
    try:
        n=float(v)
        return n if math.isfinite(n) else default
    except Exception:
        return default


def _max(values, start, length):
    a=[_num(x, float('nan')) for x in (values or [])[start:start+length]]
    a=[x for x in a if math.isfinite(x)]
    return max(a) if a else 0.0


def _min(values, start, length):
    a=[_num(x, float('nan')) for x in (values or [])[start:start+length]]
    a=[x for x in a if math.isfinite(x)]
    return min(a) if a else None


def _idx(times, current):
    if not times:return 0
    try:return min(range(len(times)),key=lambda i:abs(datetime.fromisoformat(times[i]).timestamp()-datetime.fromisoformat(current).timestamp()))
    except Exception:return 0


def _shear(hourly, i, length):
    s10=(hourly.get('wind_speed_10m') or [])[i:i+length]
    d10=(hourly.get('wind_direction_10m') or [])[i:i+length]
    s5=(hourly.get('wind_speed_500hPa') or [])[i:i+length]
    d5=(hourly.get('wind_direction_500hPa') or [])[i:i+length]
    out=[]
    for a,da,b,db in zip(s10,d10,s5,d5):
        a,b,da,db=map(_num,(a,b,da,db));r1=math.radians(da);r2=math.radians(db)
        u1=-a*math.sin(r1);v1=-a*math.cos(r1);u2=-b*math.sin(r2);v2=-b*math.cos(r2)
        out.append(math.hypot(u2-u1,v2-v1))
    return max(out) if out else 0.0


def _hail(cape, li, shear, freezing):
    if li is None:return 0,'SIN DATOS'
    if cape<100 or li>=2:return 2,'MUY BAJO'
    if cape<300 or li>=0:return 8,'BAJO'
    s=35 if cape>=2500 else 28 if cape>=1500 else 20 if cape>=800 else 10
    s+=25 if li<=-6 else 20 if li<=-4 else 14 if li<=-2 else 6 if li<0 else 0
    s+=25 if shear>=75 else 20 if shear>=55 else 14 if shear>=35 else 7 if shear>=20 else 0
    if freezing is not None:s+=15 if 1800<=freezing<=3200 else 9 if 3200<freezing<=3800 else 8 if 1000<=freezing<1800 else 0
    p=max(0,min(100,round(s)))
    return p,'MUY ALTO' if p>=70 else 'ALTO' if p>=50 else 'MODERADO' if p>=30 else 'BAJO' if p>=10 else 'MUY BAJO'


def _level(rank):
    return ['NORMAL','VIGILANCIA','ALERTA','ALERTA_IMPORTANTE'][max(0,min(3,rank))]


def _weather(zone):
    current='temperature_2m,relative_humidity_2m,dew_point_2m,cloud_cover,wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,weather_code,is_day,precipitation,rain,showers'
    hourly='temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,precipitation_probability,weather_code,cape,lifted_index,freezing_level_height,wind_speed_10m,wind_direction_10m,wind_gusts_10m,wind_speed_500hPa,wind_direction_500hPa'
    daily='temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset'
    r=requests.get(OPEN_METEO,params={'latitude':zone['lat'],'longitude':zone['lon'],'current':current,'hourly':hourly,'daily':daily,'forecast_days':7,'timezone':'America/Argentina/Mendoza','wind_speed_unit':'kmh'},timeout=20)
    r.raise_for_status();return r.json()


def _zone_state(zone):
    a=_weather(zone);c=a.get('current') or {};h=a.get('hourly') or {};i=_idx(h.get('time'),c.get('time'))
    cape=_max(h.get('cape'),i,12);li=_min(h.get('lifted_index'),i,12);freezing=_min(h.get('freezing_level_height'),i,12);shear=_shear(h,i,12)
    rain=_max(h.get('precipitation_probability'),i,12);gust=_max(h.get('wind_gusts_10m'),i,12) or _num(c.get('wind_gusts_10m'))
    cloud=_num(c.get('cloud_cover'));humidity=_num(c.get('relative_humidity_2m'));dew=_num(c.get('dew_point_2m'));temp=_num(c.get('temperature_2m'));spread=temp-dew
    storm=min(100,round((30 if cape>=2000 else 22 if cape>=1000 else 14 if cape>=500 else 5 if cape>=100 else 0)+(25 if li is not None and li<=-6 else 19 if li is not None and li<=-4 else 12 if li is not None and li<=-2 else 5 if li is not None and li<0 else 0)+(15 if shear>=55 else 10 if shear>=35 else 5 if shear>=20 else 0)+(12 if cloud>=80 else 7 if cloud>=60 else 0)))
    hail,hail_label=_hail(cape,li,shear,freezing)
    future=[_num(x,float('nan')) for x in (h.get('temperature_2m') or [])[i:i+72]];future=[x for x in future if math.isfinite(x)];frost=min(future) if future else None
    frost_state='HELADA PREVISTA' if frost is not None and frost<0 else 'RIESGO ALTO' if frost is not None and frost<=1 else 'VIGILANCIA' if frost is not None and frost<=3 else 'SIN RIESGO'
    wind_state='MUY FUERTES' if gust>=80 else 'FUERTES' if gust>=60 else 'PRECAUCIÓN' if gust>=40 else 'NORMALES'
    direction=_num(c.get('wind_direction_10m'));west=225<=direction<=315;zscore=(35 if west else 0)+(30 if humidity<=25 else 15 if humidity<=35 else 0)+(25 if gust>=60 else 12 if gust>=40 else 0)+(10 if spread>=15 else 0)
    zonda='INDICIO ALTO' if zscore>=70 else 'A VIGILAR' if zscore>=45 else 'SIN INDICIOS'
    events={
      'rain':{'level':_level(2 if rain>=80 else 1 if rain>=60 else 0),'value':round(rain),'label':'POSIBLES PRECIPITACIONES' if rain>=60 else 'NORMAL'},
      'storm':{'level':_level(3 if storm>=75 else 2 if storm>=55 else 1 if storm>=35 else 0),'value':storm,'label':'POSIBLE DESARROLLO DE TORMENTAS' if storm>=35 else 'NORMAL'},
      'hail':{'level':_level(3 if hail>=70 else 2 if hail>=50 else 1 if hail>=30 else 0),'value':hail,'label':hail_label},
      'frost':{'level':_level(2 if frost_state=='HELADA PREVISTA' else 2 if frost_state=='RIESGO ALTO' else 1 if frost_state=='VIGILANCIA' else 0),'value':None if frost is None else round(frost,1),'label':frost_state},
      'wind':{'level':_level(3 if gust>=80 else 2 if gust>=60 else 1 if gust>=40 else 0),'value':round(gust),'label':wind_state},
      'zonda':{'level':_level(2 if zscore>=70 else 1 if zscore>=45 else 0),'value':zscore,'label':zonda},
    }
    rec={
      'rain':'Precaución en zonas con acumulación de agua y seguimiento de la evolución.',
      'storm':'Evitar actividades al aire libre si aumenta la actividad eléctrica y seguir las actualizaciones.',
      'hail':'Resguardar vehículos, cultivos y elementos expuestos si el potencial continúa aumentando.',
      'frost':'Proteger cultivos, animales e instalaciones sensibles al descenso térmico.',
      'wind':'Asegurar objetos exteriores y evitar árboles, postes y estructuras expuestas.',
      'zonda':'Reducir actividades al aire libre, asegurar objetos y extremar precauciones por viento y sequedad.',
    }
    active=[]
    for k,v in events.items():
      v['recommendation']=rec[k]
      if v['level']!='NORMAL':active.append(k)
    return {'id':zone['id'],'zone':zone['name'],'lat':zone['lat'],'lon':zone['lon'],'generatedAt':datetime.now(timezone.utc).isoformat(),'metrics':{'cape':round(cape),'liftedIndex':None if li is None else round(li,1),'shearKmh':round(shear),'freezingLevelM':None if freezing is None else round(freezing),'humidity':round(humidity),'gustKmh':round(gust)},'events':events,'activeEvents':active}


def register_alert_routes(app):
    @app.get('/alerts')
    def alerts(zone: str|None=Query(None)):
        selected=ZONES
        if zone:
            q=zone.strip().lower();selected=[z for z in ZONES if z['id']==q or q in z['name'].lower()]
            if not selected:return {'ok':False,'error':'Zona no encontrada','zones':[z['id'] for z in ZONES]}
        out=[];errors=[]
        for z in selected:
            try:out.append(_zone_state(z))
            except Exception as e:errors.append({'zone':z['name'],'error':str(e)})
        return {'ok':bool(out),'service':'Territoria Alerts','generatedAt':datetime.now(timezone.utc).isoformat(),'zones':out,'errors':errors}
