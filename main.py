import math, tempfile
from datetime import datetime, timezone
import boto3, numpy as np, xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pyproj import CRS, Transformer

app=FastAPI(title='Territoria GOES-19 Processor',version='4.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET'],allow_headers=['*'])
BUCKET='noaa-goes19'; PRODUCT='ABI-L2-FDCF'
s3=boto3.client('s3',config=Config(signature_version=UNSIGNED,retries={'max_attempts':3}))

def haversine(a,b,c,d):
 r=6371.; p1,p2=math.radians(a),math.radians(c); dp=math.radians(c-a); dl=math.radians(d-b); q=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2; return r*2*math.atan2(math.sqrt(q),math.sqrt(1-q))

def bearing(a,b,c,d):
 p1,p2=math.radians(a),math.radians(c); dl=math.radians(d-b); y=math.sin(dl)*math.cos(p2); x=math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl); deg=(math.degrees(math.atan2(y,x))+360)%360; return ['N','NE','E','SE','S','SO','O','NO'][round(deg/45)%8]

def latest_key():
 now=datetime.now(timezone.utc)
 for back in range(6):
  d=datetime.fromtimestamp(now.timestamp()-back*3600,timezone.utc)
  prefix=f'{PRODUCT}/{d.year}/{d.timetuple().tm_yday:03d}/{d.hour:02d}/'
  res=s3.list_objects_v2(Bucket=BUCKET,Prefix=prefix)
  keys=[x['Key'] for x in res.get('Contents',[]) if x['Key'].endswith('.nc')]
  if keys:return sorted(keys)[-1]
 raise RuntimeError('No se encontró archivo GOES-19 reciente')

def decoded_value(var,y,x):
 raw=var.values[y,x]; fill=var.attrs.get('_FillValue')
 if fill is not None and raw==fill:return None
 return float(raw)*float(var.attrs.get('scale_factor',1.0))+float(var.attrs.get('add_offset',0.0))

def scalar(ds,names,y,x):
 for n in names:
  if n in ds:
   try:return decoded_value(ds[n],y,x)
   except:pass
 return None

def extract(key,slat,slon,radius):
 with tempfile.NamedTemporaryFile(suffix='.nc') as f:
  s3.download_fileobj(BUCKET,key,f); f.flush()
  ds=xr.open_dataset(f.name,engine='h5netcdf',mask_and_scale=False)
  mask=np.asarray(ds['Mask'].values)
  codes=np.array([10,11,12,13,14,15,30,31,32,33,34,35],dtype=mask.dtype)
  ys,xs=np.where(np.isin(mask,codes))
  proj=ds['goes_imager_projection']; h=float(proj.perspective_point_height)
  crs=CRS.from_proj4(f'+proj=geos +h={h} +lon_0={float(proj.longitude_of_projection_origin)} +sweep={str(proj.sweep_angle_axis)} +ellps=GRS80')
  tr=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
  xv=np.asarray(ds['x'].values,dtype='float64')*float(ds['x'].attrs.get('scale_factor',1.0))+float(ds['x'].attrs.get('add_offset',0.0))
  yv=np.asarray(ds['y'].values,dtype='float64')*float(ds['y'].attrs.get('scale_factor',1.0))+float(ds['y'].attrs.get('add_offset',0.0))
  lons,lats=tr.transform(xv[xs]*h,yv[ys]*h)
  labels={10:'processed',11:'saturated',12:'cloudy',13:'high',14:'medium',15:'low',30:'processed-temporal',31:'saturated-temporal',32:'cloudy-temporal',33:'high-temporal',34:'medium-temporal',35:'low-temporal'}
  out=[]
  for y,x,la,lo in zip(ys,xs,lats,lons):
   if not np.isfinite(la) or not np.isfinite(lo):continue
   km=haversine(slat,slon,float(la),float(lo))
   code=int(mask[y,x])
   out.append({'id':f'{key}-{y}-{x}','lat':round(float(la),6),'lon':round(float(lo),6),'distanceKm':round(km,2),'bearing':bearing(slat,slon,float(la),float(lo)),'classification':labels.get(code,str(code)),'maskCode':code,'frpMw':scalar(ds,['Power','FRP'],y,x),'temperatureK':scalar(ds,['Temp','Temperature'],y,x),'sizeKm2':scalar(ds,['Area'],y,x),'time':datetime.now(timezone.utc).isoformat()})
  ds.close()
  return sorted(out,key=lambda z:z['distanceKm']),int(len(ys))

@app.get('/')
def root():
 return {'ok':True,'service':'Territoria GOES-19 Processor','version':'4.0.0','product':PRODUCT}

@app.get('/fires')
def fires(lat:float=Query(...),lon:float=Query(...),radius:float=Query(5000,ge=5,le=20000),satellite:str='G19',product:str=PRODUCT):
 try:
  key=latest_key(); detections,scene_total=extract(key,lat,lon,radius)
  return {'ok':True,'satellite':'GOES-19','product':PRODUCT,'sourceKey':key,'detections':detections,'total':len(detections),'sceneFirePixels':scene_total,'radiusKm':radius,'coverage':'full-disk'}
 except Exception as e:
  return {'ok':False,'error':str(e),'detections':[]}
