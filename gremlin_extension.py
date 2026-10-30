"""Extension GREMLIN segura para Territoria.

No inventa bucket, prefijo ni datos. Se integra a main.py con:
    from gremlin_extension import register_gremlin_routes
    register_gremlin_routes(app)

Cuando se verifique la fuente real, configurar:
    GREMLIN_BUCKET
    GREMLIN_PREFIX
    GREMLIN_REGION (opcional)
"""
import os
from datetime import datetime, timezone
from fastapi import Query


def register_gremlin_routes(app):
    @app.get('/gremlin')
    def gremlin_status(
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        radius: float = Query(30, ge=5, le=100),
    ):
        bucket = os.getenv('GREMLIN_BUCKET', '').strip()
        prefix = os.getenv('GREMLIN_PREFIX', '').strip()
        configured = bool(bucket and prefix)

        if not configured:
            return {
                'ok': False,
                'configured': False,
                'code': 'SOURCE_NOT_CONFIGURED',
                'product': 'GREMLIN',
                'source': 'CIRA/NOAA GeoCloud',
                'message': 'La fuente NetCDF GREMLIN real todavia no fue verificada ni configurada.',
                'lat': lat,
                'lon': lon,
                'radiusKm': radius,
                'generatedAt': datetime.now(timezone.utc).isoformat(),
                'maxDbz': None,
                'imageUrl': None,
            }

        # Bloque deliberadamente seguro: saber bucket/prefix no basta para procesar
        # el NetCDF hasta confirmar nombres de variables, proyeccion y permisos.
        return {
            'ok': False,
            'configured': True,
            'code': 'PROCESSOR_SCHEMA_PENDING',
            'product': 'GREMLIN',
            'source': 'CIRA/NOAA GeoCloud',
            'message': 'Fuente configurada; falta validar el esquema NetCDF antes de generar imagenes.',
            'lat': lat,
            'lon': lon,
            'radiusKm': radius,
            'generatedAt': datetime.now(timezone.utc).isoformat(),
            'maxDbz': None,
            'imageUrl': None,
        }
