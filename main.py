export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Accept",
      "Access-Control-Max-Age": "86400"
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }

    if (request.method !== "GET") {
      return json({ ok: false, error: "Método no permitido" }, 405, cors);
    }

    const incoming = new URL(request.url);
    let route = incoming.pathname.replace(/\/+$/, "");
    if (!route) route = "/";

    // La raíz responde desde el propio Worker. No depende del procesador.
    if (route === "/") {
      return json(
        {
          ok: true,
          service: "Territoria GOES-19 Gateway",
          version: "6.2.0",
          availableRoutes: ["/fires", "/lightning", "/rainfall"],
          processorConfigured: Boolean(env.GOES_PROCESSOR_URL),
          generatedAt: new Date().toISOString()
        },
        200,
        cors
      );
    }

    const allowedRoutes = new Set(["/fires", "/lightning", "/rainfall"]);
    if (!allowedRoutes.has(route)) {
      return json(
        {
          ok: false,
          error: "Ruta no encontrada",
          route,
          availableRoutes: ["/fires", "/lightning", "/rainfall"]
        },
        404,
        cors
      );
    }

    if (!env.GOES_PROCESSOR_URL) {
      return json(
        { ok: false, error: "Falta la variable GOES_PROCESSOR_URL", route },
        503,
        cors
      );
    }

    let upstream;
    try {
      upstream = new URL(env.GOES_PROCESSOR_URL);
    } catch {
      return json(
        { ok: false, error: "GOES_PROCESSOR_URL no es una URL válida", route },
        503,
        cors
      );
    }

    // GOES_PROCESSOR_URL debe ser la raíz del backend. Si por error termina
    // en /fires, /lightning o /rainfall, lo normalizamos antes de añadir la ruta.
    let basePath = upstream.pathname.replace(/\/+$/, "");
    basePath = basePath.replace(/\/(fires|lightning|rainfall)$/i, "");
    upstream.pathname = `${basePath}${route}` || route;
    upstream.search = "";

    // Copiar todos los parámetros de la petición original.
    incoming.searchParams.forEach((value, key) => {
      upstream.searchParams.set(key, value);
    });

    // Valores predeterminados por producto.
    if (route === "/fires") {
      setDefault(upstream.searchParams, "radius", "5000");
      setDefault(upstream.searchParams, "scenes", "6");
    }

    if (route === "/lightning") {
      // Limita los rayos a Argentina y evita bloquear GitHub Pages/Leaflet.
      upstream.searchParams.set("argentina", "true");
      setDefault(upstream.searchParams, "radius", "5000");
      setDefault(upstream.searchParams, "minutes", "15");
      setDefault(upstream.searchParams, "max_flashes", "1500");
    }

    if (route === "/rainfall") {
      setDefault(upstream.searchParams, "radius", "5000");
    }

    const lat = Number(upstream.searchParams.get("lat"));
    const lon = Number(upstream.searchParams.get("lon"));

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return json({ ok: false, error: "lat/lon requeridos", route }, 400, cors);
    }

    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return json(
        { ok: false, error: "Coordenadas fuera de rango", route },
        400,
        cors
      );
    }

    try {
      const attempts = route === "/lightning" ? 3 : 1;
      let response = null;
      let body = "";
      let data = null;
      let lastParseError = false;

      for (let attempt = 1; attempt <= attempts; attempt++) {
        response = await fetch(upstream.toString(), {
          method: "GET",
          headers: { Accept: "application/json" },
          cf: {
            cacheTtl: route === "/lightning" ? 20 : route === "/rainfall" ? 300 : 60,
            cacheEverything: true,
            cacheKey: upstream.toString()
          }
        });

        body = await response.text();
        lastParseError = false;
        try {
          data = JSON.parse(body);
        } catch {
          data = null;
          lastParseError = true;
        }

        if (response.ok && data && data.ok !== false) break;
        if (attempt < attempts) await sleep(attempt * 1500);
      }

      if (lastParseError || !data) {
        return json(
          {
            ok: false,
            error: "El procesador no devolvió JSON después de reintentar",
            route,
            attempts,
            upstreamStatus: response?.status || 502,
            upstreamUrl: upstream.toString(),
            preview: body.slice(0, 250)
          },
          502,
          cors
        );
      }

      if (!response.ok || data?.ok === false) {
        return json(
          {
            ok: false,
            error: data?.error || `Procesador GOES HTTP ${response.status}`,
            route,
            upstreamStatus: response.status,
            upstreamUrl: upstream.toString(),
            upstream: data
          },
          502,
          cors
        );
      }

      return json(
        {
          ...data,
          workerRoute: route,
          generatedAt: new Date().toISOString()
        },
        200,
        cors
      );
    } catch (error) {
      return json(
        {
          ok: false,
          error: error?.message || String(error),
          route,
          upstreamUrl: upstream.toString()
        },
        502,
        cors
      );
    }
  }
};

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function setDefault(params, key, value) {
  if (!params.has(key)) params.set(key, value);
}

function json(value, status, corsHeaders) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store"
    }
  });
}
