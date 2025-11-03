import React, { useState, useEffect, useRef } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Popup,
  GeoJSON,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
import "./RoutingMap.css";
import LoadingOverlay from "./LoadingOverlay";
import {
  computeRoute,
  computeDemoRoute,
  refreshRoutingGraph,
  getAllWardsFloodPrediction,
} from "./api";

// Ensure default Leaflet marker icons load correctly in bundlers
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

// Distinct start/end markers for visibility
const startIcon = L.divIcon({
  className: "start-marker",
  html: "S",
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});
const endIcon = L.divIcon({
  className: "end-marker",
  html: "E",
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

// Styles
const routeStyle = { color: "#2ecc71", weight: 6 };
const blockedStyle = { color: "#e74c3c", weight: 3, dashArray: "6 6", opacity: 0.8 };
const ghostStyle = { color: "#a3e4d7", weight: 6, opacity: 0.4 };

// Base map tile styles to reduce green tint when needed
const TILE_STYLES = {
  light: {
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    attribution:
      '&copy; OpenStreetMap contributors &copy; CARTO',
  },
  standard: {
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
  },
  dark: {
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    attribution:
      '&copy; OpenStreetMap contributors &copy; CARTO',
  },
};

// Mumbai default center
const MUMBAI_CENTER = [19.076, 72.8777];

// Demo scenarios (handpicked) + backend risk scenarios for deterministic detours
const DEMO_SCENARIOS = [
  {
    id: "bandra_andheri",
    label: "Bandra West → Andheri East",
    start: [19.0596, 72.8295],
    end: [19.1136, 72.8697],
    riskScenario: "central_flood",
  },
  {
    id: "colaba_goregaon",
    label: "Colaba → Goregaon",
    start: [18.9067, 72.8147],
    end: [19.155, 72.849],
    riskScenario: "western_line_flood",
  },
  {
    id: "kurla_borivali",
    label: "Kurla → Borivali",
    start: [19.0728, 72.8826],
    end: [19.2307, 72.8567],
    riskScenario: "western_line_flood",
  },
];

// Utility
const coordsToLatLngs = (coords) => coords.map(([lng, lat]) => [lat, lng]);

const FitToRoute = ({ line }) => {
  const map = useMap();
  useEffect(() => {
    if (map && line && line.length > 1) {
      const bounds = L.latLngBounds(line);
      map.fitBounds(bounds, { padding: [40, 40] });
    }
  }, [map, line]);
  return null;
};

const RoutingMap = () => {
  // Core routing state
  const [start, setStart] = useState(null); // [lat, lng]
  const [end, setEnd] = useState(null); // [lat, lng]
  const [route, setRoute] = useState(null); // GeoJSON feature
  const [blockedEdges, setBlockedEdges] = useState([]); // [[ [lat,lng], ... ], ...]
  const [metrics, setMetrics] = useState(null);

  // Params
  const [avoidThreshold, setAvoidThreshold] = useState(0.7);
  const [alpha, setAlpha] = useState(10);

  // UI state
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [message, setMessage] = useState("");
  const [showWards, setShowWards] = useState(true);
  const [showBlockedEdges, setShowBlockedEdges] = useState(true);
  const [baseMap, setBaseMap] = useState("light"); // light | standard | dark

  // Inputs (manual coords)
  const [startLat, setStartLat] = useState("");
  const [startLng, setStartLng] = useState("");
  const [endLat, setEndLat] = useState("");
  const [endLng, setEndLng] = useState("");

  // Wards overlay
  const [wardsGeoJson, setWardsGeoJson] = useState(null);
  const [wardRiskMap, setWardRiskMap] = useState({}); // ward_code -> probability

  // Animation state
  const [animatedLine, setAnimatedLine] = useState([]); // incrementally drawn polyline
  const [isAnimating, setIsAnimating] = useState(false);
  const [animSpeed, setAnimSpeed] = useState(1.0); // 0.5x .. 3x
  const animIndexRef = useRef(0);
  const animTimerRef = useRef(null);
  const moverRef = useRef(null); // moving marker

  // Animated counters for metrics
  const [animatedMetrics, setAnimatedMetrics] = useState({
    distance: 0,
    cost: 0,
    risk: 0,
  });
  const counterRAF = useRef(null);

  // Load wards + flood probabilities
  useEffect(() => {
    fetch("/mumbai-wards-cleaned.geojson")
      .then((r) => r.json())
      .then(setWardsGeoJson)
      .catch(() => {});

    (async () => {
      try {
        const all = await getAllWardsFloodPrediction();
        const map = {};
        (all.predictions || []).forEach((p) => {
          map[(p.ward_code || "").toUpperCase()] = p.bayesian_probability || 0;
        });
        setWardRiskMap(map);
      } catch (_) {}
    })();
  }, []);

  // Map click handler: Start → End → Reset to Start
  const handleMapClick = (e) => {
    const latlng = [e.latlng.lat, e.latlng.lng];
    if (!start) {
      setStart(latlng);
      setStartLat(latlng[0].toFixed(6));
      setStartLng(latlng[1].toFixed(6));
    } else if (!end) {
      setEnd(latlng);
      setEndLat(latlng[0].toFixed(6));
      setEndLng(latlng[1].toFixed(6));
    } else {
      // reset to new start
      setStart(latlng);
      setStartLat(latlng[0].toFixed(6));
      setStartLng(latlng[1].toFixed(6));
      setEnd(null);
      setEndLat("");
      setEndLng("");
      resetRouteState();
    }
  };

  const wardStyle = (feature) => {
    const code = (feature.properties?.ward_code || "").toUpperCase();
    const p = wardRiskMap[code] ?? 0;
    const th = Number(avoidThreshold) || 0.7;

    // In demo mode: highlight only scenario-affected wards as flooded
    if (mode === "demo" && wardsGeoJson) {
      const scenario = DEMO_SCENARIOS.find((d) => d.id === demoScenario)?.riskScenario || "central_flood";
      const selector = demoFloodSelectors[scenario];
      const affected = selector ? selector(feature) : false;
      if (!affected) {
        return { color: "#bbb", weight: 1, fillOpacity: 0 };
      }
      // affected → shade red regardless of p, but scale with p if present
      const base = Math.max(p, th);
      const t = Math.max(0, Math.min(1, (base - th) / Math.max(1e-6, 1 - th)));
      const lerp = (a, b, x) => Math.round(a + (b - a) * x);
      const R = lerp(254, 185, t);
      const G = lerp(202, 28, t);
      const B = lerp(202, 28, t);
      return { color: "#7f1d1d", weight: 2, fillColor: `rgb(${R},${G},${B})`, fillOpacity: 0.55 };
    }

    // Live mode: threshold-based
    if (p < th) {
      return { color: "#888", weight: 1, fillOpacity: 0 };
    }
    const t = Math.max(0, Math.min(1, (p - th) / Math.max(1e-6, 1 - th)));
    const lerp = (a, b, x) => Math.round(a + (b - a) * x);
    const R = lerp(254, 185, t);
    const G = lerp(202, 28, t);
    const B = lerp(202, 28, t);
    return { color: "#7f1d1d", weight: 2, fillColor: `rgb(${R},${G},${B})`, fillOpacity: 0.55 };
  };

  const onEachWard = (feature, layer) => {
    const name =
      feature.properties?.ward_name_full ||
      feature.properties?.name ||
      feature.properties?.NAME ||
      feature.properties?.ward_code ||
      "Ward";
    // Tooltip for quick identification
    layer.bindTooltip(name, {
      permanent: false,
      direction: "top",
      className: "routing-ward-tooltip",
      offset: [0, -8],
    });
    // Hover highlight
    layer.on({
      mouseover: (e) => {
        e.target.setStyle({ weight: 2.5, color: "#111", fillOpacity: 0.6 });
      },
      mouseout: (e) => {
        e.target.setStyle(wardStyle(feature));
      },
    });
  };

  const resetRouteState = () => {
    setRoute(null);
    setBlockedEdges([]);
    setMetrics(null);
    stopAnimation();
    setAnimatedLine([]);
    setAnimatedMetrics({ distance: 0, cost: 0, risk: 0 });
    setMessage("");
  };

  const assignStartFromInputs = () => {
    const lat = parseFloat(String(startLat).replace(",", "."));
    const lng = parseFloat(String(startLng).replace(",", "."));
    if (isFinite(lat) && isFinite(lng)) {
      setStart([lat, lng]);
      try { window.dispatchEvent(new Event("resize")); } catch (_) {}
    }
  };
  const assignEndFromInputs = () => {
    const lat = parseFloat(String(endLat).replace(",", "."));
    const lng = parseFloat(String(endLng).replace(",", "."));
    if (isFinite(lat) && isFinite(lng)) {
      setEnd([lat, lng]);
      try { window.dispatchEvent(new Event("resize")); } catch (_) {}
    }
  };
  const swapPoints = () => {
    const s = start ? [...start] : null;
    const e = end ? [...end] : null;
    setStart(e);
    setEnd(s);
    setStartLat(endLat);
    setStartLng(endLng);
    setEndLat(startLat);
    setEndLng(startLng);
    resetRouteState();
  };
  const getMyLocation = () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition((pos) => {
      const lat = pos.coords.latitude;
      const lng = pos.coords.longitude;
      setStart([lat, lng]);
      setStartLat(lat.toFixed(6));
      setStartLng(lng.toFixed(6));
    });
  };

  const [mode, setMode] = useState("demo"); // "live" | "demo"

  // Demo overlay segmentation: which wards are considered flooded per scenario
  const demoFloodSelectors = {
    central_flood: (feature) => {
      const name = (feature.properties?.ward_name_full || feature.properties?.name || "").toLowerCase();
      // Central-ish wards keywords; adjust as needed for your dataset
      return [
        "dadar",
        "mahima",
        "matunga",
        "sion",
        "parel",
        "wadala",
        "kurla",
      ].some((k) => name.includes(k));
    },
    western_line_flood: (feature) => {
      const name = (feature.properties?.ward_name_full || feature.properties?.name || "").toLowerCase();
      return [
        "bandra",
        "khar",
        "santacruz",
        "vile parle",
        "andheri",
        "jogeshwari",
        "goregaon",
        "malad",
        "kandivali",
        "borivali",
      ].some((k) => name.includes(k));
    },
  };

  const requestRoute = async (forcedScenario) => {
    if (!start || !end) return;
    setLoading(true);
    setLoadingMessage("Computing flood-aware route...");
    setMessage("");
    stopAnimation();
    setAnimatedLine([]);

    try {
      const args = {
        from_lat: start[0],
        from_lng: start[1],
        to_lat: end[0],
        to_lng: end[1],
        avoid_threshold: Number(avoidThreshold),
        alpha: Number(alpha),
      };

      // Live or Demo routing selection
      let feature;
      if (mode === "live") {
        feature = await computeRoute(args);
      } else {
        // demo
        const scenario = forcedScenario || DEMO_SCENARIOS.find((d) => d.id === demoScenario)?.riskScenario || "central_flood";
        feature = await computeDemoRoute({ ...args, scenario });
      }

      setRoute(feature);
      const props = feature.properties || {};
      const fullCoords = feature.geometry?.coordinates || [];
      const poly = coordsToLatLngs(fullCoords);

      // Metrics
      const m = {
        total_distance_m: props.total_distance_m || 0,
        total_cost: props.total_cost || 0,
        total_risk: Number((props.total_risk || 0).toFixed(3)),
        edges_blocked_count: props.edges_blocked_count || 0,
      };
      setMetrics(m);
      animateCounters(m);

      // Blocked edges
      const blocked = (props.blocked_edges || []).map((line) =>
        line.map(([lng, lat]) => [lat, lng])
      );
      setBlockedEdges(blocked);

      if (poly.length === 0) {
        setMessage("No route found. Try lowering alpha or threshold.");
      } else {
        // Prepare animation line (reset index)
        animIndexRef.current = 0;
        setAnimatedLine([]);
      }
    } catch (err) {
      // One-shot retry after brief delay to handle warm-up/transient timeouts
      console.log("First routing attempt failed, retrying...");
      setLoadingMessage("Initializing routing graph, please wait...");
      try {
        await new Promise((res) => setTimeout(res, 1000));
        const args = {
          from_lat: start[0],
          from_lng: start[1],
          to_lat: end[0],
          to_lng: end[1],
          avoid_threshold: Number(avoidThreshold),
          alpha: Number(alpha),
        };
        let feature;
        if (mode === "live") {
          feature = await computeRoute(args);
        } else {
          const scenario = forcedScenario || DEMO_SCENARIOS.find((d) => d.id === demoScenario)?.riskScenario || "central_flood";
          feature = await computeDemoRoute({ ...args, scenario });
        }

        setRoute(feature);
        const props = feature.properties || {};
        const fullCoords = feature.geometry?.coordinates || [];
        const poly = coordsToLatLngs(fullCoords);

        const m = {
          total_distance_m: props.total_distance_m || 0,
          total_cost: props.total_cost || 0,
          total_risk: Number((props.total_risk || 0).toFixed(3)),
          edges_blocked_count: props.edges_blocked_count || 0,
        };
        setMetrics(m);
        animateCounters(m);

        const blocked = (props.blocked_edges || []).map((line) =>
          line.map(([lng, lat]) => [lat, lng])
        );
        setBlockedEdges(blocked);

        if (poly.length === 0) {
          setMessage("No route found. Try lowering alpha or threshold.");
        } else {
          animIndexRef.current = 0;
          setAnimatedLine([]);
          console.log("Routing succeeded on retry");
        }
      } catch (e2) {
        console.error("Routing failed after retry:", e2);
        
        // Provide specific error messages based on error type
        let errorMsg = "Routing failed. ";
        if (e2.code === 'ECONNABORTED' || e2.message?.includes('timeout')) {
          errorMsg += "Request timed out. The routing graph might be initializing - please wait 30 seconds and try again.";
        } else if (e2.response?.status === 503) {
          errorMsg += "Routing service not available. " + (e2.response?.data?.detail || "Please check if osmnx and geopandas are installed.");
        } else if (e2.response?.status === 500) {
          errorMsg += "Server error: " + (e2.response?.data?.detail || "Check backend logs for details.");
        } else if (e2.code === 'ERR_NETWORK' || e2.message?.includes('Network Error')) {
          errorMsg += "Cannot connect to backend. Ensure it's running on http://127.0.0.1:8000";
        } else {
          errorMsg += e2.response?.data?.detail || e2.message || "Unknown error. Check console for details.";
        }
        
        setMessage(errorMsg);
      }
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  const animateCounters = ({ total_distance_m, total_cost, total_risk }) => {
    if (counterRAF.current) cancelAnimationFrame(counterRAF.current);
    const startTime = performance.now();
    const DUR = 900; // ms
    const from = { distance: 0, cost: 0, risk: 0 };

    const step = (t) => {
      const p = Math.min(1, (t - startTime) / DUR);
      const ease = 1 - Math.pow(1 - p, 3);
      setAnimatedMetrics({
        distance: total_distance_m * ease,
        cost: total_cost * ease,
        risk: total_risk * ease,
      });
      if (p < 1) {
        counterRAF.current = requestAnimationFrame(step);
      }
    };
    counterRAF.current = requestAnimationFrame(step);
  };

  const startAnimation = () => {
    if (!route?.geometry?.coordinates?.length || isAnimating) return;
    const poly = coordsToLatLngs(route.geometry.coordinates);
    setAnimatedLine([poly[0]]);
    animIndexRef.current = 1;
    setIsAnimating(true);

    if (!moverRef.current) {
      moverRef.current = L.marker(poly[0]);
    }

    // Timer - smoother pacing tied to animSpeed
    const baseMsPerPoint = 40; // lower = faster baseline
    const intervalMs = Math.max(16, baseMsPerPoint / Math.max(0.2, animSpeed));
    animTimerRef.current = setInterval(() => {
      const i = animIndexRef.current;
      if (i >= poly.length) {
        stopAnimation();
        return;
      }
      setAnimatedLine((prev) => {
        const next = [...prev, poly[i]];
        return next;
      });
      // Move the marker
      try {
        moverRef.current?.setLatLng(poly[i]);
      } catch (_) {}

      animIndexRef.current = i + 1;
    }, intervalMs);
  };

  const pauseAnimation = () => {
    if (animTimerRef.current) clearInterval(animTimerRef.current);
    animTimerRef.current = null;
    setIsAnimating(false);
  };

  const stopAnimation = () => {
    if (animTimerRef.current) clearInterval(animTimerRef.current);
    animTimerRef.current = null;
    setIsAnimating(false);
  };

  const refreshGraph = async () => {
    setLoading(true);
    setLoadingMessage("Refreshing routing graph...");
    setMessage("");
    try {
      await refreshRoutingGraph();
      setMessage("Graph refreshed.");
    } catch (e) {
      setMessage("Failed to refresh graph.");
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  // Demo runner: select scenario, request route, then animate
  const [demoScenario, setDemoScenario] = useState(DEMO_SCENARIOS[0].id);
  const runDemo = async () => {
    const scenario = DEMO_SCENARIOS.find((d) => d.id === demoScenario);
    if (!scenario) return;

    setStart(scenario.start);
    setEnd(scenario.end);
    setStartLat(scenario.start[0].toFixed(6));
    setStartLng(scenario.start[1].toFixed(6));
    setEndLat(scenario.end[0].toFixed(6));
    setEndLng(scenario.end[1].toFixed(6));

    await requestRoute(scenario.riskScenario);
    // Defer starting animation a bit to allow render
    setTimeout(() => startAnimation(), 300);
  };

  const RouteMover = () => {
    const map = useMap();
    useEffect(() => {
      if (!moverRef.current) return;
      // Attach/detach to map
      moverRef.current.addTo(map);
      return () => {
        try { moverRef.current.remove(); } catch (_) {}
      };
    }, [map]);
    return null;
  };

  // Derived
  const routeLatLngs = route?.geometry?.coordinates ? coordsToLatLngs(route.geometry.coordinates) : [];

  return (
    <div className="routing-scope routing-layout">
      <div className="routing-map" style={{ position: "relative" }}>
        <LoadingOverlay loading={loading} message={loadingMessage} />
        <MapContainer
          center={MUMBAI_CENTER}
          zoom={12}
          style={{ height: "100%", width: "100%", borderRadius: 8 }}
          whenCreated={(map) => {
            map.on("click", handleMapClick);
          }}
        >
          <TileLayer
            url={TILE_STYLES[baseMap].url}
            attribution={TILE_STYLES[baseMap].attribution}
          />

          {showWards && wardsGeoJson && <GeoJSON data={wardsGeoJson} style={wardStyle} onEachFeature={onEachWard} />}

          {start && (
            <Marker position={start} icon={startIcon}>
              <Popup>Start</Popup>
            </Marker>
          )}
          {end && (
            <Marker position={end} icon={endIcon}>
              <Popup>End</Popup>
            </Marker>
          )}

          {/* Ghost full route for context during animation */}
          {routeLatLngs.length > 0 && (
            <Polyline positions={routeLatLngs} pathOptions={ghostStyle} className="route-line" />
          )}

          {/* Animated route segment */}
          {animatedLine.length > 1 && (
            <Polyline positions={animatedLine} pathOptions={routeStyle} className="route-line" />
          )}

          {/* Static full route when not animating */}
          {!isAnimating && routeLatLngs.length > 0 && animatedLine.length === 0 && (
            <Polyline positions={routeLatLngs} pathOptions={routeStyle} className="route-line" />
          )}

          {/* Blocked edges */}
          {showBlockedEdges && blockedEdges.map((line, idx) => (
            <Polyline key={idx} positions={line} pathOptions={blockedStyle} className="blocked-line" />
          ))}

          {/* Keep moving marker attached */}
          <RouteMover />
          <FitToRoute line={routeLatLngs} />
        </MapContainer>
      </div>

      <div className="routing-panel">
        <h3>Flood-aware Routing</h3>
        <div className="routing-hint">
          Click on map to set Start, then End. Or enter coordinates below.
        </div>

        {/* Inputs */}
        <div className="routing-grid-2">
          <div>
            <label style={{ fontWeight: 600 }}>Start Lat</label>
            <input type="number" value={startLat} onChange={(e) => setStartLat(e.target.value)} placeholder="19.0596" className="input" />
          </div>
          <div>
            <label style={{ fontWeight: 600 }}>Start Lng</label>
            <input type="number" value={startLng} onChange={(e) => setStartLng(e.target.value)} placeholder="72.8295" className="input" />
          </div>
          <div>
            <label style={{ fontWeight: 600 }}>End Lat</label>
            <input type="number" value={endLat} onChange={(e) => setEndLat(e.target.value)} placeholder="19.1136" className="input" />
          </div>
          <div>
            <label style={{ fontWeight: 600 }}>End Lng</label>
            <input type="number" value={endLng} onChange={(e) => setEndLng(e.target.value)} placeholder="72.8697" className="input" />
          </div>
        </div>
        <div className="routing-grid-3">
          <button onClick={() => assignStartFromInputs()} className="btn">Set Start</button>
          <button onClick={() => assignEndFromInputs()} className="btn">Set End</button>
          <button onClick={() => swapPoints()} className="btn">Swap</button>
        </div>
        <div className="routing-grid-2" style={{ marginBottom: 12 }}>
          <button onClick={() => getMyLocation()} className="btn">Use My Location</button>
          <button onClick={() => { setStart(null); setEnd(null); setStartLat(""); setStartLng(""); setEndLat(""); setEndLng(""); resetRouteState(); }} className="btn">Clear</button>
        </div>

        {/* Params */}
        <div style={{ marginBottom: 12 }}>
          <label>avoid_threshold (default 0.7)</label>
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={avoidThreshold}
            onChange={(e) => setAvoidThreshold(e.target.value)}
            className="input"
          />
        </div>
        <div style={{ marginBottom: 12 }}>
          <label>alpha (default 10)</label>
          <input
            type="number"
            step="1"
            min="0"
            value={alpha}
            onChange={(e) => setAlpha(e.target.value)}
            className="input"
          />
        </div>

        <button onClick={() => requestRoute()} disabled={!start || !end || loading} className="btn-wide">
          {loading ? "Routing..." : "Compute Route"}
        </button>
        <button onClick={() => refreshGraph()} disabled={loading} className="btn-wide" style={{ marginTop: 8 }}>
          Refresh Graph
        </button>

        {/* Toggles */}
        {/* Mode toggle */}
        <div className="toggle-row">
          <div className="mode-label">Mode:</div>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="radio" name="mode" value="live" checked={mode === "live"} onChange={() => setMode("live")} />
            Live
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="radio" name="mode" value="demo" checked={mode === "demo"} onChange={() => setMode("demo")} />
            Demo
          </label>
        </div>

        <div className="toggle-row" style={{ marginTop: 8 }}>
          <div className="mode-label">Base map:</div>
          <select value={baseMap} onChange={(e) => setBaseMap(e.target.value)} className="input" style={{ maxWidth: 160 }}>
            <option value="light">Light</option>
            <option value="standard">Standard</option>
            <option value="dark">Dark</option>
          </select>
        </div>

        <div className="toggle-row">
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showWards} onChange={(e) => setShowWards(e.target.checked)} />
            Show Flood Overlay
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showBlockedEdges} onChange={(e) => setShowBlockedEdges(e.target.checked)} />
            Show Blocked
          </label>
        </div>

        {/* Animation controls */}
        <div className="section">
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Route Animation</div>
          <div className="routing-grid-3" style={{ marginBottom: 0 }}>
            <button onClick={() => startAnimation()} disabled={!route || isAnimating} className="btn">Play</button>
            <button onClick={() => pauseAnimation()} disabled={!isAnimating} className="btn">Pause</button>
            <button onClick={() => stopAnimation()} disabled={!route} className="btn">Stop</button>
          </div>
          <div style={{ marginTop: 10 }}>
            <label className="speed-label">Speed: {animSpeed.toFixed(1)}x</label>
            <input type="range" min="0.5" max="3" step="0.1" value={animSpeed} onChange={(e) => setAnimSpeed(parseFloat(e.target.value))} style={{ width: "100%" }} />
          </div>
        </div>

        {/* Demo */}
        {mode === "demo" && (
          <div className="section">
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Demo Scenarios</div>
            <select value={demoScenario} onChange={(e) => setDemoScenario(e.target.value)} className="input" style={{ marginBottom: 8 }}>
              {DEMO_SCENARIOS.map((d) => (
                <option key={d.id} value={d.id}>{d.label}</option>
              ))}
            </select>
            <button onClick={() => runDemo()} className="btn-wide">Run Demo</button>
          </div>
        )}

        {/* Metrics */}
        {metrics && (
          <div style={{ marginTop: 16, fontSize: 14 }}>
            <div><strong>Total Distance:</strong> {Math.round(animatedMetrics.distance)} m</div>
            <div><strong>Total Cost:</strong> {Math.round(animatedMetrics.cost)}</div>
            <div><strong>Avg Risk:</strong> {animatedMetrics.risk.toFixed(3)}</div>
            <div><strong>Edges Blocked:</strong> {metrics.edges_blocked_count}</div>
          </div>
        )}

        {message && (
          <div style={{ marginTop: 12, color: "#e74c3c" }}>{message}</div>
        )}

        <div style={{ marginTop: 16, fontSize: 12, color: "#666" }}>
          Tip: If no route is found, try decreasing alpha or increasing avoid_threshold.
        </div>
      </div>
    </div>
  );
};

export default RoutingMap;
