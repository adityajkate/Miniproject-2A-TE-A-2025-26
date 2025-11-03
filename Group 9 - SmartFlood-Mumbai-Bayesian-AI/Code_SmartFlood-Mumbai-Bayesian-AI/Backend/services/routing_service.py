"""
Routing service for flood-aware A* pathfinding in Mumbai.

- Builds and caches an OSMnx drive graph clipped to Mumbai wards
- Assigns each edge a ward_code based on midpoint point-in-polygon
- On each request, fetches ward flood risks from local FastAPI (/predict/all-wards)
- Computes dynamic edge weights: weight = length * (1 + alpha * flood_risk)
- Blocks edges whose ward risk >= avoid_threshold (removed by subgraph view)
- Uses NetworkX A* with haversine heuristic
- Returns GeoJSON LineString and metrics
"""

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import requests
import networkx as nx

try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point, LineString, mapping
    from shapely.ops import unary_union
except Exception as e:
    # These imports are required at runtime. We keep a clear message if missing.
    raise RuntimeError(
        "Routing dependencies missing. Please install osmnx, geopandas, shapely."
    ) from e


# Constants/paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BACKEND_DIR, "Dataset")
WARDS_GEOJSON = os.path.join(DATASET_DIR, "mumbai-wards-cleaned.geojson")
GRAPH_PATH = os.path.join(BACKEND_DIR, "models", "mumbai_drive.graphml")

# Local API base to fetch flood risks
LOCAL_API_BASE = os.environ.get("ROUTING_API_BASE", "http://127.0.0.1:8000")


@dataclass
class RouteParams:
    avoid_threshold: float = 0.7
    alpha: float = 10.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two (lat, lon)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def normalize_ward_code(code: str) -> str:
    if not code:
        return ""
    # Keep letters only and uppercase for robust matching (e.g., "G/S" -> "GS")
    return "".join([ch for ch in code.upper() if ch.isalpha()])


class GraphManager:
    def __init__(self):
        self.G: Optional[nx.MultiDiGraph] = None
        self.nodes_gdf: Optional[gpd.GeoDataFrame] = None
        self.edges_gdf: Optional[gpd.GeoDataFrame] = None
        self.wards_gdf: Optional[gpd.GeoDataFrame] = None
        self.wards_index = None
        self._last_built_at: Optional[float] = None
        # Concurrency guard to prevent overlapping graph builds/refreshes
        import threading

        self._build_lock = threading.Lock()

    def ensure_graph(self, force_rebuild: bool = False) -> None:
        """Load graph from disk or build it if missing.
        Also reload from disk if a newer cached graph exists (e.g., after /graph/refresh in another worker).
        """
        with self._build_lock:
            # Check on-disk cache timestamp
            disk_mtime = None
            try:
                if os.path.exists(GRAPH_PATH):
                    disk_mtime = os.path.getmtime(GRAPH_PATH)
            except Exception:
                disk_mtime = None

            # If graph in memory and not forced, consider disk freshness
            if self.G is not None and not force_rebuild:
                if (
                    self._last_built_at is not None
                    and disk_mtime is not None
                    and disk_mtime > self._last_built_at
                ):
                    # Reload newer graph from disk
                    try:
                        self.G = ox.load_graphml(GRAPH_PATH)
                        try:
                            self.G = ox.distance.add_edge_lengths(self.G)
                        except AttributeError:
                            for u, v, k, data in self.G.edges(keys=True, data=True):
                                if "length" not in data:
                                    try:
                                        geom = data.get("geometry")
                                        if isinstance(geom, LineString):
                                            data["length"] = float(
                                                geom.length * 111139.0
                                            )
                                        else:
                                            y1, x1 = (
                                                self.G.nodes[u]["y"],
                                                self.G.nodes[u]["x"],
                                            )
                                            y2, x2 = (
                                                self.G.nodes[v]["y"],
                                                self.G.nodes[v]["x"],
                                            )
                                            data["length"] = haversine_m(y1, x1, y2, x2)
                                    except Exception:
                                        data["length"] = 1.0
                    except Exception:
                        # If reload fails, fall through to full rebuild
                        pass
                    else:
                        # Update GDFs and wards after reload
                        self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G)
                        self.nodes_gdf["lat"] = self.nodes_gdf.geometry.y
                        self.nodes_gdf["lon"] = self.nodes_gdf.geometry.x

                        if not os.path.exists(WARDS_GEOJSON):
                            raise FileNotFoundError(
                                f"Wards GeoJSON not found at {WARDS_GEOJSON}"
                            )
                        self.wards_gdf = gpd.read_file(WARDS_GEOJSON)

                        if "ward_code" in self.wards_gdf.columns:
                            self.wards_gdf["ward_code_norm"] = self.wards_gdf[
                                "ward_code"
                            ].apply(normalize_ward_code)
                        elif "Name" in self.wards_gdf.columns:
                            self.wards_gdf["ward_code_norm"] = self.wards_gdf[
                                "Name"
                            ].apply(normalize_ward_code)
                        else:
                            self.wards_gdf["ward_code_norm"] = self.wards_gdf.get(
                                "ward_name_full", ""
                            ).apply(
                                lambda x: normalize_ward_code(
                                    str(x).split(" ")[0] if x else ""
                                )
                            )
                        self.wards_gdf = self.wards_gdf.set_geometry(
                            self.wards_gdf.geometry
                        )
                        self.wards_gdf = self.wards_gdf.to_crs(epsg=4326)
                        try:
                            self.wards_index = self.wards_gdf.sindex
                        except Exception:
                            self.wards_index = None

                        self._assign_wards_to_edges()
                        self._last_built_at = (
                            disk_mtime if disk_mtime is not None else time.time()
                        )
                        return
                # Already loaded and up-to-date
                return

            # Load wards
            if not os.path.exists(WARDS_GEOJSON):
                raise FileNotFoundError(f"Wards GeoJSON not found at {WARDS_GEOJSON}")
            self.wards_gdf = gpd.read_file(WARDS_GEOJSON)

            # Build from cache if available
            if os.path.exists(GRAPH_PATH) and not force_rebuild:
                self.G = ox.load_graphml(GRAPH_PATH)
                # Ensure edges have length attribute
                try:
                    self.G = ox.distance.add_edge_lengths(self.G)
                except AttributeError:
                    # Fallback for very old/new OSMnx versions
                    for u, v, k, data in self.G.edges(keys=True, data=True):
                        if "length" not in data:
                            try:
                                geom = data.get("geometry")
                                if isinstance(geom, LineString):
                                    data["length"] = float(geom.length * 111139.0)
                                else:
                                    y1, x1 = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
                                    y2, x2 = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
                                    data["length"] = haversine_m(y1, x1, y2, x2)
                            except Exception:
                                data["length"] = 1.0
            else:
                # Build polygon from wards union to tightly clip the graph
                wards_union = unary_union(self.wards_gdf.geometry)
                self.G = ox.graph_from_polygon(
                    wards_union, network_type="drive", simplify=True
                )
                try:
                    self.G = ox.distance.add_edge_lengths(self.G)
                except AttributeError:
                    for u, v, k, data in self.G.edges(keys=True, data=True):
                        if "length" not in data:
                            try:
                                geom = data.get("geometry")
                                if isinstance(geom, LineString):
                                    data["length"] = float(geom.length * 111139.0)
                                else:
                                    y1, x1 = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
                                    y2, x2 = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
                                    data["length"] = haversine_m(y1, x1, y2, x2)
                            except Exception:
                                data["length"] = 1.0
                # Persist to disk for reuse
                os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
                ox.save_graphml(self.G, GRAPH_PATH)
                try:
                    disk_mtime = os.path.getmtime(GRAPH_PATH)
                except Exception:
                    pass

            # Convert to GeoDataFrames for spatial operations
            self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G)
            self.nodes_gdf["lat"] = self.nodes_gdf.geometry.y
            self.nodes_gdf["lon"] = self.nodes_gdf.geometry.x

            # Prepare ward spatial index
            if "ward_code" in self.wards_gdf.columns:
                self.wards_gdf["ward_code_norm"] = self.wards_gdf["ward_code"].apply(
                    normalize_ward_code
                )
            elif "Name" in self.wards_gdf.columns:
                self.wards_gdf["ward_code_norm"] = self.wards_gdf["Name"].apply(
                    normalize_ward_code
                )
            else:
                self.wards_gdf["ward_code_norm"] = self.wards_gdf.get(
                    "ward_name_full", ""
                ).apply(
                    lambda x: normalize_ward_code(str(x).split(" ")[0] if x else "")
                )
            self.wards_gdf = self.wards_gdf.set_geometry(self.wards_gdf.geometry)
            self.wards_gdf = self.wards_gdf.to_crs(epsg=4326)
            try:
                self.wards_index = self.wards_gdf.sindex
            except Exception:
                self.wards_index = None

            # Assign ward to each edge by midpoint
            self._assign_wards_to_edges()
            self._last_built_at = disk_mtime if disk_mtime is not None else time.time()

    def _assign_wards_to_edges(self) -> None:
        def edge_midpoint(row) -> Point:
            geom = row.get("geometry")
            if isinstance(geom, LineString):
                try:
                    return geom.interpolate(0.5, normalized=True)
                except Exception:
                    pass
            # Fallback: average of endpoints
            u, v, key = row["u"], row["v"], row.get("key", 0)
            yu, xu = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
            yv, xv = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
            return Point((xu + xv) / 2.0, (yu + yv) / 2.0)

        # Build a list of midpoints
        midpoints: List[Point] = []
        for _, row in self.edges_gdf.iterrows():
            midpoints.append(edge_midpoint(row))
        mp_gdf = gpd.GeoDataFrame(
            self.edges_gdf.copy(), geometry=midpoints, crs="EPSG:4326"
        )

        ward_codes: List[str] = []
        for idx, pt in mp_gdf.geometry.items():
            # Spatial index for candidate wards if available; otherwise, brute-force scan
            cand_idx: List[int]
            try:
                if self.wards_index is not None:
                    cand_idx = list(self.wards_index.query(pt))
                else:
                    # No spatial index (rtree/pygeos missing); fallback to all wards
                    cand_idx = list(range(len(self.wards_gdf)))
            except Exception:
                try:
                    alt_index = self.wards_gdf.sindex
                    if alt_index is not None:
                        cand_idx = list(alt_index.query(pt))
                    else:
                        cand_idx = list(range(len(self.wards_gdf)))
                except Exception:
                    cand_idx = list(range(len(self.wards_gdf)))

            code = None
            for wi in cand_idx:
                poly = self.wards_gdf.iloc[wi].geometry
                if poly.contains(pt) or poly.touches(pt):
                    code = self.wards_gdf.iloc[wi]["ward_code_norm"]
                    break
            ward_codes.append(code or "")

        self.edges_gdf["ward_code_norm"] = ward_codes
        # Push ward assignment back to graph edges
        for (u, v, k), ward_code in zip(self.edges_gdf.index, ward_codes):
            if self.G.has_edge(u, v, k):
                self.G[u][v][k]["ward_code_norm"] = ward_code

    def _fetch_ward_risks(self) -> Dict[str, float]:
        """Return dict: normalized_ward_code -> flood_probability (0..1).
        Tries internal API; falls back to demo risks to avoid deadlocks when single-worker server handles /route.
        """
        # Allow forcing demo via env for stability/testing
        if os.environ.get("ROUTING_USE_DEMO", "0").strip() in ("1", "true", "True"):
            return self.generate_demo_ward_risks()

        url = f"{LOCAL_API_BASE}/predict/all-wards"
        try:
            # Reasonable timeout to avoid blocking while giving API time to respond
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            risks: Dict[str, float] = {}
            for pred in data.get("predictions", []):
                raw_code = pred.get("ward_code")
                prob = float(pred.get("bayesian_probability", 0.0))
                risks[normalize_ward_code(raw_code)] = max(0.0, min(1.0, prob))
            # If API returned nothing, fallback to demo risks
            if not risks:
                return self.generate_demo_ward_risks()
            return risks
        except Exception:
            # Fallback prevents deadlocks/timeouts if API call cannot complete
            return self.generate_demo_ward_risks()

    def _edge_risk(self, data: Dict[str, Any], ward_risks: Dict[str, float]) -> float:
        wc = normalize_ward_code(data.get("ward_code_norm", ""))
        return ward_risks.get(wc, 0.0)

    def astar_route(
        self,
        from_lat: float,
        from_lng: float,
        to_lat: float,
        to_lng: float,
        params: RouteParams,
        show_blocked_geoms: bool = False,
        blocked_geom_limit: int = 500,
    ) -> Dict[str, Any]:
        """Flood-aware routing with graceful fallback.
        1) Try hard-avoid: remove edges with risk >= avoid_threshold.
        2) If no path, retry soft-avoid: keep all edges but heavily weight by risk.
        3) Optionally include a limited set of blocked edge geometries for visualization.
        """
        self.ensure_graph()

        # Nearest nodes
        orig = ox.distance.nearest_nodes(self.G, X=from_lng, Y=from_lat)
        dest = ox.distance.nearest_nodes(self.G, X=to_lng, Y=to_lat)

        # Fetch current ward risks
        ward_risks = self._fetch_ward_risks()

        # Identify blocked edges
        blocked_edges = set()
        for u, v, k, data in self.G.edges(keys=True, data=True):
            risk = self._edge_risk(data, ward_risks)
            if risk >= params.avoid_threshold:
                blocked_edges.add((u, v, k))

        # Create a filtered view that hides blocked edges (hard-avoid)
        def edge_filter(u, v, k):
            return (u, v, k) not in blocked_edges

        H = nx.subgraph_view(self.G, filter_edge=edge_filter)

        # Heuristic function based on haversine
        def hfun(n1: int, n2: int) -> float:
            y1, x1 = self.G.nodes[n1]["y"], self.G.nodes[n1]["x"]
            y2, x2 = self.G.nodes[n2]["y"], self.G.nodes[n2]["x"]
            return haversine_m(y1, x1, y2, x2)

        # Weight function reflecting flood risk
        def weight(u: int, v: int, data: Dict[str, Any]) -> float:
            length = float(data.get("length", 1.0))
            risk = self._edge_risk(data, ward_risks)
            return length * (1.0 + params.alpha * risk)

        # Try A* with hard-avoid first
        path = None
        try:
            path = nx.astar_path(H, orig, dest, heuristic=hfun, weight=weight)
            used_soft_fallback = False
        except nx.NetworkXNoPath:
            # Soft fallback: don't block edges; just weight them by risk
            try:
                used_soft_fallback = True
                path = nx.astar_path(self.G, orig, dest, heuristic=hfun, weight=weight)
            except nx.NetworkXNoPath:
                # Still no path (graph disconnected) → return structured no-route
                return {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": []},
                    "properties": {
                        "message": "No route found even after soft fallback",
                        "total_distance_m": 0.0,
                        "total_cost": 0.0,
                        "total_risk": 0.0,
                        "edges_blocked_count": len(blocked_edges),
                        "avoid_threshold": params.avoid_threshold,
                        "alpha": params.alpha,
                        "fallback_mode": "none",
                    },
                }

        # Build coordinates along the chosen edges
        coords: List[Tuple[float, float]] = []  # (lon, lat)
        total_distance = 0.0
        total_cost = 0.0
        risk_length_sum = 0.0

        for u, v in zip(path[:-1], path[1:]):
            # Choose best key by weight among parallel edges
            best_k = None
            best_w = float("inf")
            best_data = None
            for k, data in self.G[u][v].items():
                # If hard-avoid was used, skip blocked edges when picking geometry
                if (u, v, k) in blocked_edges and not used_soft_fallback:
                    continue
                w = weight(u, v, data)
                if w < best_w:
                    best_w = w
                    best_k = k
                    best_data = data

            if best_data is None:
                # Shouldn't happen since path is feasible
                continue

            geom = best_data.get("geometry")
            if isinstance(geom, LineString):
                seg_coords = list(geom.coords)
                if coords and seg_coords:
                    # Avoid duplicate coordinate at joins
                    if coords[-1] == (seg_coords[0][0], seg_coords[0][1]):
                        seg_coords = seg_coords[1:]
                coords.extend([(x, y) for x, y, *rest in seg_coords])
            else:
                # Fallback: straight line between nodes
                y1, x1 = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
                y2, x2 = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
                if not coords or coords[-1] != (x1, y1):
                    coords.append((x1, y1))
                coords.append((x2, y2))

            length = float(
                best_data.get(
                    "length",
                    haversine_m(
                        self.G.nodes[u]["y"],
                        self.G.nodes[u]["x"],
                        self.G.nodes[v]["y"],
                        self.G.nodes[v]["x"],
                    ),
                )
            )
            risk = self._edge_risk(best_data, ward_risks)
            total_distance += length
            total_cost += length * (1.0 + params.alpha * risk)
            risk_length_sum += risk * length

        avg_risk = (risk_length_sum / total_distance) if total_distance > 0 else 0.0

        # Collect blocked edge geometries for frontend visualization (optional + limited)
        blocked_lines: List[List[Tuple[float, float]]] = []
        if show_blocked_geoms and blocked_geom_limit > 0:
            cnt = 0
            for u, v, k in blocked_edges:
                if cnt >= blocked_geom_limit:
                    break
                data = self.G[u][v][k]
                geom = data.get("geometry")
                if isinstance(geom, LineString):
                    blocked_lines.append([(x, y) for x, y, *rest in geom.coords])
                else:
                    y1, x1 = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
                    y2, x2 = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
                    blocked_lines.append([(x1, y1), (x2, y2)])
                cnt += 1

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "total_distance_m": round(total_distance, 2),
                "total_cost": round(total_cost, 2),
                "total_risk": round(avg_risk, 4),
                "edges_blocked_count": len(blocked_edges),
                "avoid_threshold": params.avoid_threshold,
                "alpha": params.alpha,
                "blocked_edges": blocked_lines,
                "fallback_mode": "soft" if used_soft_fallback else "hard",
            },
        }
        return feature

    def generate_demo_ward_risks(
        self, scenario: str = "central_flood"
    ) -> Dict[str, float]:
        """Create a synthetic ward risk map for demo purposes.
        Returns dict: normalized_ward_code -> flood_probability (0..1)
        """
        self.ensure_graph()
        risks: Dict[str, float] = {}
        # Default low risk for all wards
        for _, row in self.wards_gdf.iterrows():
            code = normalize_ward_code(str(row.get("ward_code_norm", "")))
            if code:
                risks[code] = 0.1
        scenario = (scenario or "").lower()

        def set_high(codes: List[str]):
            for c in codes:
                risks[normalize_ward_code(c)] = 0.9

        if scenario in ("central_flood", "central"):
            set_high(
                ["GN", "GS", "HE", "HW", "KE", "KW", "PN"]
            )  # central-west corridor
        elif scenario in ("western_line_flood", "western"):
            set_high(["HW", "KW", "PN", "RS", "RC", "RN"])  # Western Railway corridor
        elif scenario in ("eastern_flood", "eastern"):
            set_high(["ME", "MW", "N", "S", "T", "L"])  # Eastern suburbs
        else:
            # Generic pattern: every third ward becomes high risk
            idx = 0
            for k in sorted(risks.keys()):
                if idx % 3 == 0:
                    risks[k] = 0.85
                idx += 1
        return risks

    def astar_route_demo(
        self,
        from_lat: float,
        from_lng: float,
        to_lat: float,
        to_lng: float,
        params: RouteParams,
        scenario: str = "central_flood",
        show_blocked_geoms: bool = False,
        blocked_geom_limit: int = 500,
    ) -> Dict[str, Any]:
        """A* route using synthetic demo flood risks to showcase detours."""
        self.ensure_graph()

        # Nearest nodes
        orig = ox.distance.nearest_nodes(self.G, X=from_lng, Y=from_lat)
        dest = ox.distance.nearest_nodes(self.G, X=to_lng, Y=to_lat)

        # Use synthetic ward risks for demo
        ward_risks = self.generate_demo_ward_risks(scenario)

        # Identify blocked edges
        blocked_edges = set()
        for u, v, k, data in self.G.edges(keys=True, data=True):
            risk = self._edge_risk(data, ward_risks)
            if risk >= params.avoid_threshold:
                blocked_edges.add((u, v, k))

        # Create a filtered view that hides blocked edges
        def edge_filter(u, v, k):
            return (u, v, k) not in blocked_edges

        H = nx.subgraph_view(self.G, filter_edge=edge_filter)

        # Heuristic function based on haversine
        def hfun(n1: int, n2: int) -> float:
            y1, x1 = self.G.nodes[n1]["y"], self.G.nodes[n1]["x"]
            y2, x2 = self.G.nodes[n2]["y"], self.G.nodes[n2]["x"]
            return haversine_m(y1, x1, y2, x2)

        # Weight function reflecting flood risk
        def weight(u: int, v: int, data: Dict[str, Any]) -> float:
            length = float(data.get("length", 1.0))
            risk = self._edge_risk(data, ward_risks)
            return length * (1.0 + params.alpha * risk)

        # Run A* with demo risks: try hard-avoid first, then soft fallback
        path = None
        try:
            path = nx.astar_path(H, orig, dest, heuristic=hfun, weight=weight)
            used_soft_fallback = False
        except nx.NetworkXNoPath:
            try:
                used_soft_fallback = True
                path = nx.astar_path(self.G, orig, dest, heuristic=hfun, weight=weight)
            except nx.NetworkXNoPath:
                return {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": []},
                    "properties": {
                        "message": "No route found considering demo flood constraints even after soft fallback",
                        "total_distance_m": 0.0,
                        "total_cost": 0.0,
                        "total_risk": 0.0,
                        "edges_blocked_count": len(blocked_edges),
                        "avoid_threshold": params.avoid_threshold,
                        "alpha": params.alpha,
                        "blocked_edges": [],
                        "demo_scenario": scenario,
                        "fallback_mode": "none",
                    },
                }

        # Build coordinates along the chosen edges
        coords: List[Tuple[float, float]] = []  # (lon, lat)
        total_distance = 0.0
        total_cost = 0.0
        risk_length_sum = 0.0

        for u, v in zip(path[:-1], path[1:]):
            # Choose best key by weight among parallel edges
            best_k = None
            best_w = float("inf")
            best_data = None
            for k, data in self.G[u][v].items():
                # If hard-avoid was used, skip blocked edges when picking geometry
                if (u, v, k) in blocked_edges and not used_soft_fallback:
                    continue
                w = weight(u, v, data)
                if w < best_w:
                    best_w = w
                    best_k = k
                    best_data = data

            if best_data is None:
                continue

            geom = best_data.get("geometry")
            if isinstance(geom, LineString):
                seg_coords = list(geom.coords)
                if coords and seg_coords:
                    if coords[-1] == (seg_coords[0][0], seg_coords[0][1]):
                        seg_coords = seg_coords[1:]
                coords.extend([(x, y) for x, y, *rest in seg_coords])
            else:
                y1, x1 = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
                y2, x2 = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
                if not coords or coords[-1] != (x1, y1):
                    coords.append((x1, y1))
                coords.append((x2, y2))

            length = float(
                best_data.get(
                    "length",
                    haversine_m(
                        self.G.nodes[u]["y"],
                        self.G.nodes[u]["x"],
                        self.G.nodes[v]["y"],
                        self.G.nodes[v]["x"],
                    ),
                )
            )
            risk = self._edge_risk(best_data, ward_risks)
            total_distance += length
            total_cost += length * (1.0 + params.alpha * risk)
            risk_length_sum += risk * length

        avg_risk = (risk_length_sum / total_distance) if total_distance > 0 else 0.0

        # Collect blocked edge geometries for frontend visualization (optional + limited)
        blocked_lines: List[List[Tuple[float, float]]] = []
        if show_blocked_geoms and blocked_geom_limit > 0:
            cnt = 0
            for u, v, k in blocked_edges:
                if cnt >= blocked_geom_limit:
                    break
                data = self.G[u][v][k]
                geom = data.get("geometry")
                if isinstance(geom, LineString):
                    blocked_lines.append([(x, y) for x, y, *rest in geom.coords])
                else:
                    y1, x1 = self.G.nodes[u]["y"], self.G.nodes[u]["x"]
                    y2, x2 = self.G.nodes[v]["y"], self.G.nodes[v]["x"]
                    blocked_lines.append([(x1, y1), (x2, y2)])
                cnt += 1

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "total_distance_m": round(total_distance, 2),
                "total_cost": round(total_cost, 2),
                "total_risk": round(avg_risk, 4),
                "edges_blocked_count": len(blocked_edges),
                "avoid_threshold": params.avoid_threshold,
                "alpha": params.alpha,
                "blocked_edges": blocked_lines,
                "demo_scenario": scenario,
                "fallback_mode": "soft" if used_soft_fallback else "hard",
            },
        }
        return feature

    def refresh_graph(self) -> Dict[str, Any]:
        self.ensure_graph(force_rebuild=True)
        return {
            "status": "ok",
            "message": "Graph rebuilt",
            "built_at": self._last_built_at,
            "nodes": int(self.G.number_of_nodes()) if self.G is not None else 0,
            "edges": int(self.G.number_of_edges()) if self.G is not None else 0,
            "graph_path": GRAPH_PATH,
        }


# Singleton instance
GRAPH_MANAGER = GraphManager()

if __name__ == "__main__":
    # Minimal CLI/demo to produce output when run directly
    import argparse

    parser = argparse.ArgumentParser(description="Flood-aware routing demo")
    parser.add_argument(
        "--from-lat", type=float, default=19.0760, help="Origin latitude"
    )
    parser.add_argument(
        "--from-lng", type=float, default=72.8777, help="Origin longitude"
    )
    parser.add_argument(
        "--to-lat", type=float, default=19.2183, help="Destination latitude"
    )
    parser.add_argument(
        "--to-lng", type=float, default=72.9781, help="Destination longitude"
    )
    parser.add_argument(
        "--avoid-threshold",
        type=float,
        default=0.7,
        help="Block edges with risk >= threshold (0..1)",
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0, help="Risk cost multiplier"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Use synthetic demo risks or call local API for live risks",
    )
    parser.add_argument(
        "--scenario",
        default="central_flood",
        help="Demo scenario when --mode=demo (e.g., central_flood, western, eastern)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild the OSMnx graph (slow; caches afterward)",
    )

    args = parser.parse_args()

    try:
        if args.force_rebuild:
            GRAPH_MANAGER.ensure_graph(force_rebuild=True)
        else:
            GRAPH_MANAGER.ensure_graph()

        params = RouteParams(avoid_threshold=args.avoid_threshold, alpha=args.alpha)

        if args.mode == "demo":
            feature = GRAPH_MANAGER.astar_route_demo(
                (
                    args["from_lat"]
                    if isinstance(args, dict) and "from_lat" in args
                    else args.from_lat
                ),
                (
                    args["from_lng"]
                    if isinstance(args, dict) and "from_lng" in args
                    else args.from_lng
                ),
                (
                    args["to_lat"]
                    if isinstance(args, dict) and "to_lat" in args
                    else args.to_lat
                ),
                (
                    args["to_lng"]
                    if isinstance(args, dict) and "to_lng" in args
                    else args.to_lng
                ),
                params,
                scenario=args.scenario,
            )
        else:
            feature = GRAPH_MANAGER.astar_route(
                (
                    args["from_lat"]
                    if isinstance(args, dict) and "from_lat" in args
                    else args.from_lat
                ),
                (
                    args["from_lng"]
                    if isinstance(args, dict) and "from_lng" in args
                    else args.from_lng
                ),
                (
                    args["to_lat"]
                    if isinstance(args, dict) and "to_lat" in args
                    else args.to_lat
                ),
                (
                    args["to_lng"]
                    if isinstance(args, dict) and "to_lng" in args
                    else args.to_lng
                ),
                params,
            )

        print(json.dumps(feature, indent=2))
    except Exception as e:
        # Print a concise error to stdout so users see output
        print(json.dumps({"error": str(e)}, indent=2))
