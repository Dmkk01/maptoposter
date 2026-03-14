from matplotlib.figure import Figure
from networkx import MultiDiGraph
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
import sys
from datetime import datetime
import argparse
import pickle
import asyncio
from pathlib import Path
from hashlib import md5
from typing import cast
from geopandas import GeoDataFrame
import pickle
from shapely.geometry import Point

class CacheError(Exception):
    """Raised when a cache operation fails."""
    pass

CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)


THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

CACHE_DIR = ".cache"

def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}")


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}")


def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    
    # Verify fonts exist
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    
    return fonts

FONTS = load_fonts()

def generate_output_filename(city, theme_name, output_format, output_path=None):
    """
    Generate unique output filename with city, theme, and datetime.
    Can accept either a directory path or a full file path.
    """
    if output_path is None:
        output_dir = POSTERS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        city_slug = city.lower().replace(' ', '_')
        ext = output_format.lower()
        filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
        full_path = os.path.join(output_dir, filename)
    else:
        # Check if output_path has a file extension (is a file path)
        if os.path.splitext(output_path)[1]:  # Has extension, treat as full file path
            full_path = output_path
            output_dir = os.path.dirname(output_path)
        else:  # No extension, treat as directory
            output_dir = output_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            city_slug = city.lower().replace(' ', '_')
            ext = output_format.lower()
            filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
            full_path = os.path.join(output_dir, filename)
    
    # Create directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return full_path

def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        # Fallback to embedded default theme
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    
    with open(theme_file, 'r') as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return theme

# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def get_edge_colors_by_type(G):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []
    
    for u, v, data in G.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')
        
        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign color based on road type
        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']
        
        edge_colors.append(color)
    
    return edge_colors

def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign width based on road importance
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        
        edge_widths.append(width)
    
    return edge_widths

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}")

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError("Geocoder returned a coroutine while an event loop is already running. Run this script in a synchronous environment.")
            location = loop.run_until_complete(location)
    
    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")
    
def get_crop_limits(target_crs, center_lat_lon, fig, dist):
    """
    Crop inward to preserve aspect ratio while guaranteeing
    full coverage of the requested radius.
    """
    lat, lon = center_lat_lon

    # Project center point into graph CRS
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=target_crs
        )[0]
    )
    center_point = cast(Point, center)
    center_x, center_y = center_point.x, center_point.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    # Start from the *requested* radius
    half_x = dist
    half_y = dist

    # Cut inward to match aspect
    if aspect > 1:  # landscape → reduce height
        half_y = half_x / aspect
    else:           # portrait → reduce width
        half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def get_road_query_options(dist: float, natural_mode: bool, natural_detail: str) -> tuple[str, str | None]:
    """Return adaptive road query settings for the requested scale."""
    if dist >= 140000:
        return (
            'drive',
            '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]',
        )

    if dist >= 70000:
        return (
            'drive',
            '["highway"~"motorway|trunk|primary|secondary|motorway_link|trunk_link|primary_link|secondary_link"]',
        )

    if not natural_mode:
        return 'all', None

    if dist >= 180000 or natural_detail == 'low':
        custom_filter = (
            '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]'
        )
    elif dist >= 70000:
        custom_filter = (
            '["highway"~"motorway|trunk|primary|secondary|motorway_link|trunk_link|primary_link|secondary_link"]'
        )
    elif dist >= 25000:
        custom_filter = (
            '["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
        )
    else:
        custom_filter = (
            '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
        )

    return 'drive', custom_filter


def get_feature_query_dist(dist: float, query_dist: float) -> float:
    """Reduce non-road feature downloads for large-area renders."""
    if dist >= 140000:
        return query_dist * 0.4
    if dist >= 70000:
        return query_dist * 0.6
    return query_dist


def prune_features_for_scale(data: GeoDataFrame | None, dist: float) -> GeoDataFrame | None:
    """Drop tiny polygons and short lines that add clutter at large scales."""
    if data is None or data.empty:
        return data

    try:
        projected = ox.projection.project_gdf(data)
    except Exception:
        return data

    min_area = max(2500.0, (dist / 80.0) ** 2)
    min_length = max(150.0, dist / 40.0)

    geom_types = projected.geometry.geom_type
    polygon_mask = geom_types.isin(['Polygon', 'MultiPolygon'])
    line_mask = geom_types.isin(['LineString', 'MultiLineString'])
    other_mask = ~(polygon_mask | line_mask)

    keep_mask = other_mask.copy()
    keep_mask.loc[polygon_mask] = projected.loc[polygon_mask].geometry.area >= min_area
    keep_mask.loc[line_mask] = projected.loc[line_mask].geometry.length >= min_length

    return cast(GeoDataFrame, data.loc[keep_mask])


def fetch_graph(point, dist, network_type='all', custom_filter=None, cache_variant='default') -> MultiDiGraph | None:
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}_{network_type}_{cache_variant}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        G = ox.graph_from_point(
            point,
            dist=dist,
            dist_type='bbox',
            network_type=network_type,
            custom_filter=custom_filter,
            truncate_by_edge=True,
        )
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, G)
        except CacheError as e:
            print(e)
        return G
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None

def fetch_graph_from_place(place_name) -> tuple[MultiDiGraph | None, tuple[float, float] | None]:
    """
    Fetch graph for a large area (country, state, region) using place name.
    Only fetches major roads for performance.
    Returns tuple of (graph, center_point) or (None, None) if failed.
    """
    graph = f"graph_place_{place_name.lower().replace(' ', '_').replace(',', '_')}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached place network")
        # Cached data is tuple of (graph, center_point)
        return cast(tuple[MultiDiGraph, tuple[float, float]], cached)

    try:
        print(f"Downloading map data for: {place_name}")
        
        # Set a much larger max query area for place-based queries
        # Default is 50000000 (50 sq km), we'll increase to 500000000000 (500000 sq km)
        ox.settings.max_query_area_size = 500000000000
        
        # For large areas, include enough roads to show structure but not overwhelm
        print("⚠ Note: Fetching major roads (motorway, trunk, primary, secondary, tertiary)...")
        print("   This may take 2-5 minutes for countries. Please be patient.")
        
        # Include roads down to tertiary to show city structure and contours
        custom_filter = (
            '["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link"]'
        )
        
        G = ox.graph_from_place(
            place_name, 
            network_type='drive',  # Drive network (excludes pedestrian/bike-only)
            custom_filter=custom_filter,  # Major roads plus tertiary for structure
            truncate_by_edge=True,
            simplify=True  # Simplify the graph to reduce nodes
        )
        
        # Calculate center point from the graph's bounding box
        nodes = ox.graph_to_gdfs(G, edges=False)
        center_lat = (nodes['y'].min() + nodes['y'].max()) / 2
        center_lon = (nodes['x'].min() + nodes['x'].max()) / 2
        center_point = (center_lat, center_lon)
        
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, (G, center_point))
        except CacheError as e:
            print(e)
        return G, center_point
    except Exception as e:
        print(f"OSMnx error while fetching place graph: {e}")
        print("\nTroubleshooting:")
        print("  - Very large countries may still timeout. Try a smaller region instead.")
        print("  - Make sure the place name is recognizable by OpenStreetMap")
        print("  - Examples that work well: 'Singapore', 'Belgium', 'Malta', 'Manhattan, New York'")
        print("  - For huge countries, try a state/province: 'California, USA', 'Bavaria, Germany'")
        return None, None

def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None

def fetch_features_from_place(place_name, tags, name) -> GeoDataFrame | None:
    """
    Fetch geographic features (water, parks) for a large area using place name.
    Only fetches major features to improve performance.
    """
    tag_str = "_".join(tags.keys())
    features = f"{name}_place_{place_name.lower().replace(' ', '_').replace(',', '_')}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cached

    try:
        # For large areas, limit the features to improve performance
        # We can skip this entirely for very large areas as the details won't be visible anyway
        print(f"⚠ Skipping {name} features for large area (not visible at this scale)")
        return None
    except Exception as e:
        print(f"OSMnx error while fetching {name} features: {e}")
        return None

def fetch_natural_features(point, dist, feature_type='forests', scale_dist=None) -> GeoDataFrame | None:
    """
    Fetch natural features like forests, grasslands, meadows, streams, trails,
    and rugged terrain polygons.
    Useful for natural/terrain-focused maps.
    """
    lat, lon = point
    effective_scale = scale_dist if scale_dist is not None else dist
    feature_variant = f"{feature_type}_{int(dist)}"
    features = f"natural_{feature_variant}_{lat}_{lon}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {feature_type}")
        return cast(GeoDataFrame, cached)

    try:
        # Define tags based on feature type
        tags: dict[str, bool | str | list[str]]
        if feature_type == 'forests':
            tags = {'natural': 'wood', 'landuse': 'forest'}
        elif feature_type == 'grasslands':
            if effective_scale >= 120000:
                tags = {'natural': ['grassland', 'heath'], 'landuse': 'meadow'}
            else:
                tags = {'natural': ['grassland', 'heath', 'scrub'], 'landuse': ['meadow', 'grass']}
        elif feature_type == 'water_detailed':
            if effective_scale >= 100000:
                tags = {'waterway': ['river', 'canal', 'riverbank'], 'natural': 'water'}
            else:
                tags = {'natural': 'water', 'waterway': ['river', 'stream', 'canal', 'riverbank']}
        elif feature_type == 'trails':
            if effective_scale >= 35000:
                print("⚠ Skipping trails at this scale")
                return None
            if effective_scale >= 18000:
                tags = {'highway': ['path', 'track']}
            else:
                tags = {'highway': ['path', 'track', 'footway', 'bridleway', 'cycleway', 'steps']}
        elif feature_type == 'rugged':
            if effective_scale >= 140000:
                print("⚠ Skipping rugged terrain at this scale")
                return None
            if effective_scale >= 50000:
                tags = {'natural': ['wetland', 'bare_rock', 'scree']}
            else:
                tags = {'natural': ['wetland', 'bare_rock', 'scree', 'shingle', 'sand', 'beach']}
        else:
            return None
        
        data = ox.features_from_point(point, tags=tags, dist=dist)
        data = prune_features_for_scale(cast(GeoDataFrame, data), effective_scale)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching {feature_type}: {e}")
        return None


def get_natural_style(detail_level: str) -> dict[str, float | bool]:
    """Return rendering knobs for natural mode."""
    styles: dict[str, dict[str, float | bool]] = {
        'low': {
            'road_alpha': 0.45,
            'road_width_scale': 0.7,
            'show_trails': False,
            'show_rugged': False,
            'forest_alpha': 0.55,
            'grass_alpha': 0.45,
        },
        'medium': {
            'road_alpha': 0.28,
            'road_width_scale': 0.5,
            'show_trails': True,
            'show_rugged': False,
            'forest_alpha': 0.68,
            'grass_alpha': 0.55,
        },
        'high': {
            'road_alpha': 0.18,
            'road_width_scale': 0.35,
            'show_trails': True,
            'show_rugged': True,
            'forest_alpha': 0.78,
            'grass_alpha': 0.62,
        },
    }
    return styles.get(detail_level, styles['medium'])



def create_poster(city, country, point, dist, output_file, output_format, width=12, height=16, country_label=None, name_label=None, place_name=None, natural_mode=False, natural_detail='medium', include_text=True):
    """
    Create a map poster. Can use either:
    - point + dist for radius-based maps
    - place_name for bounding box maps (countries, regions)
    - natural_mode: emphasizes natural features (forests, water, terrain) instead of roads
    """
    if place_name:
        print(f"\nGenerating map for {place_name}...")
    else:
        print(f"\nGenerating map for {city}, {country}...")
    
    if natural_mode:
        print("🌲 Natural mode enabled - emphasizing forests, water, and terrain")
        print(f"   Detail level: {natural_detail}")
    if not place_name and dist >= 70000:
        print("⚠ Large-area mode enabled - reducing downloaded detail for faster fetches")

    natural_style = get_natural_style(natural_detail)
    query_dist = dist
    feature_query_dist = dist
    graph_fetch_failed = False
    
    # Progress bar for data fetching
    extra_steps = 0
    if natural_mode and not place_name:
        extra_steps = 1
        if dist < 70000:
            extra_steps = 3
            if natural_style['show_trails']:
                extra_steps += 1
            if natural_style['show_rugged']:
                extra_steps += 1

    total_steps = 3 + extra_steps
    with tqdm(total=total_steps, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        
        if place_name:
            # Use place-based fetching for large areas
            G, center_point = fetch_graph_from_place(place_name)
            if G is None:
                raise RuntimeError("Failed to retrieve street network data for place.")
            # Override point with calculated center
            point = center_point
        else:
            # Use point + radius for smaller areas
            query_dist = dist * (max(height, width) / min(height, width)) / 4
            feature_query_dist = get_feature_query_dist(dist, query_dist)
            network_type, custom_filter = get_road_query_options(dist, natural_mode, natural_detail)
            cache_variant = md5(f"{network_type}|{custom_filter}".encode('utf-8')).hexdigest()[:10]
            G = fetch_graph(point, query_dist, network_type=network_type, custom_filter=custom_filter, cache_variant=cache_variant)
            if G is None:
                if natural_mode:
                    print("⚠ No street network found in the requested area; continuing with natural features only")
                    graph_fetch_failed = True
                else:
                    raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)
        
        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        if place_name:
            water = fetch_features_from_place(place_name, tags={'natural': 'water', 'waterway': 'riverbank'}, name='water')
        else:
            if dist >= 70000:
                water_tags = {'natural': 'water'}
            else:
                water_tags = {'natural': 'water', 'waterway': 'riverbank'}
            water = fetch_features(point, feature_query_dist, tags=water_tags, name='water')
            if natural_mode:
                water = prune_features_for_scale(water, dist)
        pbar.update(1)
        
        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        if place_name:
            parks = fetch_features_from_place(place_name, tags={'leisure': 'park', 'landuse': 'grass'}, name='parks')
        else:
            park_tags: dict[str, bool | str | list[str]]
            if dist >= 140000 and not natural_mode:
                print("⚠ Skipping parks at this scale")
                parks = None
            elif natural_mode and dist >= 100000:
                park_tags = {'leisure': 'park'}
                parks = fetch_features(point, feature_query_dist, tags=park_tags, name='parks')
            elif dist >= 70000:
                park_tags = {'leisure': 'park'}
                parks = fetch_features(point, feature_query_dist, tags=park_tags, name='parks')
            else:
                park_tags = {'leisure': 'park', 'landuse': 'grass'}
                parks = fetch_features(point, feature_query_dist, tags=park_tags, name='parks')
            if natural_mode:
                parks = prune_features_for_scale(parks, dist)
        pbar.update(1)
        
        # Additional natural features if in natural mode
        forests = None
        grasslands = None
        water_detailed = None
        trails = None
        rugged = None
        if natural_mode and not place_name:  # Only for point-based maps
            # 4. Fetch Forests
            pbar.set_description("Downloading forests/woods")
            forests = fetch_natural_features(point, feature_query_dist, 'forests', scale_dist=dist)
            pbar.update(1)

            if dist < 70000:
                # 5. Fetch Grasslands/Meadows
                pbar.set_description("Downloading grasslands/meadows")
                grasslands = fetch_natural_features(point, feature_query_dist, 'grasslands', scale_dist=dist)
                pbar.update(1)

                # 6. Fetch Detailed Water (rivers, streams)
                pbar.set_description("Downloading rivers/streams")
                water_detailed = fetch_natural_features(point, feature_query_dist, 'water_detailed', scale_dist=dist)
                pbar.update(1)

                if natural_style['show_trails']:
                    pbar.set_description("Downloading trails/paths")
                    trails = fetch_natural_features(point, feature_query_dist, 'trails', scale_dist=dist)
                    pbar.update(1)

                if natural_style['show_rugged']:
                    pbar.set_description("Downloading rugged terrain")
                    rugged = fetch_natural_features(point, feature_query_dist, 'rugged', scale_dist=dist)
                    pbar.update(1)
    
    print("✓ All data retrieved successfully!")
    
    # 2. Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    if point is None:
        raise RuntimeError("Failed to determine map center point.")

    lat, lon = point
    projected_center, target_crs = ox.projection.project_geometry(Point(lon, lat), crs="EPSG:4326")
    center_point = cast(Point, projected_center)

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    G_proj = ox.project_graph(cast(MultiDiGraph, G)) if G is not None else None
    if G_proj is not None:
        target_crs = G_proj.graph['crs']
    
    # 3. Plot Layers
    # Layer 1: Polygons (filter to only plot polygon/multipolygon geometries, not points)
    if water is not None and not water.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            # Project water features in the same CRS as the graph
            try:
                water_polys = ox.projection.project_gdf(water_polys, to_crs=target_crs)
            except Exception:
                water_polys = water_polys.to_crs(target_crs)
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    
    # Natural mode: add forests, grasslands, and detailed water
    if natural_mode:
        # Grasslands (lighter green, behind forests)
        if grasslands is not None and not grasslands.empty:
            grass_polys = grasslands[grasslands.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not grass_polys.empty:
                try:
                    grass_polys = ox.projection.project_gdf(grass_polys, to_crs=target_crs)
                except Exception:
                    grass_polys = grass_polys.to_crs(target_crs)
                grassland_color = THEME.get('grasslands', '#E8F4D0')
                grass_polys.plot(ax=ax, facecolor=grassland_color, edgecolor='none', alpha=cast(float, natural_style['grass_alpha']), zorder=2)
        
        # Forests (darker green)
        if forests is not None and not forests.empty:
            forest_polys = forests[forests.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not forest_polys.empty:
                try:
                    forest_polys = ox.projection.project_gdf(forest_polys, to_crs=target_crs)
                except Exception:
                    forest_polys = forest_polys.to_crs(target_crs)
                forest_color = THEME.get('forests', '#6B8E6B')
                forest_polys.plot(ax=ax, facecolor=forest_color, edgecolor='none', alpha=cast(float, natural_style['forest_alpha']), zorder=3)

        if rugged is not None and not rugged.empty:
            rugged_polys = rugged[rugged.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not rugged_polys.empty:
                try:
                    rugged_polys = ox.projection.project_gdf(rugged_polys, to_crs=target_crs)
                except Exception:
                    rugged_polys = rugged_polys.to_crs(target_crs)
                rugged_color = THEME.get('rugged', '#CDBFA5')
                rugged_polys.plot(ax=ax, facecolor=rugged_color, edgecolor='none', alpha=0.4, zorder=3)
        
        # Detailed water (rivers, streams) - as lines
        if water_detailed is not None and not water_detailed.empty:
            # Handle both polygons and lines for water features
            water_lines = water_detailed[water_detailed.geometry.type.isin(['LineString', 'MultiLineString'])]
            water_polys_detailed = water_detailed[water_detailed.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            if not water_lines.empty:
                try:
                    water_lines = ox.projection.project_gdf(water_lines, to_crs=target_crs)
                except Exception:
                    water_lines = water_lines.to_crs(target_crs)
                water_lines.plot(ax=ax, edgecolor=THEME['water'], linewidth=1.5, zorder=4)
            
            if not water_polys_detailed.empty:
                try:
                    water_polys_detailed = ox.projection.project_gdf(water_polys_detailed, to_crs=target_crs)
                except Exception:
                    water_polys_detailed = water_polys_detailed.to_crs(target_crs)
                water_polys_detailed.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=4)

        if trails is not None and not trails.empty:
            trail_lines = trails[trails.geometry.type.isin(['LineString', 'MultiLineString'])]
            if not trail_lines.empty:
                try:
                    trail_lines = ox.projection.project_gdf(trail_lines, to_crs=target_crs)
                except Exception:
                    trail_lines = trail_lines.to_crs(target_crs)
                trail_color = THEME.get('trails', THEME['text'])
                trail_lines.plot(ax=ax, color=trail_color, linewidth=0.7, alpha=0.45, zorder=6)
    
    if parks is not None and not parks.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            # Project park features in the same CRS as the graph
            try:
                parks_polys = ox.projection.project_gdf(parks_polys, to_crs=target_crs)
            except Exception:
                parks_polys = parks_polys.to_crs(target_crs)
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=5)
    
    # Layer 2: Roads with hierarchy coloring
    # In natural mode, make roads lighter and thinner
    edge_colors = None
    edge_widths = None
    if G_proj is not None:
        print("Applying road hierarchy colors...")
        edge_colors = get_edge_colors_by_type(G_proj)
        edge_widths = get_edge_widths_by_type(G_proj)
        
        # Adjust roads for natural mode
        if natural_mode:
            road_alpha = cast(float, natural_style['road_alpha'])
            road_width_scale = cast(float, natural_style['road_width_scale'])
            edge_colors = [mcolors.to_hex(mcolors.to_rgba(c, alpha=road_alpha), keep_alpha=True) for c in edge_colors]
            edge_widths = [w * road_width_scale for w in edge_widths]

    # Determine cropping limits to maintain the poster aspect ratio
    if place_name:
        # For place-based maps, adjust bounding box to match figure aspect ratio
        assert G_proj is not None
        nodes = ox.graph_to_gdfs(G_proj, edges=False)
        data_x_min, data_x_max = nodes.geometry.x.min(), nodes.geometry.x.max()
        data_y_min, data_y_max = nodes.geometry.y.min(), nodes.geometry.y.max()
        
        # Calculate the data's dimensions
        data_width = data_x_max - data_x_min
        data_height = data_y_max - data_y_min
        data_aspect = data_width / data_height
        
        # Figure aspect ratio
        fig_aspect = width / height
        
        # Adjust limits to match figure aspect while containing all data
        if data_aspect > fig_aspect:
            # Data is wider than figure - expand height
            data_center_y = (data_y_min + data_y_max) / 2
            new_height = data_width / fig_aspect
            crop_xlim = (data_x_min, data_x_max)
            crop_ylim = (data_center_y - new_height/2, data_center_y + new_height/2)
        else:
            # Data is taller than figure - expand width
            data_center_x = (data_x_min + data_x_max) / 2
            new_width = data_height * fig_aspect
            crop_xlim = (data_center_x - new_width/2, data_center_x + new_width/2)
            crop_ylim = (data_y_min, data_y_max)
    else:
        # For point-based maps, use the compensated distance
        crop_xlim, crop_ylim = get_crop_limits(target_crs, point, fig, query_dist)
    # Plot the projected graph and then apply the cropped limits
    if G_proj is not None:
        assert edge_colors is not None
        assert edge_widths is not None
        ox.plot_graph(
            G_proj, ax=ax, bgcolor=THEME['bg'],
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=edge_widths,
            show=False, close=False
        )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)
    ax.set_axis_off()
    
    if include_text:
        # Layer 3: Gradients (Top and Bottom)
        create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
        create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

        # Calculate scale factor based on poster width (reference width 12 inches)
        scale_factor = width / 12.0

        # Base font sizes (at 12 inches width)
        BASE_MAIN = 60
        BASE_TOP = 40
        BASE_SUB = 22
        BASE_COORDS = 14
        BASE_ATTR = 8

        # 4. Typography using Roboto font
        if FONTS:
            font_main = FontProperties(fname=FONTS['bold'], size=BASE_MAIN * scale_factor)
            font_top = FontProperties(fname=FONTS['bold'], size=BASE_TOP * scale_factor)
            font_sub = FontProperties(fname=FONTS['light'], size=BASE_SUB * scale_factor)
            font_coords = FontProperties(fname=FONTS['regular'], size=BASE_COORDS * scale_factor)
            font_attr = FontProperties(fname=FONTS['light'], size=BASE_ATTR * scale_factor)
        else:
            # Fallback to system fonts
            font_main = FontProperties(family='monospace', weight='bold', size=BASE_MAIN * scale_factor)
            font_top = FontProperties(family='monospace', weight='bold', size=BASE_TOP * scale_factor)
            font_sub = FontProperties(family='monospace', weight='normal', size=BASE_SUB * scale_factor)
            font_coords = FontProperties(family='monospace', size=BASE_COORDS * scale_factor)
            font_attr = FontProperties(family='monospace', size=BASE_ATTR * scale_factor)

        spaced_city = "  ".join(list(city.upper()))

        # Dynamically adjust font size based on city name length to prevent truncation.
        base_adjusted_main = BASE_MAIN * scale_factor
        city_char_count = len(city)

        if city_char_count > 10:
            length_factor = 10 / city_char_count
            adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
        else:
            adjusted_font_size = base_adjusted_main

        if FONTS:
            font_main_adjusted = FontProperties(fname=FONTS['bold'], size=adjusted_font_size)
        else:
            font_main_adjusted = FontProperties(family='monospace', weight='bold', size=adjusted_font_size)

        # --- BOTTOM TEXT ---
        ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
                color=THEME['text'], ha='center', fontproperties=font_main_adjusted, zorder=11)

        country_text = country_label if country_label is not None else country
        ax.text(0.5, 0.10, country_text.upper(), transform=ax.transAxes,
                color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)

        coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
        if lon < 0:
            coords = coords.replace("E", "W")

        ax.text(0.5, 0.07, coords, transform=ax.transAxes,
                color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)

        ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
                color=THEME['text'], linewidth=1 * scale_factor, zorder=11)

        # --- ATTRIBUTION (bottom right) ---
        if FONTS:
            font_attr = FontProperties(fname=FONTS['light'], size=8)
        else:
            font_attr = FontProperties(family='monospace', size=8)

        ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
                color=THEME['text'], alpha=0.5, ha='right', va='bottom',
                fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0,)

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    
    # Print the absolute path for programmatic capture
    print(f"OUTPUT_FILE: {os.path.abspath(output_file)}")
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]
  python create_map_poster.py --lat <latitude> --lon <longitude> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # Using custom coordinates
  python create_map_poster.py --lat 52.3676 --lon 4.9041 --city-label "Amsterdam" --country-label "Netherlands" -t japanese_ink -d 15000
  python create_map_poster.py --lat 40.7580 --lon -73.9855 --city-label "Times Square" -t neon_cyberpunk -d 5000
  python create_map_poster.py --lat 51.5074 --lon -0.1278 -t noir -d 10000  # Will show "CUSTOM LOCATION"
    python create_map_poster.py --lat 44.1640832 --lon 6.1878515 --city-label "Alpes-de-Haute-Provence" --country-label "France" -t forest -d 25000 --natural --natural-detail high
  
  # Countries, states, and large areas (uses bounding box, ignores --distance)
  # Best for small-medium countries and regions:
  python create_map_poster.py --place "Singapore" -t ocean              # Small country (works great!)
  python create_map_poster.py -p "Belgium" -t contrast_zones            # Medium country
  python create_map_poster.py -p "Manhattan, New York" -t noir          # Borough
  python create_map_poster.py -p "Île-de-France, France" -t pastel_dream  # Region
  python create_map_poster.py -p "Netherlands" --city-label "Nederland" -t blueprint  # Larger country (may take 2-3 min)
  
  # Using custom output path
    python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t japanese_ink -o "/path/to/output/amsterdam.png"
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -o "./my_maps/"  # Directory only

    # Text-free export
    python create_map_poster.py -c "Zurich" -C "Switzerland" -t blueprint --no-text
  
  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (use with --country, or omit if using --lat/--lon or --place)
  --country, -C     Country name (use with --city, or omit if using --lat/--lon or --place)
  --place, -p       Place name for large areas: countries, states, regions, boroughs
                    (alternative to --city/--country or --lat/--lon, uses bounding box)
  --lat             Latitude coordinate (use with --lon instead of --city/--country)
  --lon             Longitude coordinate (use with --lat instead of --city/--country)
  --city-label      Override city text displayed on poster
  --country-label   Override country text displayed on poster
    --no-text         Hide all text overlays and gradient title bands
  --theme, -t       Theme name (default: feature_based)
  --all-themes      Generate posters for all themes
  --distance, -d    Map radius in meters (default: 29000, ignored when using --place)
    --natural, -n     Emphasize forests, water, grasslands, trails instead of roads
    --natural-detail  Natural mode strength: low, medium, high
  --output, -o      Output path: directory or full file path
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--place', '-p', type=str, help='Place name for large areas (e.g., "Netherlands", "Manhattan, New York") - alternative to city/country or lat/lon')
    parser.add_argument('--lat', type=float, help='Latitude (use with --lon instead of --city/--country)')
    parser.add_argument('--lon', type=float, help='Longitude (use with --lat instead of --city/--country)')
    parser.add_argument('--city-label', dest='city_label', type=str, help='Override city text displayed on poster')
    parser.add_argument('--country-label', dest='country_label', type=str, help='Override country text displayed on poster')
    parser.add_argument('--no-text', action='store_true', help='Hide all text overlays and gradient title bands')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--all-themes', '--All-themes', dest='all_themes', action='store_true', help='Generate posters for all themes')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--width', '-W', type=float, default=12, help='Image width in inches (default: 12)')
    parser.add_argument('--height', '-H', type=float, default=16, help='Image height in inches (default: 16)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'],help='Output format for the poster (default: png)')
    parser.add_argument('--output', '-o', type=str, help='Output path: directory (e.g., ./maps/) or full file path (e.g., ./maps/my_map.png)')
    parser.add_argument('--natural', '-n', action='store_true', help='Natural/terrain mode: emphasize forests, water, grasslands instead of roads (ideal for national parks and natural regions)')
    parser.add_argument('--natural-detail', choices=['low', 'medium', 'high'], default='medium', help='How strongly natural mode emphasizes terrain features (default: medium)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)
    
    # Validate required arguments: either (city + country) OR (lat + lon) OR (place)
    has_city_country = bool(args.city and args.country)
    has_lat_lon = bool(args.lat is not None and args.lon is not None)
    has_place = bool(args.place is not None)
    
    if not has_city_country and not has_lat_lon and not has_place:
        print("Error: Either (--city and --country) OR (--lat and --lon) OR (--place) are required.\n")
        print_examples()
        sys.exit(1)
    
    if sum([has_city_country, has_lat_lon, has_place]) > 1:
        print("Error: Use only ONE of: (--city + --country), (--lat + --lon), or (--place).\n")
        print_examples()
        sys.exit(1)
    
    if has_lat_lon and (args.lat < -90 or args.lat > 90):
        print("Error: Latitude must be between -90 and 90.")
        sys.exit(1)
    
    if has_lat_lon and (args.lon < -180 or args.lon > 180):
        print("Error: Longitude must be between -180 and 180.")
        sys.exit(1)
    
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]
    
    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)
    
    # Get coordinates and generate poster
    try:
        # Determine which mode we're in
        if args.place:
            # Place-based mode (for countries, regions, large areas)
            coords = None  # Will be calculated from the place
            place_name = args.place
            display_city = args.city_label or args.place
            display_country = args.country_label or ""
            filename_base = args.place.replace(",", "").replace(" ", "_").lower()
        elif args.lat is not None and args.lon is not None:
            # Custom coordinates mode
            coords = (args.lat, args.lon)
            place_name = None
            print(f"✓ Using custom coordinates: {args.lat}, {args.lon}")
            display_city = args.city_label or args.city or "CUSTOM LOCATION"
            display_country = args.country_label or args.country or ""
            filename_base = args.city or "custom"
        else:
            # City/country geocoding mode
            coords = get_coordinates(args.city, args.country)
            place_name = None
            display_city = args.city_label or args.city
            display_country = args.country_label or args.country
            filename_base = args.city
        
        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(filename_base, theme_name, args.format, output_path=args.output)
            create_poster(display_city, display_country, coords, args.distance, output_file, args.format, args.width, args.height, country_label=args.country_label, place_name=place_name, natural_mode=args.natural, natural_detail=args.natural_detail, include_text=not args.no_text)
        
        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
