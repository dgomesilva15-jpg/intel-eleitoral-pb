"""
routing_engine.py
==================
Motor de Roteirização e Cálculo de Isócronas.

Responsabilidades:
- Integrar com a API do OpenRouteService (ORS).
- Calcular isócronas (polígonos de tempo de viagem).
- Calcular matrizes de distância/tempo entre pontos.
- Cache agressivo para minimizar chamadas à API.
- Fallback de distância Haversine quando ORS indisponível.

Autor: Arquitetura MVP – Inteligência Logística Eleitoral
"""

from __future__ import annotations

import json
import logging
import math
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ORS_BASE_URL = "https://api.openrouteservice.org"
ORS_ISOCHRONE_EP = f"{ORS_BASE_URL}/v2/isochrones/driving-car"
ORS_MATRIX_EP = f"{ORS_BASE_URL}/v2/matrix/driving-car"
ORS_DIRECTIONS_EP = f"{ORS_BASE_URL}/v2/directions/driving-car"

EARTH_RADIUS_KM = 6371.0
DEFAULT_TIMEOUT = 30  # segundos


# ---------------------------------------------------------------------------
# Modelo de resultado
# ---------------------------------------------------------------------------
class RouteResult:
    """Encapsula o resultado de um cálculo de rota."""

    def __init__(
        self,
        origem: str,
        destino: str,
        distancia_km: float,
        tempo_minutos: float,
        eleitores_destino: int,
        custo_beneficio: float,
        via_ors: bool = False,
        google_maps_url: str = "",
    ) -> None:
        self.origem = origem
        self.destino = destino
        self.distancia_km = round(distancia_km, 2)
        self.tempo_minutos = round(tempo_minutos, 1)
        self.eleitores_destino = eleitores_destino
        # KPI: eleitores por minuto de viagem
        self.custo_beneficio = round(custo_beneficio, 1)
        self.via_ors = via_ors
        self.google_maps_url = google_maps_url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Origem": self.origem,
            "Destino": self.destino,
            "Distância (km)": self.distancia_km,
            "Tempo Est. (min)": self.tempo_minutos,
            "Eleitores Destino": f"{self.eleitores_destino:,}",
            "Eleitores/Min": f"{self.custo_beneficio:.1f}",
            "Fonte": "ORS (Real)" if self.via_ors else "Haversine (Estimado)",
        }


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância em linha reta (km) entre dois pontos geográficos
    usando a fórmula de Haversine.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def haversine_tempo_estimado(distancia_km: float, velocidade_media_kmh: float = 65.0) -> float:
    """Estima o tempo em minutos com base na distância e velocidade média nordestina."""
    return (distancia_km / velocidade_media_kmh) * 60.0


def gerar_url_google_maps(pontos: List[Tuple[float, float]], nomes: List[str]) -> str:
    """
    Gera URL do Google Maps para uma rota com múltiplos pontos de parada.
    Abre no celular do motorista sem custo para a aplicação.

    Args:
        pontos: Lista de (lat, lon).
        nomes: Nomes legíveis dos pontos.

    Returns:
        URL formatada do Google Maps Directions.
    """
    if len(pontos) < 2:
        return ""

    origem_lat, origem_lon = pontos[0]
    destino_lat, destino_lon = pontos[-1]
    waypoints_str = ""
    if len(pontos) > 2:
        waypt = "|".join(f"{lat},{lon}" for lat, lon in pontos[1:-1])
        waypoints_str = f"&waypoints={waypt}"

    url = (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={origem_lat},{origem_lon}"
        f"&destination={destino_lat},{destino_lon}"
        f"{waypoints_str}"
        f"&travelmode=driving"
    )
    return url


# ---------------------------------------------------------------------------
# Cliente ORS
# ---------------------------------------------------------------------------
class ORSClient:
    """
    Cliente para a API do OpenRouteService com cache integrado.

    O cache usa lru_cache em chamadas normalizadas para garantir que
    requisições idênticas não sejam repetidas durante a sessão.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, application/geo+json",
        })
        self._isochrone_cache: Dict[str, Any] = {}
        self._matrix_cache: Dict[str, Any] = {}

    def _cache_key(self, *args: Any) -> str:
        return json.dumps(args, sort_keys=True)

    def get_isochrones(
        self,
        lat: float,
        lon: float,
        time_minutes: int,
        profile: str = "driving-car",
    ) -> Optional[Dict[str, Any]]:
        """
        Retorna o polígono GeoJSON de isócrona para um ponto central.

        Args:
            lat: Latitude do ponto central (cidade-polo).
            lon: Longitude do ponto central (cidade-polo).
            time_minutes: Tempo máximo de viagem em minutos.
            profile: Perfil de roteamento ORS.

        Returns:
            GeoJSON FeatureCollection ou None em caso de erro.
        """
        cache_key = self._cache_key(lat, lon, time_minutes, profile)
        if cache_key in self._isochrone_cache:
            logger.debug("Isócrona retornada do cache.")
            return self._isochrone_cache[cache_key]

        endpoint = f"{ORS_BASE_URL}/v2/isochrones/{profile}"
        payload = {
            "locations": [[lon, lat]],
            "range": [time_minutes * 60],  # ORS aceita segundos
            "range_type": "time",
            "smoothing": 0.75,
            "attributes": ["area", "reachfactor"],
        }

        try:
            resp = self.session.post(endpoint, json=payload, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            result = resp.json()
            self._isochrone_cache[cache_key] = result
            logger.info(f"Isócrona calculada para ({lat:.4f}, {lon:.4f}) – {time_minutes} min")
            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"ORS HTTP Error: {e.response.status_code} – {e.response.text[:200]}")
        except requests.exceptions.ConnectionError:
            logger.error("ORS: Sem conexão com a internet.")
        except Exception as e:
            logger.error(f"ORS Erro inesperado: {e}")

        return None

    def calculate_matrix(
        self,
        origins: List[Tuple[float, float]],
        destinations: List[Tuple[float, float]],
        metrics: List[str] = ["distance", "duration"],
    ) -> Optional[Dict[str, Any]]:
        """
        Calcula a matriz de distância e tempo entre origens e destinos.

        Args:
            origins: Lista de (lat, lon) para as origens.
            destinations: Lista de (lat, lon) para os destinos.
            metrics: Métricas a calcular ('distance', 'duration').

        Returns:
            Dict com chaves 'distances' e 'durations' (matrizes numpy).
        """
        cache_key = self._cache_key(origins, destinations, metrics)
        if cache_key in self._matrix_cache:
            logger.debug("Matriz retornada do cache.")
            return self._matrix_cache[cache_key]

        # ORS: [lon, lat]
        payload = {
            "locations": [[lon, lat] for lat, lon in (origins + destinations)],
            "sources": list(range(len(origins))),
            "destinations": list(range(len(origins), len(origins) + len(destinations))),
            "metrics": metrics,
            "units": "km",
        }

        try:
            resp = self.session.post(ORS_MATRIX_EP, json=payload, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            result = {
                "distances": np.array(data.get("distances", [])),
                "durations": np.array(data.get("durations", [])) / 60.0,  # → minutos
            }
            self._matrix_cache[cache_key] = result
            logger.info(f"Matriz {len(origins)}x{len(destinations)} calculada via ORS.")
            return result

        except Exception as e:
            logger.error(f"ORS Matrix Error: {e}")
            return None

    def get_directions(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém a rota detalhada entre múltiplos pontos de parada.

        Args:
            waypoints: Lista de (lat, lon) na ordem de visita.

        Returns:
            GeoJSON da rota ou None.
        """
        if len(waypoints) < 2:
            return None

        coords = [[lon, lat] for lat, lon in waypoints]
        payload = {"coordinates": coords, "format": "geojson"}

        try:
            resp = self.session.post(ORS_DIRECTIONS_EP, json=payload, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"ORS Directions Error: {e}")
            return None


# ---------------------------------------------------------------------------
# Facade pública – usada pelo app.py
# ---------------------------------------------------------------------------
def calcular_rota_custo_beneficio(
    polo_nome: str,
    polo_lat: float,
    polo_lon: float,
    cidades_alvo: List[Dict[str, Any]],
    ors_client: Optional[ORSClient] = None,
) -> List[RouteResult]:
    """
    Calcula o custo-benefício eleitoral de visitar um conjunto de cidades
    a partir de uma cidade-polo.

    Para cada cidade-alvo retorna:
    - Distância (km), Tempo estimado (min), Eleitores/Minuto.

    Args:
        polo_nome: Nome da cidade-polo de partida.
        polo_lat: Latitude da cidade-polo.
        polo_lon: Longitude da cidade-polo.
        cidades_alvo: Lista de dicts com 'municipio', 'lat', 'lon', 'eleitorado_total'.
        ors_client: Instância de ORSClient (opcional). Se None, usa Haversine.

    Returns:
        Lista ordenada (maior custo-benefício primeiro) de RouteResult.
    """
    resultados: List[RouteResult] = []

    origins = [(polo_lat, polo_lon)]
    destinations = [(c["lat"], c["lon"]) for c in cidades_alvo]

    matrix_result = None
    if ors_client:
        matrix_result = ors_client.calculate_matrix(origins, destinations)

    for i, cidade in enumerate(cidades_alvo):
        eleitores = int(cidade.get("eleitorado_total", 0))

        if matrix_result is not None:
            try:
                dist_km = float(matrix_result["distances"][0][i])
                tempo_min = float(matrix_result["durations"][0][i])
                via_ors = True
            except (IndexError, TypeError, ValueError):
                dist_km = haversine_km(polo_lat, polo_lon, cidade["lat"], cidade["lon"])
                tempo_min = haversine_tempo_estimado(dist_km)
                via_ors = False
        else:
            dist_km = haversine_km(polo_lat, polo_lon, cidade["lat"], cidade["lon"])
            tempo_min = haversine_tempo_estimado(dist_km)
            via_ors = False

        cb = eleitores / max(tempo_min, 1.0)

        google_url = gerar_url_google_maps(
            [(polo_lat, polo_lon), (cidade["lat"], cidade["lon"])],
            [polo_nome, cidade["municipio"]],
        )

        resultados.append(RouteResult(
            origem=polo_nome,
            destino=cidade["municipio"],
            distancia_km=dist_km,
            tempo_minutos=tempo_min,
            eleitores_destino=eleitores,
            custo_beneficio=cb,
            via_ors=via_ors,
            google_maps_url=google_url,
        ))

    resultados.sort(key=lambda r: r.custo_beneficio, reverse=True)
    return resultados


def calcular_isocronas_com_fallback(
    lat: float,
    lon: float,
    time_minutes: int,
    ors_client: Optional[ORSClient] = None,
) -> Optional[Dict[str, Any]]:
    """
    Tenta calcular isócronas via ORS. Se falhar, retorna um círculo
    aproximado (buffer) em formato GeoJSON como fallback.

    Args:
        lat: Latitude do ponto central.
        lon: Longitude do ponto central.
        time_minutes: Tempo em minutos para a isócrona.
        ors_client: Cliente ORS (opcional).

    Returns:
        GeoJSON FeatureCollection.
    """
    if ors_client:
        result = ors_client.get_isochrones(lat, lon, time_minutes)
        if result:
            return result

    # Fallback: círculo com raio proporcional (60 km/h média)
    logger.info("Usando fallback de círculo para isócrona.")
    raio_km = (time_minutes / 60.0) * 60.0  # km estimados
    raio_deg = raio_km / 111.32  # conversão grau aproximada

    num_pontos = 64
    angulos = np.linspace(0, 2 * math.pi, num_pontos)
    coords = [
        [lon + raio_deg * math.cos(a), lat + raio_deg * math.sin(a)]
        for a in angulos
    ]
    coords.append(coords[0])  # fechar o polígono

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "value": time_minutes * 60,
                "area": math.pi * raio_km ** 2,
                "reachfactor": 0.8,
                "fonte": "estimativa_circular",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            }
        }]
    }
