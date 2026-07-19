"""
political_algorithm.py
========================
Algoritmo Político Híbrido de Clusterização (K-Means Ponderado + Override Manual).

Responsabilidades:
- Clusterizar os municípios da Paraíba em setores eleitorais usando K-Means.
- Aplicar pesos políticos: eleitorado, alinhamento do prefeito, peso de liderança.
- Implementar o "Override Manual" (Human-in-the-Loop): cidades fixadas como polo
  são respeitadas e os demais clusters são recalculados ao redor delas.
- Identificar automaticamente a Cidade-Polo de cada setor.

Autor: Arquitetura MVP – Inteligência Logística Eleitoral
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de peso padrão
# ---------------------------------------------------------------------------
ESTRATEGIAS_PESOS = {
    "Geográfica (Padrão)": {
        "geo": 15.0,           
        "eleitorado": 0.5,     
        "alinhamento": 0.5,    
        "lideranca": 0.5,      
        "status_aliado": 0.5,  
    },
    "Equilíbrio de Eleitores": {
        "geo": 3.0,
        "eleitorado": 8.0,
        "alinhamento": 0.5,
        "lideranca": 0.5,
        "status_aliado": 0.5,
    },
    "Afinidade Política": {
        "geo": 4.0,
        "eleitorado": 0.5,
        "alinhamento": 5.0,
        "lideranca": 2.0,
        "status_aliado": 6.0,
    },
    "Faixas Longitudinais (Norte-Sul)": {
        "geo_lat": 0.5,       # Pouco peso na latitude (permite esticar norte-sul)
        "geo_lon": 25.0,      # Muito peso na longitude (fatia verticalmente)
        "eleitorado": 0.5,
        "alinhamento": 0.5,
        "lideranca": 0.5,
        "status_aliado": 0.5,
    }
}

STATUS_COLORS = {
    "Aliado": "#22c55e",       # Verde
    "Oposição": "#ef4444",     # Vermelho
    "Neutro": "#94a3b8",       # Cinza
    "Fixado (Polo)": "#f59e0b",# Âmbar
}


# ---------------------------------------------------------------------------
# Estrutura de dados de um Setor
# ---------------------------------------------------------------------------
@dataclass
class Setor:
    """Representa um setor eleitoral com sua cidade-polo e municípios satélites."""
    id_setor: int
    polo: Dict                   # Linha do GeoDataFrame da cidade-polo
    satelites: List[Dict]        # Demais municípios do setor
    total_eleitorado: int = 0
    total_municipios: int = 0
    cor_hex: str = "#3b82f6"     # Cor para renderização no mapa

    def __post_init__(self) -> None:
        self.total_municipios = 1 + len(self.satelites)
        self.total_eleitorado = (
            int(self.polo.get("eleitorado_total", 0))
            + sum(int(s.get("eleitorado_total", 0)) for s in self.satelites)
        )


# ---------------------------------------------------------------------------
# Paleta de cores para setores
# ---------------------------------------------------------------------------
_PALETA_SETORES = [
    "#3b82f6", "#8b5cf6", "#ec4899", "#f97316", "#eab308",
    "#22c55e", "#14b8a6", "#06b6d4", "#6366f1", "#f43f5e",
    "#84cc16", "#a855f7", "#0ea5e9", "#fb923c", "#4ade80",
    "#c084fc", "#38bdf8", "#fbbf24", "#f87171", "#34d399",
]


def _construir_feature_matrix(
    gdf: gpd.GeoDataFrame,
    pesos: Dict[str, float],
) -> np.ndarray:
    """
    Constrói a matriz de features ponderada para o K-Means.

    Colunas geradas (normalizadas 0-1 e multiplicadas pelo peso):
    - lat, lon (geográfico)
    - eleitorado_total normalizado
    - alinhamento_prefeito normalizado
    - peso_lideranca normalizado
    - bonus_aliado (1 se status_politico == "Aliado", else 0)

    Args:
        gdf: GeoDataFrame com os municípios.
        pesos: Dicionário de pesos por dimensão.

    Returns:
        Array numpy (n_municipios, n_features) normalizado e ponderado.
    """
    scaler = MinMaxScaler()

    # Colunas numéricas base
    lat = gdf["lat"].fillna(0).values.reshape(-1, 1)
    lon = gdf["lon"].fillna(0).values.reshape(-1, 1)
    eleitorado = gdf["eleitorado_total"].fillna(0).values.reshape(-1, 1)
    alinhamento = gdf["alinhamento_prefeito"].fillna(1).values.reshape(-1, 1)
    lideranca = gdf["peso_lideranca"].fillna(1).values.reshape(-1, 1)

    # Normalizar
    lat_n = scaler.fit_transform(lat)
    lon_n = scaler.fit_transform(lon)
    el_n = scaler.fit_transform(eleitorado)
    al_n = scaler.fit_transform(alinhamento)
    lid_n = scaler.fit_transform(lideranca)

    # Bonus por status aliado
    aliado = (gdf["status_politico"].fillna("Neutro") == "Aliado").astype(float).values.reshape(-1, 1)

    # Matriz ponderada: distância geográfica atrai, alto eleitorado/alinhamento repele clusters fracos
    X = np.hstack([
        lat_n * pesos.get("geo_lat", pesos.get("geo", 1.0)),
        lon_n * pesos.get("geo_lon", pesos.get("geo", 1.0)),
        el_n * pesos.get("eleitorado", 2.5),
        al_n * pesos.get("alinhamento", 1.8),
        lid_n * pesos.get("lideranca", 1.5),
        aliado * pesos.get("status_aliado", 2.0),
    ])
    return X


def _identificar_polo(grupo: gpd.GeoDataFrame) -> int:
    """
    Identifica o município-polo dentro de um grupo (setor).

    Critério composto:
    1. Prioridade absoluta: fixado_polo == True
    2. Score = eleitorado * 0.5 + alinhamento_prefeito * 0.3 + indice_infraestrutura * 0.2

    Returns:
        Índice (iloc) do polo no DataFrame.
    """
    # Verificar se há polo fixado manualmente
    fixados = grupo[grupo["fixado_polo"] == True]
    if not fixados.empty:
        return fixados.index[0]

    # Score ponderado
    g = grupo.copy()
    g["_el_n"] = g["eleitorado_total"] / (g["eleitorado_total"].max() + 1e-6)
    g["_al_n"] = g["alinhamento_prefeito"] / 5.0
    g["_infra_n"] = g["indice_infraestrutura"].fillna(0.5)
    g["_score"] = g["_el_n"] * 0.5 + g["_al_n"] * 0.3 + g["_infra_n"] * 0.2
    return g["_score"].idxmax()


def clusterizar_municipios(
    gdf: gpd.GeoDataFrame,
    n_setores: int = 15,
    estrategia: str = "Geográfica (Padrão)",
    polos_fixados: Optional[List[int]] = None,
    random_state: int = 42,
) -> Tuple[gpd.GeoDataFrame, List[Setor]]:
    """
    Algoritmo principal de clusterização K-Means ponderado com Override Manual.
    """
    pesos = ESTRATEGIAS_PESOS.get(estrategia, ESTRATEGIAS_PESOS["Geográfica (Padrão)"])

    if polos_fixados is None:
        polos_fixados = []

    gdf = gdf.copy()
    gdf["fixado_polo"] = gdf["cod_ibge"].isin(polos_fixados)
    gdf["setor"] = -1

    n_municipios = len(gdf)
    n_setores = min(n_setores, n_municipios)
    n_fixados = len(polos_fixados)

    logger.info(
        f"Clusterizando {n_municipios} municípios em {n_setores} setores "
        f"({n_fixados} polo(s) fixado(s))."
    )

    # ------------------------------------------------------------------
    # Passo 1: Alocar polos fixados como setores independentes
    # ------------------------------------------------------------------
    setor_id = 0
    for cod in polos_fixados:
        mask = gdf["cod_ibge"] == cod
        if mask.any():
            gdf.loc[mask, "setor"] = setor_id
            setor_id += 1

    n_setores_restantes = n_setores - n_fixados
    municipios_sem_setor = gdf[gdf["setor"] == -1].copy()

    if n_setores_restantes <= 0 or municipios_sem_setor.empty:
        # Todos os setores estão fixados; atribuir restantes ao polo mais próximo
        _atribuir_ao_polo_mais_proximo(gdf, polos_fixados)
    else:
        # ------------------------------------------------------------------
        # Passo 2: K-Means para municípios não fixados
        # ------------------------------------------------------------------
        X_total = _construir_feature_matrix(gdf, pesos)
        X_livres = X_total[gdf["setor"] == -1]

        # Inicialização inteligente: se há fixados, usá-los como centroides iniciais
        init_method: str | np.ndarray = "k-means++"
        if n_fixados > 0 and n_fixados < n_setores_restantes:
            fixed_indices = [gdf.index.get_loc(gdf[gdf["cod_ibge"] == c].index[0]) for c in polos_fixados if c in gdf["cod_ibge"].values]
            if fixed_indices:
                fixed_centers = X_total[fixed_indices]
                # Completar com k-means++ para os setores livres
                init_method = "k-means++"

        kmeans = KMeans(
            n_clusters=n_setores_restantes,
            init=init_method,
            n_init=15,
            max_iter=500,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(X_livres)

        # Ajustar labels para não conflitar com setores fixados
        gdf.loc[gdf["setor"] == -1, "setor"] = labels + setor_id

    # ------------------------------------------------------------------
    # Passo 3: Construir objetos Setor
    # ------------------------------------------------------------------
    setores: List[Setor] = []
    setores_ids = sorted(gdf["setor"].unique())

    for sid in setores_ids:
        grupo_mask = gdf["setor"] == sid
        grupo = gdf[grupo_mask]

        if grupo.empty:
            continue

        polo_idx = _identificar_polo(grupo)
        polo_row = gdf.loc[polo_idx].to_dict()
        satelites = [
            row.to_dict()
            for _, row in gdf[grupo_mask & (gdf.index != polo_idx)].iterrows()
        ]

        cor = _PALETA_SETORES[int(sid) % len(_PALETA_SETORES)]
        setor = Setor(
            id_setor=int(sid),
            polo=polo_row,
            satelites=satelites,
            cor_hex=cor,
        )
        setores.append(setor)

    logger.info(f"Clusterização concluída: {len(setores)} setores gerados.")
    return gdf, setores


def _atribuir_ao_polo_mais_proximo(
    gdf: gpd.GeoDataFrame,
    polos_fixados: List[int],
) -> None:
    """
    Para cada município sem setor, atribui ao polo fixado mais próximo (Haversine).
    Modifica gdf in-place.
    """
    from routing_engine import haversine_km

    polos_df = gdf[gdf["cod_ibge"].isin(polos_fixados)]
    sem_setor_mask = gdf["setor"] == -1

    for idx in gdf[sem_setor_mask].index:
        row = gdf.loc[idx]
        min_dist = float("inf")
        setor_mais_proximo = 0

        for _, polo in polos_df.iterrows():
            d = haversine_km(row["lat"], row["lon"], polo["lat"], polo["lon"])
            if d < min_dist:
                min_dist = d
                setor_mais_proximo = int(polo["setor"])

        gdf.at[idx, "setor"] = setor_mais_proximo


def filtrar_satelites_no_raio(
    polo_lat: float,
    polo_lon: float,
    satelites: List[Dict],
    raio_km: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filtra municípios satélites dentro e fora de um raio em KM.

    Args:
        polo_lat: Latitude da cidade-polo.
        polo_lon: Longitude da cidade-polo.
        satelites: Lista de dicts com 'lat' e 'lon'.
        raio_km: Raio de ação em quilômetros.

    Returns:
        Tupla (dentro_do_raio, fora_do_raio).
    """
    from routing_engine import haversine_km

    dentro: List[Dict] = []
    fora: List[Dict] = []

    for sat in satelites:
        dist = haversine_km(polo_lat, polo_lon, sat["lat"], sat["lon"])
        sat_copy = {**sat, "distancia_polo_km": round(dist, 2)}
        if dist <= raio_km:
            dentro.append(sat_copy)
        else:
            fora.append(sat_copy)

    dentro.sort(key=lambda x: x["distancia_polo_km"])
    return dentro, fora


def calcular_score_polo(municipio: Dict) -> float:
    """
    Calcula o score de relevância político-estratégica de um município.
    Usado para ordenar sugestões de polo na interface.

    Score = (eleitorado_n * 0.4) + (alinhamento_n * 0.25) +
            (lideranca_n * 0.20) + (aliado_bonus * 0.15)
    """
    el = float(municipio.get("eleitorado_total", 0))
    al = float(municipio.get("alinhamento_prefeito", 1)) / 5.0
    lid = float(municipio.get("peso_lideranca", 1)) / 5.0
    aliado = 1.0 if str(municipio.get("status_politico", "")) == "Aliado" else 0.0
    infra = float(municipio.get("indice_infraestrutura", 0.5))

    # Não normalizamos el aqui pois é relativo ao dataset; retornamos bruto para comparação
    return el * 0.4 + al * 0.25 + lid * 0.20 + aliado * 0.15 + infra * 0.10
