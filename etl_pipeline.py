"""
etl_pipeline.py
================
Pipeline ETL (Extract, Transform, Load) para a base de dados geoespaciais.

Responsabilidades:
- Carregar a malha municipal da Paraíba (via geobr ou GeoJSON local de fallback).
- Gerar/simular dados de Eleitorado (TSE) e Infraestrutura (IBGE).
- Produzir um GeoDataFrame unificado ("Base Mestra") que serve de
  esqueleto para o restante da aplicação.

Autor: Arquitetura MVP – Inteligência Logística Eleitoral
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PARAIBA_UF_CODE: int = 25          # Código IBGE da Paraíba
CACHE_PATH: Path = Path(".cache")   # Pasta local de cache
GEOJSON_FALLBACK: Path = Path("data/paraiba_municipios.geojson")

# ---------------------------------------------------------------------------
# Dados sintéticos – base simulando TSE + IBGE (223 municípios da PB)
# ---------------------------------------------------------------------------
_MUNICIPIOS_PB_SEED: list[dict] = [
    # Municípios reais da Paraíba com coordenadas aproximadas
    {"cod_ibge": 2500106, "municipio": "Água Branca",      "lat": -7.51,  "lon": -37.63},
    {"cod_ibge": 2500205, "municipio": "Aguiar",           "lat": -7.08,  "lon": -38.19},
    {"cod_ibge": 2500304, "municipio": "Alagoa Grande",    "lat": -7.05,  "lon": -35.61},
    {"cod_ibge": 2500403, "municipio": "Alagoa Nova",      "lat": -7.07,  "lon": -35.75},
    {"cod_ibge": 2500502, "municipio": "Alagoinha",        "lat": -6.95,  "lon": -35.54},
    {"cod_ibge": 2500536, "municipio": "Alcantil",         "lat": -7.57,  "lon": -36.28},
    {"cod_ibge": 2500577, "municipio": "Algodão de Jandaíra", "lat": -6.88, "lon": -36.11},
    {"cod_ibge": 2500601, "municipio": "Alhandra",         "lat": -7.43,  "lon": -34.91},
    {"cod_ibge": 2500734, "municipio": "Amparo",           "lat": -7.56,  "lon": -36.58},
    {"cod_ibge": 2500775, "municipio": "Aparecida",        "lat": -6.78,  "lon": -38.25},
    {"cod_ibge": 2500809, "municipio": "Araçagi",          "lat": -6.85,  "lon": -35.37},
    {"cod_ibge": 2500908, "municipio": "Arara",            "lat": -6.83,  "lon": -35.75},
    {"cod_ibge": 2501005, "municipio": "Araruna",          "lat": -6.53,  "lon": -35.74},
    {"cod_ibge": 2501104, "municipio": "Areia",            "lat": -6.96,  "lon": -35.70},
    {"cod_ibge": 2501153, "municipio": "Areia de Baraúnas", "lat": -6.65, "lon": -37.67},
    {"cod_ibge": 2501203, "municipio": "Areial",           "lat": -7.08,  "lon": -35.95},
    {"cod_ibge": 2501302, "municipio": "Aroeiras",         "lat": -7.55,  "lon": -35.71},
    {"cod_ibge": 2501351, "municipio": "Assunção",         "lat": -7.38,  "lon": -36.93},
    {"cod_ibge": 2501401, "municipio": "Baía da Traição",  "lat": -6.69,  "lon": -34.97},
    {"cod_ibge": 2501500, "municipio": "Bananeiras",       "lat": -6.75,  "lon": -35.63},
    {"cod_ibge": 2501534, "municipio": "Baraúna",          "lat": -6.62,  "lon": -37.63},
    {"cod_ibge": 2501575, "municipio": "Barra de Santana", "lat": -7.53,  "lon": -36.04},
    {"cod_ibge": 2501609, "municipio": "Barra de Santa Rosa", "lat": -6.72, "lon": -36.06},
    {"cod_ibge": 2501708, "municipio": "Barra de São Miguel", "lat": -7.75, "lon": -36.32},
    {"cod_ibge": 2501807, "municipio": "Bayeux",           "lat": -7.13,  "lon": -34.93},
    {"cod_ibge": 2501906, "municipio": "Belém",            "lat": -6.74,  "lon": -35.47},
    {"cod_ibge": 2502003, "municipio": "Belém do Brejo do Cruz", "lat": -6.19, "lon": -37.17},
    {"cod_ibge": 2502052, "municipio": "Bernardino Batista", "lat": -6.47, "lon": -38.31},
    {"cod_ibge": 2502102, "municipio": "Boa Ventura",      "lat": -7.68,  "lon": -38.17},
    {"cod_ibge": 2502151, "municipio": "Boa Vista",        "lat": -7.26,  "lon": -36.21},
    {"cod_ibge": 2502201, "municipio": "Bom Jesus",        "lat": -7.69,  "lon": -38.46},
    {"cod_ibge": 2502300, "municipio": "Bom Sucesso",      "lat": -7.56,  "lon": -38.05},
    {"cod_ibge": 2502409, "municipio": "Bonito de Santa Fé", "lat": -7.31, "lon": -38.52},
    {"cod_ibge": 2502508, "municipio": "Boqueirão",        "lat": -7.48,  "lon": -36.13},
    {"cod_ibge": 2502607, "municipio": "Borborema",        "lat": -6.80,  "lon": -35.62},
    {"cod_ibge": 2502706, "municipio": "Brejo do Cruz",    "lat": -6.34,  "lon": -37.00},
    {"cod_ibge": 2502805, "municipio": "Brejo dos Santos", "lat": -6.31,  "lon": -37.06},
    {"cod_ibge": 2502904, "municipio": "Caaporã",          "lat": -7.52,  "lon": -34.91},
    {"cod_ibge": 2503001, "municipio": "Cabaceiras",       "lat": -7.49,  "lon": -36.29},
    {"cod_ibge": 2503100, "municipio": "Cabedelo",         "lat": -6.98,  "lon": -34.83},
    {"cod_ibge": 2503209, "municipio": "Cachoeira dos Índios", "lat": -6.71, "lon": -38.52},
    {"cod_ibge": 2503308, "municipio": "Cacimba de Areia", "lat": -7.18,  "lon": -37.14},
    {"cod_ibge": 2503407, "municipio": "Cacimba de Dentro", "lat": -6.65, "lon": -35.78},
    {"cod_ibge": 2503506, "municipio": "Cacimbas",         "lat": -7.22,  "lon": -37.14},
    {"cod_ibge": 2503555, "municipio": "Caiçara",          "lat": -6.63,  "lon": -35.32},
    {"cod_ibge": 2503605, "municipio": "Cajazeiras",       "lat": -6.89,  "lon": -38.56},
    {"cod_ibge": 2503704, "municipio": "Cajazeirinhas",    "lat": -6.89,  "lon": -38.33},
    {"cod_ibge": 2503803, "municipio": "Caldas Brandão",   "lat": -7.02,  "lon": -35.30},
    {"cod_ibge": 2503902, "municipio": "Camalaú",          "lat": -7.90,  "lon": -36.79},
    {"cod_ibge": 2504009, "municipio": "Campina Grande",   "lat": -7.23,  "lon": -35.88},
    {"cod_ibge": 2504033, "municipio": "Campo de Santana", "lat": -7.74,  "lon": -37.06},
    {"cod_ibge": 2504074, "municipio": "Capim",            "lat": -6.85,  "lon": -35.22},
    {"cod_ibge": 2504108, "municipio": "Caraúbas",         "lat": -7.74,  "lon": -36.49},
    {"cod_ibge": 2504157, "municipio": "Carrapateira",     "lat": -7.18,  "lon": -38.39},
    {"cod_ibge": 2504207, "municipio": "Casserengue",      "lat": -6.80,  "lon": -35.84},
    {"cod_ibge": 2504306, "municipio": "Catingueira",      "lat": -7.66,  "lon": -37.60},
    {"cod_ibge": 2504405, "municipio": "Catolé do Rocha",  "lat": -6.34,  "lon": -37.74},
    {"cod_ibge": 2504504, "municipio": "Caturité",         "lat": -7.42,  "lon": -36.01},
    {"cod_ibge": 2504603, "municipio": "Conceição",        "lat": -7.56,  "lon": -38.49},
    {"cod_ibge": 2504702, "municipio": "Condado",          "lat": -6.91,  "lon": -37.59},
    {"cod_ibge": 2504801, "municipio": "Conde",            "lat": -7.26,  "lon": -34.91},
    {"cod_ibge": 2504900, "municipio": "Congo",            "lat": -7.80,  "lon": -36.66},
    {"cod_ibge": 2505006, "municipio": "Coremas",          "lat": -7.02,  "lon": -37.95},
    {"cod_ibge": 2505105, "municipio": "Coxixola",         "lat": -7.64,  "lon": -36.60},
    {"cod_ibge": 2505204, "municipio": "Cruz do Espírito Santo", "lat": -7.14, "lon": -35.08},
    {"cod_ibge": 2505303, "municipio": "Cubati",           "lat": -6.96,  "lon": -36.35},
    {"cod_ibge": 2505402, "municipio": "Cuité",            "lat": -6.49,  "lon": -36.16},
    {"cod_ibge": 2505501, "municipio": "Cuité de Mamanguape", "lat": -6.73, "lon": -35.25},
    {"cod_ibge": 2505600, "municipio": "Curral de Cima",   "lat": -6.68,  "lon": -35.18},
    {"cod_ibge": 2505709, "municipio": "Curral Velho",     "lat": -7.20,  "lon": -38.05},
    {"cod_ibge": 2505808, "municipio": "Damião",           "lat": -6.61,  "lon": -36.31},
    {"cod_ibge": 2505907, "municipio": "Desterro",         "lat": -7.18,  "lon": -37.17},
    {"cod_ibge": 2506004, "municipio": "Dois Riachos",     "lat": -9.26,  "lon": -37.09},
    {"cod_ibge": 2506103, "municipio": "Diamante",         "lat": -7.43,  "lon": -38.29},
    {"cod_ibge": 2506202, "municipio": "Dona Inês",        "lat": -6.63,  "lon": -35.62},
    {"cod_ibge": 2506301, "municipio": "Downton Abbey",    "lat": -7.10,  "lon": -36.81},
    {"cod_ibge": 2506400, "municipio": "Emas",             "lat": -7.42,  "lon": -37.67},
    {"cod_ibge": 2506509, "municipio": "Esperança",        "lat": -7.02,  "lon": -35.86},
    {"cod_ibge": 2506608, "municipio": "Fagundes",         "lat": -7.36,  "lon": -35.99},
    {"cod_ibge": 2506707, "municipio": "Frei Martinho",    "lat": -6.80,  "lon": -36.26},
    {"cod_ibge": 2506806, "municipio": "Gado Bravo",       "lat": -7.64,  "lon": -35.84},
    {"cod_ibge": 2506905, "municipio": "Guarabira",        "lat": -6.86,  "lon": -35.49},
    {"cod_ibge": 2507002, "municipio": "Gurinhém",         "lat": -7.13,  "lon": -35.41},
    {"cod_ibge": 2507101, "municipio": "Gurjão",           "lat": -7.27,  "lon": -36.48},
    {"cod_ibge": 2507200, "municipio": "Ibiara",           "lat": -7.52,  "lon": -38.29},
    {"cod_ibge": 2507309, "municipio": "Imaculada",        "lat": -7.41,  "lon": -37.51},
    {"cod_ibge": 2507408, "municipio": "Ingá",             "lat": -7.28,  "lon": -35.61},
    {"cod_ibge": 2507507, "municipio": "Itabaiana",        "lat": -7.33,  "lon": -35.32},
    {"cod_ibge": 2507606, "municipio": "Itaporanga",       "lat": -7.30,  "lon": -38.15},
    {"cod_ibge": 2507705, "municipio": "Itapororoca",      "lat": -6.82,  "lon": -35.24},
    {"cod_ibge": 2507804, "municipio": "Itatuba",          "lat": -7.44,  "lon": -35.66},
    {"cod_ibge": 2507903, "municipio": "Jacaraú",          "lat": -6.60,  "lon": -35.29},
    {"cod_ibge": 2508000, "municipio": "Jericó",           "lat": -6.53,  "lon": -37.82},
    {"cod_ibge": 2508109, "municipio": "João Pessoa",      "lat": -7.11,  "lon": -34.86},
    {"cod_ibge": 2508208, "municipio": "Juarez Távora",    "lat": -7.17,  "lon": -35.58},
    {"cod_ibge": 2508307, "municipio": "Juazeirinho",      "lat": -7.07,  "lon": -36.57},
    {"cod_ibge": 2508406, "municipio": "Junco do Seridó",  "lat": -6.99,  "lon": -36.71},
    {"cod_ibge": 2508505, "municipio": "Juripiranga",      "lat": -7.38,  "lon": -35.27},
    {"cod_ibge": 2508604, "municipio": "Juru",             "lat": -7.53,  "lon": -37.83},
    {"cod_ibge": 2508703, "municipio": "Lagoa",            "lat": -6.60,  "lon": -37.89},
    {"cod_ibge": 2508802, "municipio": "Lagoa de Dentro",  "lat": -6.68,  "lon": -35.46},
    {"cod_ibge": 2508901, "municipio": "Lagoa Seca",       "lat": -7.17,  "lon": -35.85},
    {"cod_ibge": 2509008, "municipio": "Lastro",           "lat": -6.50,  "lon": -38.01},
    {"cod_ibge": 2509107, "municipio": "Livramento",       "lat": -7.37,  "lon": -36.77},
    {"cod_ibge": 2509156, "municipio": "Logradouro",       "lat": -6.64,  "lon": -35.72},
    {"cod_ibge": 2509206, "municipio": "Lucena",           "lat": -6.90,  "lon": -34.90},
    {"cod_ibge": 2509305, "municipio": "Mãe d'Água",       "lat": -7.26,  "lon": -37.42},
    {"cod_ibge": 2509404, "municipio": "Malta",            "lat": -6.89,  "lon": -37.53},
    {"cod_ibge": 2509503, "municipio": "Mamanguape",       "lat": -6.84,  "lon": -35.12},
    {"cod_ibge": 2509602, "municipio": "Manaíra",          "lat": -7.70,  "lon": -38.14},
    {"cod_ibge": 2509701, "municipio": "Marcação",         "lat": -6.76,  "lon": -35.02},
    {"cod_ibge": 2509800, "municipio": "Mari",             "lat": -7.06,  "lon": -35.33},
    {"cod_ibge": 2509909, "municipio": "Marizópolis",      "lat": -6.74,  "lon": -38.52},
    {"cod_ibge": 2510006, "municipio": "Massaranduba",     "lat": -7.16,  "lon": -35.77},
    {"cod_ibge": 2510105, "municipio": "Mataraca",         "lat": -6.58,  "lon": -35.05},
    {"cod_ibge": 2510204, "municipio": "Matinhas",         "lat": -7.04,  "lon": -35.73},
    {"cod_ibge": 2510303, "municipio": "Mato Grosso",      "lat": -6.57,  "lon": -37.72},
    {"cod_ibge": 2510402, "municipio": "Maturéia",         "lat": -7.37,  "lon": -37.43},
    {"cod_ibge": 2510501, "municipio": "Mogeiro",          "lat": -7.29,  "lon": -35.48},
    {"cod_ibge": 2510600, "municipio": "Montadas",         "lat": -7.05,  "lon": -35.95},
    {"cod_ibge": 2510709, "municipio": "Monte Horebe",     "lat": -7.05,  "lon": -38.47},
    {"cod_ibge": 2510808, "municipio": "Monteiro",         "lat": -7.89,  "lon": -37.12},
    {"cod_ibge": 2510907, "municipio": "Mulungu",          "lat": -6.79,  "lon": -35.46},
    {"cod_ibge": 2511004, "municipio": "Natuba",           "lat": -7.66,  "lon": -35.55},
    {"cod_ibge": 2511103, "municipio": "Nazarezinho",      "lat": -6.91,  "lon": -38.32},
    {"cod_ibge": 2511202, "municipio": "Nova Floresta",    "lat": -6.46,  "lon": -36.20},
    {"cod_ibge": 2511301, "municipio": "Nova Olinda",      "lat": -7.51,  "lon": -38.03},
    {"cod_ibge": 2511400, "municipio": "Nova Palmeira",    "lat": -6.73,  "lon": -36.41},
    {"cod_ibge": 2511509, "municipio": "Olho d'Água",      "lat": -6.77,  "lon": -37.74},
    {"cod_ibge": 2511608, "municipio": "Olivedos",         "lat": -6.95,  "lon": -36.22},
    {"cod_ibge": 2511707, "municipio": "Ouro Velho",       "lat": -7.71,  "lon": -37.20},
    {"cod_ibge": 2511806, "municipio": "Parari",           "lat": -7.39,  "lon": -36.65},
    {"cod_ibge": 2511905, "municipio": "Passagem",         "lat": -6.77,  "lon": -36.60},
    {"cod_ibge": 2512002, "municipio": "Patos",            "lat": -7.02,  "lon": -37.28},
    {"cod_ibge": 2512101, "municipio": "Paulista",         "lat": -6.57,  "lon": -37.62},
    {"cod_ibge": 2512200, "municipio": "Pedra Branca",     "lat": -7.49,  "lon": -37.68},
    {"cod_ibge": 2512309, "municipio": "Pedra Lavrada",    "lat": -6.76,  "lon": -36.47},
    {"cod_ibge": 2512408, "municipio": "Pedras de Fogo",   "lat": -7.40,  "lon": -35.11},
    {"cod_ibge": 2512507, "municipio": "Piancó",           "lat": -7.20,  "lon": -37.93},
    {"cod_ibge": 2512606, "municipio": "Picuí",            "lat": -6.55,  "lon": -36.35},
    {"cod_ibge": 2512705, "municipio": "Pilar",            "lat": -7.26,  "lon": -35.25},
    {"cod_ibge": 2512804, "municipio": "Pilõezinhos",      "lat": -6.85,  "lon": -35.38},
    {"cod_ibge": 2512903, "municipio": "Pirpirituba",      "lat": -6.76,  "lon": -35.46},
    {"cod_ibge": 2513000, "municipio": "Pitimbu",          "lat": -7.47,  "lon": -34.81},
    {"cod_ibge": 2513109, "municipio": "Pocinhos",         "lat": -7.07,  "lon": -36.06},
    {"cod_ibge": 2513158, "municipio": "Poço Dantas",      "lat": -6.62,  "lon": -38.27},
    {"cod_ibge": 2513208, "municipio": "Poço de José de Moura", "lat": -6.64, "lon": -38.46},
    {"cod_ibge": 2513307, "municipio": "Pombal",           "lat": -6.77,  "lon": -37.80},
    {"cod_ibge": 2513406, "municipio": "Prata",            "lat": -7.69,  "lon": -37.09},
    {"cod_ibge": 2513505, "municipio": "Princesa Isabel",  "lat": -7.73,  "lon": -38.01},
    {"cod_ibge": 2513604, "municipio": "Puxinanã",         "lat": -7.16,  "lon": -35.96},
    {"cod_ibge": 2513703, "municipio": "Queimadas",        "lat": -7.36,  "lon": -35.90},
    {"cod_ibge": 2513802, "municipio": "Quixabá",          "lat": -7.12,  "lon": -37.23},
    {"cod_ibge": 2513901, "municipio": "Remígio",          "lat": -6.93,  "lon": -35.80},
    {"cod_ibge": 2514008, "municipio": "Riachão",          "lat": -6.52,  "lon": -35.63},
    {"cod_ibge": 2514107, "municipio": "Riachão do Bacamarte", "lat": -7.18, "lon": -35.67},
    {"cod_ibge": 2514206, "municipio": "Riachão do Poço",  "lat": -7.38,  "lon": -35.28},
    {"cod_ibge": 2514305, "municipio": "Riacho de Santo Antônio", "lat": -7.67, "lon": -36.01},
    {"cod_ibge": 2514404, "municipio": "Riacho dos Cavalos", "lat": -6.43, "lon": -37.68},
    {"cod_ibge": 2514503, "municipio": "Rio Tinto",        "lat": -6.81,  "lon": -35.07},
    {"cod_ibge": 2514602, "municipio": "Salgadinho",       "lat": -6.89,  "lon": -36.59},
    {"cod_ibge": 2514701, "municipio": "Salgado de São Félix", "lat": -7.36, "lon": -35.45},
    {"cod_ibge": 2514800, "municipio": "Santa Cecília",    "lat": -7.10,  "lon": -37.39},
    {"cod_ibge": 2514909, "municipio": "Santa Cruz",       "lat": -6.48,  "lon": -36.02},
    {"cod_ibge": 2515005, "municipio": "Santa Helena",     "lat": -6.82,  "lon": -38.49},
    {"cod_ibge": 2515104, "municipio": "Santa Inês",       "lat": -7.17,  "lon": -36.64},
    {"cod_ibge": 2515203, "municipio": "Santa Luzia",      "lat": -6.87,  "lon": -36.92},
    {"cod_ibge": 2515302, "municipio": "Santana de Mangueira", "lat": -7.56, "lon": -38.11},
    {"cod_ibge": 2515401, "municipio": "Santana dos Garrotes", "lat": -7.38, "lon": -37.97},
    {"cod_ibge": 2515500, "municipio": "Joca Claudino",    "lat": -6.66,  "lon": -38.61},
    {"cod_ibge": 2515609, "municipio": "Santa Rita",       "lat": -7.11,  "lon": -34.98},
    {"cod_ibge": 2515708, "municipio": "Santa Teresinha",  "lat": -7.10,  "lon": -37.47},
    {"cod_ibge": 2515807, "municipio": "Santo André",      "lat": -6.87,  "lon": -36.40},
    {"cod_ibge": 2515906, "municipio": "São Bento",        "lat": -6.49,  "lon": -37.44},
    {"cod_ibge": 2516003, "municipio": "São Bentinho",     "lat": -6.76,  "lon": -37.52},
    {"cod_ibge": 2516102, "municipio": "São Domingos do Cariri", "lat": -7.69, "lon": -36.36},
    {"cod_ibge": 2516151, "municipio": "São Domingos",     "lat": -6.76,  "lon": -37.93},
    {"cod_ibge": 2516201, "municipio": "São Francisco",    "lat": -6.44,  "lon": -38.10},
    {"cod_ibge": 2516300, "municipio": "São João do Cariri", "lat": -7.40, "lon": -36.53},
    {"cod_ibge": 2516409, "municipio": "São João do Rio do Peixe", "lat": -6.72, "lon": -38.45},
    {"cod_ibge": 2516508, "municipio": "São João do Tigre", "lat": -8.07,  "lon": -36.82},
    {"cod_ibge": 2516607, "municipio": "São José da Lagoa Tapada", "lat": -6.93, "lon": -38.13},
    {"cod_ibge": 2516706, "municipio": "São José de Caiana", "lat": -7.72, "lon": -38.30},
    {"cod_ibge": 2516805, "municipio": "São José de Espinharas", "lat": -6.84, "lon": -37.30},
    {"cod_ibge": 2516904, "municipio": "São José dos Ramos", "lat": -7.47, "lon": -35.17},
    {"cod_ibge": 2517001, "municipio": "São José de Piranhas", "lat": -7.12, "lon": -38.50},
    {"cod_ibge": 2517100, "municipio": "São José de Princesa", "lat": -7.74, "lon": -38.08},
    {"cod_ibge": 2517209, "municipio": "São José do Bonfim", "lat": -7.00,  "lon": -37.32},
    {"cod_ibge": 2517308, "municipio": "São José do Brejo do Cruz", "lat": -6.19, "lon": -37.47},
    {"cod_ibge": 2517407, "municipio": "São José do Sabugi", "lat": -6.76,  "lon": -36.83},
    {"cod_ibge": 2517506, "municipio": "São José dos Cordeiros", "lat": -7.38, "lon": -36.81},
    {"cod_ibge": 2517534, "municipio": "São Mamede",       "lat": -6.93,  "lon": -37.09},
    {"cod_ibge": 2517605, "municipio": "São Miguel de Taipu", "lat": -7.19, "lon": -35.19},
    {"cod_ibge": 2517704, "municipio": "São Sebastião de Lagoa de Roça", "lat": -7.19, "lon": -35.85},
    {"cod_ibge": 2517803, "municipio": "São Sebastião do Umbuzeiro", "lat": -8.04, "lon": -37.00},
    {"cod_ibge": 2517902, "municipio": "Sapé",             "lat": -7.09,  "lon": -35.22},
    {"cod_ibge": 2518009, "municipio": "São Vicente do Seridó", "lat": -6.89, "lon": -36.72},
    {"cod_ibge": 2518108, "municipio": "Serra Branca",     "lat": -7.49,  "lon": -36.65},
    {"cod_ibge": 2518207, "municipio": "Serra da Raiz",    "lat": -6.71,  "lon": -35.40},
    {"cod_ibge": 2518306, "municipio": "Serra Grande",     "lat": -7.14,  "lon": -37.45},
    {"cod_ibge": 2518405, "municipio": "Serra Redonda",    "lat": -7.21,  "lon": -35.87},
    {"cod_ibge": 2518504, "municipio": "Serraria",         "lat": -6.81,  "lon": -35.64},
    {"cod_ibge": 2518553, "municipio": "Sertãozinho",      "lat": -6.96,  "lon": -35.31},
    {"cod_ibge": 2518603, "municipio": "Sobrado",          "lat": -7.06,  "lon": -35.19},
    {"cod_ibge": 2518702, "municipio": "Solânea",          "lat": -6.75,  "lon": -35.65},
    {"cod_ibge": 2518801, "municipio": "Soledade",         "lat": -7.06,  "lon": -36.37},
    {"cod_ibge": 2518850, "municipio": "Sossêgo",          "lat": -6.75,  "lon": -36.17},
    {"cod_ibge": 2518900, "municipio": "Sousa",            "lat": -6.76,  "lon": -38.23},
    {"cod_ibge": 2519006, "municipio": "Sumé",             "lat": -7.67,  "lon": -36.88},
    {"cod_ibge": 2519105, "municipio": "Taperoá",          "lat": -7.21,  "lon": -36.83},
    {"cod_ibge": 2519204, "municipio": "Tavares",          "lat": -7.62,  "lon": -37.87},
    {"cod_ibge": 2519303, "municipio": "Teixeira",         "lat": -7.22,  "lon": -37.25},
    {"cod_ibge": 2519402, "municipio": "Tenório",          "lat": -6.96,  "lon": -36.50},
    {"cod_ibge": 2519501, "municipio": "Triunfo",          "lat": -7.84,  "lon": -38.10},
    {"cod_ibge": 2519600, "municipio": "Uiraúna",          "lat": -6.52,  "lon": -38.41},
    {"cod_ibge": 2519709, "municipio": "Umbuzeiro",        "lat": -7.57,  "lon": -35.67},
    {"cod_ibge": 2519808, "municipio": "Várzea",           "lat": -6.76,  "lon": -36.98},
    {"cod_ibge": 2519907, "municipio": "Vieirópolis",      "lat": -6.53,  "lon": -38.24},
    {"cod_ibge": 2520004, "municipio": "Vista Serrana",    "lat": -6.76,  "lon": -37.55},
    {"cod_ibge": 2520103, "municipio": "Zabelê",           "lat": -7.84,  "lon": -36.88},
]


def _gerar_dados_sinteticos(seed: int = 42) -> gpd.GeoDataFrame:
    """
    Gera um GeoDataFrame sintético representando os municípios da Paraíba.
    Simula dados do TSE (eleitorado) e IBGE (infraestrutura).
    """
    rng = np.random.default_rng(seed)
    records = _MUNICIPIOS_PB_SEED.copy()

    # Municípios de grande porte fixos com eleitorado realista
    grandes_municipios = {
        2504009: 340_000,  # Campina Grande
        2508109: 750_000,  # João Pessoa
        2512002: 110_000,  # Patos
        2503605: 70_000,   # Cajazeiras
        2518900: 65_000,   # Sousa
        2506905: 70_000,   # Guarabira
        2507002: 60_000,   # Gurinhém
        2501807:  80_000,  # Bayeux
        2515609:  90_000,  # Santa Rita
        2503100:  60_000,  # Cabedelo
    }

    for rec in records:
        cod = rec["cod_ibge"]
        base_el = grandes_municipios.get(cod, rng.integers(3_000, 40_000))
        rec["eleitorado_total"] = int(base_el * rng.uniform(0.9, 1.1))
        rec["votos_partido_2022"] = int(rec["eleitorado_total"] * rng.uniform(0.05, 0.35))
        rec["pct_votos_partido"] = round(rec["votos_partido_2022"] / rec["eleitorado_total"], 4)
        rec["indice_infraestrutura"] = round(float(rng.uniform(0.1, 1.0)), 2)
        rec["alinhamento_prefeito"] = int(rng.integers(1, 6))
        rec["tem_lideranca"] = bool(rng.choice([True, False], p=[0.4, 0.6]))
        rec["status_politico"] = str(rng.choice(
            ["Aliado", "Neutro", "Oposição"],
            p=[0.35, 0.40, 0.25]
        ))
        rec["peso_lideranca"] = int(rng.integers(1, 6))
        rec["fixado_polo"] = False
        rec["geometria"] = Point(rec["lon"], rec["lat"])

    gdf = gpd.GeoDataFrame(records, geometry="geometria", crs="EPSG:4326")
    return gdf


def carregar_base_mestra(
    usar_geobr: bool = False,
    forcar_regerar: bool = False
) -> gpd.GeoDataFrame:
    """
    Carrega a malha oficial do IBGE e mescla com os dados simulados do TSE/IBGE.
    """
    CACHE_PATH.mkdir(exist_ok=True)
    cache_file = CACHE_PATH / "base_mestra_v3.pkl"

    if cache_file.exists() and not forcar_regerar:
        logger.info("Carregando Base Mestra do cache local.")
        return pd.read_pickle(cache_file)

    try:
        # Carregar GeoJSON local
        logger.info(f"Carregando malha do IBGE de {GEOJSON_FALLBACK}")
        malha_pb = gpd.read_file(GEOJSON_FALLBACK)
        # O IBGE retorna 'id' (codigo) e 'name'
        malha_pb = malha_pb.rename(columns={"id": "cod_ibge", "name": "municipio"})
        # Garantir tipo int
        malha_pb["cod_ibge"] = malha_pb["cod_ibge"].astype(str).str[:7].astype(int) 

        # Dados sintéticos de eleitorado (tem o dict de 223 cidades)
        sintetico = _gerar_dados_sinteticos()
        # descartar geometria do sintético (pontos) pois usaremos os polígonos reais
        sintetico = sintetico.drop(columns=["geometria", "lat", "lon"])
        
        # Merge
        gdf = malha_pb.merge(sintetico, on="cod_ibge", how="inner", suffixes=("", "_syn"))
        
        # Calcular centroides para as operações que precisam de lat/lon (ex: ORS)
        gdf["lat"] = gdf.geometry.centroid.y
        gdf["lon"] = gdf.geometry.centroid.x

    except Exception as e:
        logger.error(f"Erro ao carregar malha do IBGE: {e}")
        logger.info("Fazendo fallback para dados sintéticos pontuais...")
        gdf = _gerar_dados_sinteticos()

    # Persistir em cache
    pd.to_pickle(gdf, cache_file)
    logger.info(f"Base Mestra gerada com {len(gdf)} municípios. Cache salvo.")
    return gdf


def _tentar_carregar_geobr() -> Optional[gpd.GeoDataFrame]:
    """
    Tenta carregar a malha municipal da Paraíba via geobr.
    Retorna None se falhar (sem internet, lib não instalada, etc.).
    """
    try:
        import geobr  # type: ignore
        logger.info("Baixando malha municipal da Paraíba via geobr...")
        municipios_pb = geobr.read_municipality(code_muni="PB", year=2020)
        municipios_pb = municipios_pb.rename(columns={
            "code_muni": "cod_ibge",
            "name_muni": "municipio",
        })
        municipios_pb["cod_ibge"] = municipios_pb["cod_ibge"].astype(int)
        municipios_pb["lat"] = municipios_pb.geometry.centroid.y
        municipios_pb["lon"] = municipios_pb.geometry.centroid.x

        # Unir com dados sintéticos de eleitorado/político
        sintetico = _gerar_dados_sinteticos()
        sintetico_meta = sintetico.drop(columns=["geometria"], errors="ignore")

        merged = municipios_pb.merge(sintetico_meta, on="cod_ibge", how="left", suffixes=("", "_syn"))
        # preencher colunas faltantes com valores sintéticos aleatórios
        rng = np.random.default_rng(99)
        for col in ["eleitorado_total", "votos_partido_2022", "indice_infraestrutura",
                    "alinhamento_prefeito", "tem_lideranca", "status_politico", "peso_lideranca"]:
            if col not in merged.columns or merged[col].isna().any():
                merged[col] = sintetico[col].values[:len(merged)]

        merged["fixado_polo"] = False
        return merged.to_crs("EPSG:4326")

    except Exception as e:
        logger.warning(f"geobr indisponível ({e}). Usando dados sintéticos.")
        return None


def salvar_base_mestra(gdf: gpd.GeoDataFrame, path: Optional[str] = None) -> str:
    """
    Salva a Base Mestra em formato CSV (sem geometria) para exportação.
    """
    output_path = path or "dados_paraiba.csv"
    export_df = gdf.copy()
    if "geometria" in export_df.columns:
        export_df = export_df.drop(columns=["geometria"])
    if "geometry" in export_df.columns:
        export_df = export_df.drop(columns=["geometry"])
    export_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Base Mestra exportada para {output_path}")
    return output_path
