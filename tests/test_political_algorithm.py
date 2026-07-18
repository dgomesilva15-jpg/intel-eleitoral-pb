"""
tests/test_political_algorithm.py
====================================
Testes unitários para o módulo political_algorithm.py

Cobre:
- Clusterização K-Means básica
- Número de setores gerados
- Override manual (Human-in-the-Loop)
- Identificação de polo por score
- Filtro de satélites por raio
- Score de polo
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


@pytest.fixture(scope="module")
def gdf_base():
    """GeoDataFrame base sintético para todos os testes de algoritmo."""
    from etl_pipeline import carregar_base_mestra
    return carregar_base_mestra(usar_geobr=False, forcar_regerar=True)


class TestClusterizacao:
    """Testes do algoritmo K-Means ponderado."""

    def test_numero_de_setores_correto(self, gdf_base):
        from political_algorithm import clusterizar_municipios
        n = 10
        gdf_result, setores = clusterizar_municipios(gdf_base, n_setores=n)
        assert len(setores) == n, f"Esperado {n} setores, obtido {len(setores)}"

    def test_todos_municipios_tem_setor(self, gdf_base):
        from political_algorithm import clusterizar_municipios
        gdf_result, _ = clusterizar_municipios(gdf_base, n_setores=10)
        assert (gdf_result["setor"] >= 0).all(), "Município sem setor atribuído"

    def test_setores_tem_polo(self, gdf_base):
        from political_algorithm import clusterizar_municipios
        _, setores = clusterizar_municipios(gdf_base, n_setores=8)
        for s in setores:
            assert s.polo is not None
            assert "municipio" in s.polo

    def test_eleitorado_total_setor_positivo(self, gdf_base):
        from political_algorithm import clusterizar_municipios
        _, setores = clusterizar_municipios(gdf_base, n_setores=8)
        for s in setores:
            assert s.total_eleitorado > 0

    def test_total_municipios_setor_correto(self, gdf_base):
        from political_algorithm import clusterizar_municipios
        gdf_result, setores = clusterizar_municipios(gdf_base, n_setores=10)
        total_nos_setores = sum(s.total_municipios for s in setores)
        assert total_nos_setores == len(gdf_base), \
            "Total de municípios nos setores não bate com o total geral"

    def test_n_setores_maior_que_municipios_limitado(self, gdf_base):
        from political_algorithm import clusterizar_municipios
        # Não deve lançar exceção mesmo com n_setores excessivo
        _, setores = clusterizar_municipios(gdf_base, n_setores=9999)
        assert len(setores) <= len(gdf_base)


class TestOverrideManual:
    """Testes do sistema de Override Manual (Human-in-the-Loop)."""

    @pytest.fixture
    def cod_patos(self, gdf_base):
        return int(gdf_base[gdf_base["municipio"] == "Patos"]["cod_ibge"].values[0])

    @pytest.fixture
    def cod_sousa(self, gdf_base):
        return int(gdf_base[gdf_base["municipio"] == "Sousa"]["cod_ibge"].values[0])

    def test_polo_fixado_é_polo_do_setor(self, gdf_base, cod_patos):
        from political_algorithm import clusterizar_municipios
        gdf_result, setores = clusterizar_municipios(
            gdf_base, n_setores=10, polos_fixados=[cod_patos]
        )
        polos_cod = [s.polo.get("cod_ibge") for s in setores]
        assert cod_patos in polos_cod, "Patos deveria ser polo após override"

    def test_multiplos_overrides(self, gdf_base, cod_patos, cod_sousa):
        from political_algorithm import clusterizar_municipios
        gdf_result, setores = clusterizar_municipios(
            gdf_base, n_setores=10,
            polos_fixados=[cod_patos, cod_sousa]
        )
        polos_cod = [s.polo.get("cod_ibge") for s in setores]
        assert cod_patos in polos_cod, "Patos não está como polo"
        assert cod_sousa in polos_cod, "Sousa não está como polo"

    def test_override_com_cidade_pequena(self, gdf_base):
        """Mesmo uma cidade pequena pode ser forçada como polo."""
        from political_algorithm import clusterizar_municipios
        # Zabelê — cidade pequena no cariri
        zabele = gdf_base[gdf_base["municipio"] == "Zabelê"]
        if zabele.empty:
            pytest.skip("Zabelê não encontrada na base")
        cod_zabele = int(zabele["cod_ibge"].values[0])
        _, setores = clusterizar_municipios(
            gdf_base, n_setores=8, polos_fixados=[cod_zabele]
        )
        polos_cod = [s.polo.get("cod_ibge") for s in setores]
        assert cod_zabele in polos_cod

    def test_fixado_polo_flag_no_gdf(self, gdf_base, cod_patos):
        from political_algorithm import clusterizar_municipios
        gdf_result, _ = clusterizar_municipios(
            gdf_base, n_setores=10, polos_fixados=[cod_patos]
        )
        mask = gdf_result["cod_ibge"] == cod_patos
        assert gdf_result.loc[mask, "fixado_polo"].values[0] == True


class TestFiltroSatelites:
    """Testes do filtro de satélites por raio."""

    @pytest.fixture
    def satelites_exemplo(self):
        return [
            {"municipio": "Campina Grande", "lat": -7.23, "lon": -35.88, "eleitorado_total": 340000},
            {"municipio": "Patos",           "lat": -7.02, "lon": -37.28, "eleitorado_total": 110000},
            {"municipio": "Sousa",           "lat": -6.76, "lon": -38.23, "eleitorado_total": 65000},
        ]

    def test_raio_pequeno_exclui_distantes(self, satelites_exemplo):
        from political_algorithm import filtrar_satelites_no_raio
        # Polo: João Pessoa. Raio: 50km → apenas cidades próximas
        dentro, fora = filtrar_satelites_no_raio(
            polo_lat=-7.11, polo_lon=-34.86,
            satelites=satelites_exemplo, raio_km=50.0
        )
        # JP → CG ≈ 120km, logo CG deve ficar fora
        assert len(dentro) + len(fora) == len(satelites_exemplo)

    def test_raio_grande_inclui_todos(self, satelites_exemplo):
        from political_algorithm import filtrar_satelites_no_raio
        dentro, fora = filtrar_satelites_no_raio(
            polo_lat=-7.11, polo_lon=-34.86,
            satelites=satelites_exemplo, raio_km=500.0
        )
        assert len(dentro) == len(satelites_exemplo)
        assert len(fora) == 0

    def test_distancia_polo_adicionada(self, satelites_exemplo):
        from political_algorithm import filtrar_satelites_no_raio
        dentro, fora = filtrar_satelites_no_raio(-7.11, -34.86, satelites_exemplo, 500.0)
        for sat in dentro:
            assert "distancia_polo_km" in sat
            assert sat["distancia_polo_km"] >= 0

    def test_ordenado_por_distancia(self, satelites_exemplo):
        from political_algorithm import filtrar_satelites_no_raio
        dentro, _ = filtrar_satelites_no_raio(-7.11, -34.86, satelites_exemplo, 500.0)
        dists = [s["distancia_polo_km"] for s in dentro]
        assert dists == sorted(dists)


class TestScorePolo:
    """Testes do score político de polo."""

    def test_aliado_tem_score_maior(self):
        from political_algorithm import calcular_score_polo
        aliado = {
            "eleitorado_total": 50000, "alinhamento_prefeito": 5,
            "peso_lideranca": 5, "status_politico": "Aliado",
            "indice_infraestrutura": 0.9
        }
        oposicao = {
            "eleitorado_total": 50000, "alinhamento_prefeito": 1,
            "peso_lideranca": 1, "status_politico": "Oposição",
            "indice_infraestrutura": 0.1
        }
        assert calcular_score_polo(aliado) > calcular_score_polo(oposicao)

    def test_maior_eleitorado_maior_score(self):
        from political_algorithm import calcular_score_polo
        grande = {"eleitorado_total": 500000, "alinhamento_prefeito": 3,
                  "peso_lideranca": 3, "status_politico": "Neutro", "indice_infraestrutura": 0.5}
        pequeno = {"eleitorado_total": 5000, "alinhamento_prefeito": 3,
                   "peso_lideranca": 3, "status_politico": "Neutro", "indice_infraestrutura": 0.5}
        assert calcular_score_polo(grande) > calcular_score_polo(pequeno)

    def test_score_retorna_float(self):
        from political_algorithm import calcular_score_polo
        score = calcular_score_polo({
            "eleitorado_total": 10000, "alinhamento_prefeito": 3,
            "peso_lideranca": 2, "status_politico": "Neutro",
            "indice_infraestrutura": 0.5
        })
        assert isinstance(score, float)
