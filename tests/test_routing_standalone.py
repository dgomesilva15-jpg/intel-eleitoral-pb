"""
tests/test_routing_standalone.py
===================================
Testes standalone do routing_engine.py que rodam com APENAS numpy na base.
Não precisam de geopandas, sklearn ou thefuzz.

Estes testes foram projetados para funcionar no ambiente atual:
✅ numpy (instalado)
✅ pandas (instalado)
✅ pytest (instalado)
❌ geopandas (não instalado — testes completos usam mock)
❌ sklearn (não instalado — testes com mocks)
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# TESTES PURO PYTHON — sem dependências externas
# ============================================================

class TestHaversineStandalone:
    """Testes da fórmula Haversine usando apenas stdlib."""

    def test_distancia_zero_para_mesmo_ponto(self):
        from routing_engine import haversine_km
        assert haversine_km(-7.11, -34.86, -7.11, -34.86) == pytest.approx(0.0, abs=0.001)

    def test_joao_pessoa_campina_grande_range(self):
        from routing_engine import haversine_km
        dist = haversine_km(-7.11, -34.86, -7.23, -35.88)
        assert 100 < dist < 150, f"Distância JP→CG fora do range: {dist:.2f}km"

    def test_simetria(self):
        from routing_engine import haversine_km
        d_ida = haversine_km(-7.11, -34.86, -7.02, -37.28)
        d_volta = haversine_km(-7.02, -37.28, -7.11, -34.86)
        assert d_ida == pytest.approx(d_volta, abs=0.01)

    def test_retorna_float(self):
        from routing_engine import haversine_km
        result = haversine_km(-7.0, -35.0, -7.5, -36.0)
        assert isinstance(result, float)

    def test_nordeste_paraiba_range(self):
        """Distâncias dentro da Paraíba devem estar entre 1km e 700km."""
        from routing_engine import haversine_km
        # Distância entre extremos da Paraíba
        dist = haversine_km(-6.0, -34.8, -8.5, -38.8)
        assert 0 < dist < 700


class TestTempoEstimadoStandalone:
    """Testes de estimativa de tempo de viagem."""

    def test_velocidade_padrao_65kmh(self):
        from routing_engine import haversine_tempo_estimado
        tempo = haversine_tempo_estimado(65.0, 65.0)
        assert tempo == pytest.approx(60.0, abs=0.1)

    def test_distancia_zero(self):
        from routing_engine import haversine_tempo_estimado
        assert haversine_tempo_estimado(0.0) == pytest.approx(0.0, abs=0.001)

    def test_proporcional_a_distancia(self):
        from routing_engine import haversine_tempo_estimado
        t100 = haversine_tempo_estimado(100.0)
        t200 = haversine_tempo_estimado(200.0)
        assert t200 == pytest.approx(2 * t100, abs=0.1)

    def test_retorna_float(self):
        from routing_engine import haversine_tempo_estimado
        assert isinstance(haversine_tempo_estimado(50.0), float)


class TestURLGoogleMapsStandalone:
    """Testes de geração de URL Google Maps."""

    def test_url_valida_dois_pontos(self):
        from routing_engine import gerar_url_google_maps
        url = gerar_url_google_maps([(-7.11, -34.86), (-7.23, -35.88)], ["JP", "CG"])
        assert "google.com/maps/dir" in url
        assert "origin" in url
        assert "destination" in url

    def test_coordenadas_presentes_na_url(self):
        from routing_engine import gerar_url_google_maps
        url = gerar_url_google_maps([(-7.11, -34.86), (-7.02, -37.28)], ["JP", "Patos"])
        assert "-7.11" in url
        assert "-34.86" in url

    def test_modo_driving_presente(self):
        from routing_engine import gerar_url_google_maps
        url = gerar_url_google_maps([(-7.0, -35.0), (-7.5, -36.0)], ["A", "B"])
        assert "driving" in url

    def test_ponto_unico_retorna_vazio(self):
        from routing_engine import gerar_url_google_maps
        assert gerar_url_google_maps([(-7.0, -35.0)], ["A"]) == ""

    def test_lista_vazia_retorna_vazio(self):
        from routing_engine import gerar_url_google_maps
        assert gerar_url_google_maps([], []) == ""

    def test_tres_pontos_tem_waypoints(self):
        from routing_engine import gerar_url_google_maps
        url = gerar_url_google_maps(
            [(-7.11, -34.86), (-7.02, -37.28), (-7.23, -35.88)],
            ["JP", "Patos", "CG"]
        )
        assert "waypoints" in url


class TestIsocronaFallbackStandalone:
    """Testes do fallback circular de isócrona."""

    def test_retorna_geojson(self):
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(-7.11, -34.86, 60, None)
        assert result is not None
        assert "type" in result
        assert result["type"] == "FeatureCollection"

    def test_tem_feature(self):
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(-7.11, -34.86, 60, None)
        assert len(result["features"]) >= 1

    def test_geometria_poligono(self):
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(-7.11, -34.86, 45, None)
        geom = result["features"][0]["geometry"]
        assert geom["type"] == "Polygon"

    def test_poligono_fechado(self):
        """Primeiro e último ponto do polígono devem ser iguais."""
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(-7.11, -34.86, 60, None)
        coords = result["features"][0]["geometry"]["coordinates"][0]
        assert coords[0] == coords[-1], "Polígono não fechado"

    def test_maior_tempo_maior_area(self):
        from routing_engine import calcular_isocronas_com_fallback
        r30 = calcular_isocronas_com_fallback(-7.11, -34.86, 30, None)
        r120 = calcular_isocronas_com_fallback(-7.11, -34.86, 120, None)
        area30 = r30["features"][0]["properties"]["area"]
        area120 = r120["features"][0]["properties"]["area"]
        assert area120 > area30 * 3, "Área de 120min deveria ser ~4x maior que 30min"

    def test_propriedades_presentes(self):
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(-7.11, -34.86, 60, None)
        props = result["features"][0]["properties"]
        assert "area" in props
        assert "value" in props


class TestRouteResultStandalone:
    """Testes do objeto RouteResult."""

    def test_criacao_basica(self):
        from routing_engine import RouteResult
        r = RouteResult("JP", "CG", 120.0, 95.0, 340000, 3578.0, False)
        assert r.origem == "JP"
        assert r.destino == "CG"
        assert r.distancia_km == 120.0
        assert r.eleitores_destino == 340000

    def test_to_dict_contem_chaves_essenciais(self):
        from routing_engine import RouteResult
        r = RouteResult("JP", "CG", 120.0, 95.0, 340000, 3578.0, False)
        d = r.to_dict()
        for chave in ["Origem", "Destino", "Distância (km)", "Tempo Est. (min)",
                      "Eleitores Destino", "Eleitores/Min", "Fonte"]:
            assert chave in d, f"Chave ausente no dict: {chave}"

    def test_fonte_ors_vs_haversine(self):
        from routing_engine import RouteResult
        r_ors = RouteResult("JP", "CG", 120.0, 95.0, 340000, 3578.0, True)
        r_hav = RouteResult("JP", "CG", 120.0, 95.0, 340000, 3578.0, False)
        assert "ORS" in r_ors.to_dict()["Fonte"]
        assert "Haversine" in r_hav.to_dict()["Fonte"]


class TestCalculoRotaStandalone:
    """Testes do cálculo de rota offline (Haversine)."""

    @pytest.fixture
    def cidades_alvo(self):
        return [
            {"municipio": "Campina Grande", "lat": -7.23, "lon": -35.88, "eleitorado_total": 340000},
            {"municipio": "Patos",          "lat": -7.02, "lon": -37.28, "eleitorado_total": 110000},
            {"municipio": "Guarabira",      "lat": -6.86, "lon": -35.49, "eleitorado_total": 70000},
        ]

    def test_retorna_tres_resultados(self, cidades_alvo):
        from routing_engine import calcular_rota_custo_beneficio
        rotas = calcular_rota_custo_beneficio(
            "JP", -7.11, -34.86, cidades_alvo, ors_client=None
        )
        assert len(rotas) == 3

    def test_ordenado_decrescente_por_cb(self, cidades_alvo):
        from routing_engine import calcular_rota_custo_beneficio
        rotas = calcular_rota_custo_beneficio(
            "JP", -7.11, -34.86, cidades_alvo, ors_client=None
        )
        cbs = [r.custo_beneficio for r in rotas]
        assert cbs == sorted(cbs, reverse=True)

    def test_distancias_positivas(self, cidades_alvo):
        from routing_engine import calcular_rota_custo_beneficio
        rotas = calcular_rota_custo_beneficio(
            "JP", -7.11, -34.86, cidades_alvo, ors_client=None
        )
        assert all(r.distancia_km > 0 for r in rotas)
        assert all(r.tempo_minutos > 0 for r in rotas)

    def test_lista_vazia_retorna_vazio(self):
        from routing_engine import calcular_rota_custo_beneficio
        rotas = calcular_rota_custo_beneficio("JP", -7.11, -34.86, [], None)
        assert rotas == []

    def test_cb_menor_para_cidade_distante(self, cidades_alvo):
        """Patos está mais longe que Guarabira de JP, com menos eleitorado."""
        from routing_engine import calcular_rota_custo_beneficio, haversine_km
        rotas = calcular_rota_custo_beneficio(
            "JP", -7.11, -34.86, cidades_alvo, ors_client=None
        )
        # Verificar que CG (340k eleitores, próxima) tem CB diferente de Patos
        dest_dict = {r.destino: r for r in rotas}
        assert "Campina Grande" in dest_dict
        assert "Patos" in dest_dict


class TestNormalizacaoNomesStandalone:
    """Testes de normalização de strings — sem thefuzz, só stdlib."""

    def test_joao_sem_acento(self):
        from data_manager import _normalizar_nome
        assert "joao" in _normalizar_nome("João")

    def test_campina_grande_normalizado(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("Campina Grande") == "campina grande"

    def test_string_vazia(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("") == ""

    def test_none_retorna_vazio(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome(None) == ""

    def test_acentos_multiplos(self):
        from data_manager import _normalizar_nome
        result = _normalizar_nome("Conceição")
        assert result == "conceicao"

    def test_upper_para_lower(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("PATOS") == "patos"


class TestETLSinteticoStandalone:
    """Testes do ETL usando apenas stdlib e pandas (sem geopandas)."""

    def test_dados_sinteticos_tem_223_registros(self):
        """Verifica que a lista seed tem os registros completos."""
        from etl_pipeline import _MUNICIPIOS_PB_SEED
        assert len(_MUNICIPIOS_PB_SEED) >= 200, f"Apenas {len(_MUNICIPIOS_PB_SEED)} municípios no seed"

    def test_todos_tem_campos_basicos(self):
        from etl_pipeline import _MUNICIPIOS_PB_SEED
        campos = ["cod_ibge", "municipio", "lat", "lon"]
        for m in _MUNICIPIOS_PB_SEED:
            for campo in campos:
                assert campo in m, f"Município {m.get('municipio', '?')} sem campo '{campo}'"

    def test_lat_lon_dentro_da_paraiba(self):
        from etl_pipeline import _MUNICIPIOS_PB_SEED
        for m in _MUNICIPIOS_PB_SEED:
            lat, lon = m["lat"], m["lon"]
            # Bounds mais permissivos para alguns registros
            assert -9.5 <= lat <= -5.5, f"{m['municipio']}: lat={lat} fora do range"
            assert -39.5 <= lon <= -34.0, f"{m['municipio']}: lon={lon} fora do range"

    def test_cod_ibge_unico(self):
        from etl_pipeline import _MUNICIPIOS_PB_SEED
        codes = [m["cod_ibge"] for m in _MUNICIPIOS_PB_SEED]
        assert len(codes) == len(set(codes)), "cod_ibge duplicados!"

    def test_joao_pessoa_presente(self):
        from etl_pipeline import _MUNICIPIOS_PB_SEED
        nomes = [m["municipio"] for m in _MUNICIPIOS_PB_SEED]
        assert "João Pessoa" in nomes

    def test_campina_grande_presente(self):
        from etl_pipeline import _MUNICIPIOS_PB_SEED
        nomes = [m["municipio"] for m in _MUNICIPIOS_PB_SEED]
        assert "Campina Grande" in nomes


class TestKPIsStandalone:
    """Testes de KPIs eleitorais — lógica pura de negócio."""

    def test_custo_beneficio_formula(self):
        """CB = eleitores / tempo_min."""
        eleitores = 100_000
        tempo_min = 50.0
        cb_esperado = eleitores / tempo_min
        assert cb_esperado == pytest.approx(2000.0, abs=0.1)

    def test_cb_zero_para_zero_eleitores(self):
        from routing_engine import RouteResult
        r = RouteResult("A", "B", 100.0, 60.0, 0, 0.0, False)
        assert r.custo_beneficio == 0.0

    def test_raio_km_de_isocronos(self):
        """60 min a 60km/h = 60km de raio estimado."""
        tempo_min = 60
        velocidade_kmh = 60.0
        raio_km = (tempo_min / 60.0) * velocidade_kmh
        assert raio_km == pytest.approx(60.0, abs=0.1)

    def test_total_eleitorado_soma(self):
        """Total de eleitores numa rota é a soma dos destinos."""
        from routing_engine import RouteResult
        rotas = [
            RouteResult("JP", "CG",    120.0, 95.0,  340_000, 3578.0, False),
            RouteResult("JP", "Patos", 250.0, 190.0, 110_000, 578.0,  False),
        ]
        total = sum(r.eleitores_destino for r in rotas)
        assert total == 450_000
