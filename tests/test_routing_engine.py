"""
tests/test_routing_engine.py
==============================
Testes unitários para o módulo routing_engine.py

Cobre:
- Cálculo Haversine (valores conhecidos)
- Estimativa de tempo
- Geração de URL Google Maps
- Fallback de isócrona circular
- RouteResult e custo-benefício
- calcular_rota_custo_beneficio (modo offline)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import math
from unittest.mock import MagicMock, patch


class TestHaversine:
    """Testes da fórmula de Haversine."""

    def test_mesma_cidade_distancia_zero(self):
        from routing_engine import haversine_km
        dist = haversine_km(-7.11, -34.86, -7.11, -34.86)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_joao_pessoa_campina_grande_aprox(self):
        """João Pessoa → Campina Grande ≈ 120 km em linha reta."""
        from routing_engine import haversine_km
        dist = haversine_km(-7.11, -34.86, -7.23, -35.88)
        assert 100 < dist < 150, f"Distância inesperada: {dist:.2f} km"

    def test_distancia_simetrica(self):
        from routing_engine import haversine_km
        d1 = haversine_km(-7.11, -34.86, -7.02, -37.28)
        d2 = haversine_km(-7.02, -37.28, -7.11, -34.86)
        assert d1 == pytest.approx(d2, abs=0.01)

    def test_patos_sousa_aprox(self):
        """Patos → Sousa ≈ 100 km em linha reta."""
        from routing_engine import haversine_km
        dist = haversine_km(-7.02, -37.28, -6.76, -38.23)
        assert 80 < dist < 130, f"Distância Patos→Sousa inesperada: {dist:.2f} km"


class TestTempoEstimado:
    """Testes de estimativa de tempo."""

    def test_100km_a_65kmh(self):
        from routing_engine import haversine_tempo_estimado
        tempo = haversine_tempo_estimado(100.0, 65.0)
        esperado = (100 / 65) * 60
        assert tempo == pytest.approx(esperado, abs=0.1)

    def test_distancia_zero_tempo_zero(self):
        from routing_engine import haversine_tempo_estimado
        tempo = haversine_tempo_estimado(0.0)
        assert tempo == pytest.approx(0.0, abs=0.001)

    def test_velocidade_customizada(self):
        from routing_engine import haversine_tempo_estimado
        tempo = haversine_tempo_estimado(120.0, 80.0)
        assert tempo == pytest.approx(90.0, abs=0.1)


class TestGoogleMapsURL:
    """Testes de geração de URL Google Maps."""

    def test_url_dois_pontos(self):
        from routing_engine import gerar_url_google_maps
        pontos = [(-7.11, -34.86), (-7.23, -35.88)]
        url = gerar_url_google_maps(pontos, ["João Pessoa", "Campina Grande"])
        assert url.startswith("https://www.google.com/maps/dir/")
        assert "-7.11" in url
        assert "driving" in url

    def test_url_rota_com_waypoints(self):
        from routing_engine import gerar_url_google_maps
        pontos = [(-7.11, -34.86), (-7.02, -37.28), (-7.23, -35.88)]
        url = gerar_url_google_maps(pontos, ["JP", "Patos", "CG"])
        assert "waypoints" in url

    def test_url_menos_de_2_pontos_retorna_vazio(self):
        from routing_engine import gerar_url_google_maps
        url = gerar_url_google_maps([(-7.11, -34.86)], ["JP"])
        assert url == ""

    def test_url_sem_pontos_retorna_vazio(self):
        from routing_engine import gerar_url_google_maps
        url = gerar_url_google_maps([], [])
        assert url == ""


class TestIsocranaFallback:
    """Testes do fallback circular para isócronas."""

    def test_fallback_retorna_geojson_valido(self):
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(
            lat=-7.11, lon=-34.86, time_minutes=60, ors_client=None
        )
        assert result is not None
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) > 0

    def test_fallback_tem_coordenadas(self):
        from routing_engine import calcular_isocronas_com_fallback
        result = calcular_isocronas_com_fallback(-7.11, -34.86, 90, ors_client=None)
        feature = result["features"][0]
        coords = feature["geometry"]["coordinates"][0]
        assert len(coords) > 10, "Polígono com poucos pontos"
        # Cada coord deve ser [lon, lat]
        for coord in coords:
            assert len(coord) == 2

    def test_fallback_maior_tempo_maior_poligono(self):
        """Isócrona de 120min deve ter área maior que de 60min."""
        from routing_engine import calcular_isocronas_com_fallback
        r60 = calcular_isocronas_com_fallback(-7.11, -34.86, 60, None)
        r120 = calcular_isocronas_com_fallback(-7.11, -34.86, 120, None)
        area60 = r60["features"][0]["properties"]["area"]
        area120 = r120["features"][0]["properties"]["area"]
        assert area120 > area60, "Isócrona maior deveria ter área maior"


class TestRouteResult:
    """Testes do objeto RouteResult."""

    def test_to_dict_contem_chaves(self):
        from routing_engine import RouteResult
        r = RouteResult(
            origem="JP", destino="CG",
            distancia_km=120.0, tempo_minutos=100.0,
            eleitores_destino=340000, custo_beneficio=3400.0,
            via_ors=False, google_maps_url="https://maps.google.com"
        )
        d = r.to_dict()
        assert "Origem" in d
        assert "Destino" in d
        assert "Distância (km)" in d
        assert "Eleitores/Min" in d

    def test_custo_beneficio_arredondado(self):
        from routing_engine import RouteResult
        r = RouteResult("JP", "CG", 120.0, 95.0, 340000, 3578.947, False)
        assert r.custo_beneficio == pytest.approx(3578.9, abs=0.2)


class TestCalcularRotaCustoBeneficio:
    """Testes do motor de custo-benefício em modo offline."""

    @pytest.fixture
    def polo_dados(self):
        return {"nome": "João Pessoa", "lat": -7.11, "lon": -34.86}

    @pytest.fixture
    def cidades_alvo(self):
        return [
            {"municipio": "Campina Grande", "lat": -7.23, "lon": -35.88, "eleitorado_total": 340000},
            {"municipio": "Patos",           "lat": -7.02, "lon": -37.28, "eleitorado_total": 110000},
            {"municipio": "Guarabira",       "lat": -6.86, "lon": -35.49, "eleitorado_total": 70000},
        ]

    def test_retorna_lista_resultados(self, polo_dados, cidades_alvo):
        from routing_engine import calcular_rota_custo_beneficio
        resultados = calcular_rota_custo_beneficio(
            polo_nome=polo_dados["nome"],
            polo_lat=polo_dados["lat"],
            polo_lon=polo_dados["lon"],
            cidades_alvo=cidades_alvo,
            ors_client=None,
        )
        assert len(resultados) == 3

    def test_ordenado_por_custo_beneficio(self, polo_dados, cidades_alvo):
        from routing_engine import calcular_rota_custo_beneficio
        resultados = calcular_rota_custo_beneficio(
            polo_nome=polo_dados["nome"],
            polo_lat=polo_dados["lat"],
            polo_lon=polo_dados["lon"],
            cidades_alvo=cidades_alvo,
            ors_client=None,
        )
        cbs = [r.custo_beneficio for r in resultados]
        assert cbs == sorted(cbs, reverse=True), "Resultados não estão ordenados por CB"

    def test_distancias_positivas(self, polo_dados, cidades_alvo):
        from routing_engine import calcular_rota_custo_beneficio
        resultados = calcular_rota_custo_beneficio(
            polo_nome=polo_dados["nome"],
            polo_lat=polo_dados["lat"],
            polo_lon=polo_dados["lon"],
            cidades_alvo=cidades_alvo,
            ors_client=None,
        )
        for r in resultados:
            assert r.distancia_km > 0
            assert r.tempo_minutos > 0

    def test_lista_vazia_retorna_vazio(self, polo_dados):
        from routing_engine import calcular_rota_custo_beneficio
        resultados = calcular_rota_custo_beneficio(
            polo_nome=polo_dados["nome"],
            polo_lat=polo_dados["lat"],
            polo_lon=polo_dados["lon"],
            cidades_alvo=[],
            ors_client=None,
        )
        assert resultados == []
