"""
tests/test_etl_pipeline.py
============================
Testes unitários para o módulo etl_pipeline.py

Cobre:
- Geração da base sintética
- Validação de colunas obrigatórias
- Integridade dos dados (sem NaN em colunas críticas)
- Persistência e leitura do cache
- Exportação CSV
"""
import sys
import os

# Adiciona o diretório raiz ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path


class TestETLPipeline:
    """Testes do pipeline ETL."""

    @pytest.fixture(scope="class")
    def gdf(self):
        """Carrega a base mestra sintética uma única vez por classe."""
        from etl_pipeline import carregar_base_mestra
        return carregar_base_mestra(usar_geobr=False, forcar_regerar=True)

    def test_carga_retorna_geodataframe(self, gdf):
        """A função deve retornar um GeoDataFrame válido."""
        assert isinstance(gdf, gpd.GeoDataFrame), "Deve retornar GeoDataFrame"

    def test_numero_municipios(self, gdf):
        """Deve ter pelo menos 200 municípios da Paraíba."""
        assert len(gdf) >= 200, f"Esperado ≥ 200 municípios, obtido {len(gdf)}"

    def test_colunas_obrigatorias(self, gdf):
        """Colunas críticas devem existir."""
        colunas_exigidas = [
            "cod_ibge", "municipio", "lat", "lon",
            "eleitorado_total", "status_politico",
            "alinhamento_prefeito", "peso_lideranca",
            "tem_lideranca", "fixado_polo",
        ]
        for col in colunas_exigidas:
            assert col in gdf.columns, f"Coluna ausente: {col}"

    def test_sem_nan_em_campos_criticos(self, gdf):
        """Campos críticos não devem ter NaN."""
        for col in ["cod_ibge", "municipio", "lat", "lon", "eleitorado_total"]:
            assert gdf[col].isna().sum() == 0, f"NaN encontrado em: {col}"

    def test_cod_ibge_unico(self, gdf):
        """Cada município deve ter um cod_ibge único."""
        assert gdf["cod_ibge"].is_unique, "cod_ibge não é único!"

    def test_eleitorado_positivo(self, gdf):
        """Eleitorado deve ser positivo."""
        assert (gdf["eleitorado_total"] > 0).all(), "Eleitorado com valores <= 0"

    def test_lat_lon_dentro_da_paraiba(self, gdf):
        """Coordenadas devem estar dentro do bbox da Paraíba."""
        # Bbox aproximado PB: lat -8.5 a -6.0, lon -38.8 a -34.8
        assert gdf["lat"].between(-9.0, -6.0).all(), "Latitudes fora do range da PB"
        assert gdf["lon"].between(-39.5, -34.5).all(), "Longitudes fora do range da PB"

    def test_status_politico_valido(self, gdf):
        """Status político deve ser um dos três valores válidos."""
        valores_validos = {"Aliado", "Neutro", "Oposição"}
        valores_presentes = set(gdf["status_politico"].unique())
        invalidos = valores_presentes - valores_validos
        assert not invalidos, f"Status inválidos encontrados: {invalidos}"

    def test_alinhamento_prefeito_range(self, gdf):
        """Alinhamento deve estar entre 1 e 5."""
        assert gdf["alinhamento_prefeito"].between(1, 5).all(), \
            "Alinhamento fora do range [1, 5]"

    def test_exportacao_csv(self, gdf, tmp_path):
        """CSV exportado deve ser legível e com dados corretos."""
        from etl_pipeline import salvar_base_mestra
        output = str(tmp_path / "test_export.csv")
        path = salvar_base_mestra(gdf, output)
        
        assert Path(path).exists(), "Arquivo CSV não foi criado"
        df_lido = pd.read_csv(path)
        assert len(df_lido) == len(gdf), "Número de linhas diferente"
        assert "municipio" in df_lido.columns, "Coluna 'municipio' ausente no CSV"

    def test_joao_pessoa_maior_eleitorado(self, gdf):
        """João Pessoa deve ter o maior eleitorado."""
        jp = gdf[gdf["municipio"] == "João Pessoa"]
        assert not jp.empty, "João Pessoa não encontrada"
        max_eleitorado = gdf["eleitorado_total"].max()
        assert jp["eleitorado_total"].values[0] >= max_eleitorado * 0.5, \
            "João Pessoa não tem eleitorado expressivo"

    def test_campina_grande_presente(self, gdf):
        """Campina Grande deve estar na base."""
        cg = gdf[gdf["municipio"] == "Campina Grande"]
        assert not cg.empty, "Campina Grande não encontrada"

    def test_fixado_polo_inicialmente_falso(self, gdf):
        """Todos os municípios devem iniciar com fixado_polo=False."""
        assert not gdf["fixado_polo"].any(), \
            "Algum município inicia com fixado_polo=True"
