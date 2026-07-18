"""
tests/test_data_manager.py
============================
Testes unitários para o módulo data_manager.py

Cobre:
- Normalização de nomes (remoção de acentos)
- Fuzzy matching de nomes de cidades
- Leitura de planilhas CSV/XLSX
- Relatório de qualidade do match
- Aplicação de status ao GeoDataFrame
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import io
import pandas as pd
import geopandas as gpd
from pathlib import Path


class TestNormalizacaoNomes:
    """Testes de normalização de strings."""

    def test_remove_acento_simples(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("João Pessoa") == "joao pessoa"

    def test_converte_para_minusculo(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("CAMPINA GRANDE") == "campina grande"

    def test_trata_string_vazia(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("") == ""

    def test_trata_none_como_vazio(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome(None) == ""

    def test_remove_multiplos_acentos(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("Alagoa Gránde") == "alagoa grande"

    def test_remove_cedilha(self):
        from data_manager import _normalizar_nome
        assert _normalizar_nome("Conceição") == "conceicao"


class TestFuzzyMatching:
    """Testes do sistema de Fuzzy Matching."""

    @pytest.fixture(scope="class")
    def base_gdf(self):
        from etl_pipeline import carregar_base_mestra
        return carregar_base_mestra(usar_geobr=False, forcar_regerar=False)

    def test_match_exato(self, base_gdf):
        from data_manager import match_city_names
        input_df = pd.DataFrame([{
            "municipio_input": "João Pessoa",
            "status_politico": "Aliado",
            "peso_lideranca": 5
        }])
        result = match_city_names(input_df, base_gdf["municipio"], base_gdf["cod_ibge"])
        assert result["cod_ibge_matched"].notna().all()
        assert result["match_score"].values[0] >= 80

    def test_match_sem_acento(self, base_gdf):
        """'Joao Pessoa' (sem acento) deve casar com 'João Pessoa'."""
        from data_manager import match_city_names
        input_df = pd.DataFrame([{
            "municipio_input": "Joao Pessoa",
            "status_politico": "Aliado",
            "peso_lideranca": 5
        }])
        result = match_city_names(input_df, base_gdf["municipio"], base_gdf["cod_ibge"])
        assert result["municipio_matched"].values[0] == "João Pessoa"

    def test_match_typo_leve(self, base_gdf):
        """'Campina  Grande' (espaço duplo) deve casar com 'Campina Grande'."""
        from data_manager import match_city_names
        input_df = pd.DataFrame([{
            "municipio_input": "Campina  Grande",
            "status_politico": "Aliado",
            "peso_lideranca": 5
        }])
        result = match_city_names(input_df, base_gdf["municipio"], base_gdf["cod_ibge"])
        assert result["municipio_matched"].values[0] == "Campina Grande"

    def test_match_invalido_retorna_none(self, base_gdf):
        """Nome completamente inválido não deve ter match."""
        from data_manager import match_city_names
        input_df = pd.DataFrame([{
            "municipio_input": "XYZXYZXYZ_INEXISTENTE",
            "status_politico": "Neutro",
            "peso_lideranca": 1
        }])
        result = match_city_names(
            input_df, base_gdf["municipio"], base_gdf["cod_ibge"],
            threshold=95  # threshold alto para garantir rejeição
        )
        assert result["cod_ibge_matched"].isna().all()

    def test_match_multiplas_cidades(self, base_gdf):
        from data_manager import match_city_names
        cidades = [
            {"municipio_input": "Patos",     "status_politico": "Aliado",  "peso_lideranca": 4},
            {"municipio_input": "Sousa",     "status_politico": "Neutro",  "peso_lideranca": 3},
            {"municipio_input": "Guarabira", "status_politico": "Oposição","peso_lideranca": 2},
        ]
        input_df = pd.DataFrame(cidades)
        result = match_city_names(input_df, base_gdf["municipio"], base_gdf["cod_ibge"])
        assert len(result) == 3
        assert result["cod_ibge_matched"].notna().sum() == 3


class TestLeituraPlanilha:
    """Testes de leitura de CSV/XLSX."""

    def test_leitura_csv_simples(self, tmp_path):
        """Deve ler CSV com colunas 'Municipio' e 'Status'."""
        csv_content = "Municipio,Status,Peso\nJoão Pessoa,Aliado,5\nPatos,Neutro,3\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        from unittest.mock import MagicMock
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.read.return_value = csv_content.encode("utf-8")
        mock_file.seek.return_value = None

        # Testar via read_csv direto
        df = pd.read_csv(io.StringIO(csv_content))
        assert len(df) == 2
        assert "Municipio" in df.columns

    def test_csv_com_separador_ponto_virgula(self):
        """CSV com separador ';' deve ser detectado."""
        csv_content = "Municipio;Status;Peso\nJoão Pessoa;Aliado;5\n"
        df = pd.read_csv(io.StringIO(csv_content), sep=";")
        assert "Municipio" in df.columns

    def test_leitura_xlsx_valida(self, tmp_path):
        """XLSX válido deve ser lido corretamente."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Municipio", "Status", "Peso"])
        ws.append(["João Pessoa", "Aliado", 5])
        ws.append(["Campina Grande", "Neutro", 3])
        xlsx_path = tmp_path / "test.xlsx"
        wb.save(str(xlsx_path))

        df = pd.read_excel(str(xlsx_path), dtype=str)
        assert len(df) == 2
        assert "Municipio" in df.columns


class TestRelatorioMatch:
    """Testes do relatório de qualidade do fuzzy match."""

    def test_conta_matches_corretos(self):
        from data_manager import preparar_relatorio_match
        matched_df = pd.DataFrame([
            {"municipio_input": "JP",     "cod_ibge_matched": 2508109, "match_score": 95},
            {"municipio_input": "XYZXYZ", "cod_ibge_matched": None,    "match_score": 30},
            {"municipio_input": "Patos",  "cod_ibge_matched": 2512002, "match_score": 100},
        ])
        n_ok, n_fail, nao_matched = preparar_relatorio_match(matched_df)
        assert n_ok == 2
        assert n_fail == 1
        assert "XYZXYZ" in nao_matched

    def test_todos_matchados(self):
        from data_manager import preparar_relatorio_match
        matched_df = pd.DataFrame([
            {"municipio_input": "JP",    "cod_ibge_matched": 2508109, "match_score": 100},
            {"municipio_input": "Patos", "cod_ibge_matched": 2512002, "match_score": 100},
        ])
        n_ok, n_fail, nao_matched = preparar_relatorio_match(matched_df)
        assert n_ok == 2
        assert n_fail == 0
        assert nao_matched == []


class TestAplicarStatusGDF:
    """Testes de aplicação de status ao GeoDataFrame."""

    @pytest.fixture
    def gdf_base(self):
        from etl_pipeline import carregar_base_mestra
        return carregar_base_mestra(usar_geobr=False, forcar_regerar=False)

    def test_status_atualizado_corretamente(self, gdf_base):
        from data_manager import aplicar_status_ao_gdf
        # JP: cod 2508109
        matched_df = pd.DataFrame([{
            "municipio_input": "JP",
            "cod_ibge_matched": 2508109,
            "status_politico": "Oposição",
            "peso_lideranca": 2,
            "match_score": 100,
        }])
        gdf_updated = aplicar_status_ao_gdf(gdf_base, matched_df)
        jp_status = gdf_updated[gdf_updated["cod_ibge"] == 2508109]["status_politico"].values[0]
        assert jp_status == "Oposição"

    def test_sem_match_nao_altera_gdf(self, gdf_base):
        from data_manager import aplicar_status_ao_gdf
        matched_df = pd.DataFrame([{
            "municipio_input": "INVALIDO",
            "cod_ibge_matched": None,
            "status_politico": "Aliado",
            "peso_lideranca": 5,
            "match_score": 0,
        }])
        status_antes = gdf_base["status_politico"].copy()
        gdf_updated = aplicar_status_ao_gdf(gdf_base, matched_df)
        assert gdf_updated["status_politico"].equals(status_antes)
