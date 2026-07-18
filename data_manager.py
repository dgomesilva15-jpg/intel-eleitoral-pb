"""
data_manager.py
================
Gerenciador de Ingestão Dinâmica de Dados.

Responsabilidades:
- Ler planilhas XLSX/CSV com status político dos municípios.
- Extrair texto estruturado de PDFs (lista de municípios/status).
- Aplicar Fuzzy Matching para vincular nomes inconsistentes ao código IBGE.
- Gerenciar o estado da sessão Streamlit (Session State) para edições em tempo real.

Autor: Arquitetura MVP – Inteligência Logística Eleitoral
"""

from __future__ import annotations

import io
import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from thefuzz import fuzz, process  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 80          # Score mínimo para aceitar um match (0-100)
SESSION_KEY_GDF = "gdf_master"
SESSION_KEY_SETORES = "setores"
SESSION_KEY_EDITED = "edited_municipios"
SESSION_KEY_ORS = "ors_client"

STATUS_OPTIONS = ["Aliado", "Neutro", "Oposição"]
STATUS_COLORS = {
    "Aliado": "#22c55e",
    "Oposição": "#ef4444",
    "Neutro": "#94a3b8",
}

# Mapa de normalização de caracteres especiais para fuzzy matching
_TRANS_TABLE = str.maketrans(
    "áàâãäéèêëíìîïóòôõöúùûüçñÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇÑ",
    "aaaaaeeeeiiiiooooouuuucnAAAAAEEEEIIIIOOOOOUUUUCN",
)


def _normalizar_nome(nome: str) -> str:
    """Remove acentos, converte para minúsculas e normaliza espaços."""
    if not isinstance(nome, str):
        return ""
    return nome.strip().lower().translate(_TRANS_TABLE)


def _extrair_colunas_status(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Tenta identificar automaticamente as colunas de município e status
    em um DataFrame importado pelo usuário.

    Heurística: busca colunas cujo nome contenha palavras-chave.

    Returns:
        DataFrame padronizado com ['municipio_input', 'status_politico', 'peso_lideranca']
        ou None se não conseguir identificar.
    """
    col_municipio = None
    col_status = None
    col_peso = None

    for col in df.columns:
        col_n = _normalizar_nome(str(col))
        if any(k in col_n for k in ["municipio", "cidade", "city", "nome"]):
            col_municipio = col
        if any(k in col_n for k in ["status", "situacao", "situação", "aliado", "oposicao"]):
            col_status = col
        if any(k in col_n for k in ["peso", "lideranca", "liderança", "weight"]):
            col_peso = col

    if col_municipio is None:
        logger.warning("Coluna de município não identificada automaticamente.")
        return None

    result = pd.DataFrame()
    result["municipio_input"] = df[col_municipio].astype(str).str.strip()

    if col_status:
        result["status_politico"] = df[col_status].astype(str).str.strip()
    else:
        result["status_politico"] = "Neutro"

    if col_peso:
        result["peso_lideranca"] = pd.to_numeric(df[col_peso], errors="coerce").fillna(1).astype(int)
    else:
        result["peso_lideranca"] = 1

    return result


# ---------------------------------------------------------------------------
# Leitura de XLSX / CSV
# ---------------------------------------------------------------------------
def ler_planilha(uploaded_file: io.BytesIO) -> Optional[pd.DataFrame]:
    """
    Lê um arquivo XLSX ou CSV enviado via st.file_uploader.

    Args:
        uploaded_file: Objeto de arquivo do Streamlit.

    Returns:
        DataFrame padronizado ou None em caso de erro.
    """
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            raw_df = pd.read_excel(uploaded_file, dtype=str)
        elif filename.endswith(".csv"):
            # Tentar detectar o separador
            content = uploaded_file.read()
            uploaded_file.seek(0)
            sep = ";" if b";" in content[:500] else ","
            raw_df = pd.read_csv(uploaded_file, sep=sep, dtype=str)
        else:
            logger.error(f"Formato não suportado: {filename}")
            return None

        logger.info(f"Planilha lida: {len(raw_df)} linhas, colunas: {list(raw_df.columns)}")
        return _extrair_colunas_status(raw_df)

    except Exception as e:
        logger.error(f"Erro ao ler planilha: {e}")
        st.error(f"❌ Erro ao ler o arquivo: {e}")
        return None


# ---------------------------------------------------------------------------
# Leitura de PDF
# ---------------------------------------------------------------------------
def ler_pdf(uploaded_file: io.BytesIO) -> Optional[pd.DataFrame]:
    """
    Extrai texto de um PDF e tenta encontrar pares município–status.

    Estratégia:
    1. Usa pdfplumber para extrair texto linha por linha.
    2. Aplica regex para identificar padrões "Cidade: Status" ou tabelas simples.
    3. Retorna DataFrame padronizado.

    Args:
        uploaded_file: PDF enviado via st.file_uploader.

    Returns:
        DataFrame com ['municipio_input', 'status_politico', 'peso_lideranca'] ou None.
    """
    try:
        import pdfplumber  # type: ignore
        records: List[Dict] = []

        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = text.split("\n")

                for line in lines:
                    line = line.strip()
                    if not line or len(line) < 3:
                        continue

                    # Padrão: "Cidade - Status" ou "Cidade: Status" ou "Cidade | Status"
                    for sep in [" - ", ": ", " | ", "\t"]:
                        if sep in line:
                            parts = line.split(sep, maxsplit=1)
                            if len(parts) == 2:
                                nome = parts[0].strip()
                                resto = parts[1].strip()

                                # Detectar status
                                status = "Neutro"
                                for s in STATUS_OPTIONS:
                                    if s.lower() in resto.lower():
                                        status = s
                                        break

                                # Detectar peso (número de 1-5)
                                peso_match = re.search(r"\b([1-5])\b", resto)
                                peso = int(peso_match.group(1)) if peso_match else 1

                                if len(nome) >= 3:
                                    records.append({
                                        "municipio_input": nome,
                                        "status_politico": status,
                                        "peso_lideranca": peso,
                                    })
                                break

        if not records:
            logger.warning("Nenhum município identificado no PDF.")
            st.warning("⚠️ Não foi possível extrair municípios do PDF. Verifique o formato do arquivo.")
            return None

        df = pd.DataFrame(records).drop_duplicates(subset=["municipio_input"])
        logger.info(f"PDF processado: {len(df)} municípios encontrados.")
        return df

    except ImportError:
        st.error("❌ pdfplumber não instalado. Execute: pip install pdfplumber")
        return None
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}")
        st.error(f"❌ Erro ao processar PDF: {e}")
        return None


# ---------------------------------------------------------------------------
# Fuzzy Matching
# ---------------------------------------------------------------------------
def match_city_names(
    input_df: pd.DataFrame,
    base_gdf_names: pd.Series,
    base_gdf_codes: pd.Series,
    threshold: int = FUZZY_THRESHOLD,
) -> pd.DataFrame:
    """
    Vincula os nomes de municípios do input (possivelmente com erros de digitação)
    aos códigos IBGE oficiais usando Fuzzy Matching (thefuzz).

    Args:
        input_df: DataFrame com coluna 'municipio_input'.
        base_gdf_names: Series com nomes oficiais (IBGE).
        base_gdf_codes: Series com cod_ibge correspondente.
        threshold: Score mínimo de similaridade (0–100).

    Returns:
        DataFrame enriquecido com 'cod_ibge_matched', 'municipio_matched', 'match_score'.
    """
    nomes_oficiais = base_gdf_names.tolist()
    nomes_normalizados = [_normalizar_nome(n) for n in nomes_oficiais]
    codigo_por_nome = dict(zip(nomes_oficiais, base_gdf_codes.tolist()))

    resultados: List[Dict] = []

    for _, row in input_df.iterrows():
        nome_input = str(row.get("municipio_input", "")).strip()
        nome_norm = _normalizar_nome(nome_input)

        if not nome_norm:
            resultados.append({
                **row.to_dict(),
                "cod_ibge_matched": None,
                "municipio_matched": None,
                "match_score": 0,
            })
            continue

        # Fuzzy match contra nomes normalizados
        match_result = process.extractOne(
            nome_norm,
            nomes_normalizados,
            scorer=fuzz.token_sort_ratio,
        )

        if match_result and match_result[1] >= threshold:
            matched_norm = match_result[0]
            matched_idx = nomes_normalizados.index(matched_norm)
            matched_oficial = nomes_oficiais[matched_idx]
            cod_ibge = codigo_por_nome[matched_oficial]

            resultados.append({
                **row.to_dict(),
                "cod_ibge_matched": int(cod_ibge),
                "municipio_matched": matched_oficial,
                "match_score": match_result[1],
            })
        else:
            logger.warning(f"Match abaixo do threshold para '{nome_input}' (score: {match_result[1] if match_result else 0})")
            resultados.append({
                **row.to_dict(),
                "cod_ibge_matched": None,
                "municipio_matched": None,
                "match_score": match_result[1] if match_result else 0,
            })

    return pd.DataFrame(resultados)


def aplicar_status_ao_gdf(
    gdf: gpd.GeoDataFrame,
    matched_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Aplica os dados de status político (Aliado/Oposição/Neutro) e peso_lideranca
    do DataFrame matchado ao GeoDataFrame base.

    Apenas municípios que tiveram match bem-sucedido são atualizados.

    Args:
        gdf: GeoDataFrame base.
        matched_df: DataFrame resultado do match_city_names com match bem-sucedido.

    Returns:
        GeoDataFrame atualizado.
    """
    import geopandas as gpd_local

    gdf_updated = gdf.copy()
    valid_matches = matched_df.dropna(subset=["cod_ibge_matched"])

    atualizados = 0
    for _, row in valid_matches.iterrows():
        mask = gdf_updated["cod_ibge"] == int(row["cod_ibge_matched"])
        if mask.any():
            gdf_updated.loc[mask, "status_politico"] = str(row.get("status_politico", "Neutro"))
            gdf_updated.loc[mask, "peso_lideranca"] = int(row.get("peso_lideranca", 1))
            atualizados += 1

    logger.info(f"{atualizados} municípios atualizados via fuzzy match.")
    return gdf_updated


# ---------------------------------------------------------------------------
# Gerenciamento de Estado da Sessão Streamlit
# ---------------------------------------------------------------------------
def inicializar_session_state(gdf: "gpd.GeoDataFrame") -> None:
    """
    Inicializa o estado da sessão Streamlit com o GeoDataFrame base.
    Garante que edições não se percam ao interagir com widgets.

    Args:
        gdf: GeoDataFrame base da aplicação.
    """
    if SESSION_KEY_GDF not in st.session_state:
        st.session_state[SESSION_KEY_GDF] = gdf.copy()

    if SESSION_KEY_EDITED not in st.session_state:
        st.session_state[SESSION_KEY_EDITED] = _gdf_para_editor_df(gdf)

    if SESSION_KEY_SETORES not in st.session_state:
        st.session_state[SESSION_KEY_SETORES] = []

    if SESSION_KEY_ORS not in st.session_state:
        st.session_state[SESSION_KEY_ORS] = None


def _gdf_para_editor_df(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """
    Converte o GeoDataFrame para um DataFrame editável (sem geometria).
    Seleciona apenas as colunas relevantes para edição.
    """
    cols = [
        "cod_ibge", "municipio", "eleitorado_total", "votos_partido_2022",
        "status_politico", "alinhamento_prefeito", "peso_lideranca",
        "tem_lideranca", "fixado_polo", "indice_infraestrutura",
        "lat", "lon",
    ]
    cols_presentes = [c for c in cols if c in gdf.columns]
    return gdf[cols_presentes].copy()


def sincronizar_editor_com_gdf(
    edited_df: pd.DataFrame,
    gdf: "gpd.GeoDataFrame",
) -> "gpd.GeoDataFrame":
    """
    Sincroniza as edições feitas no st.data_editor de volta ao GeoDataFrame.

    Args:
        edited_df: DataFrame retornado pelo st.data_editor (com edições).
        gdf: GeoDataFrame base.

    Returns:
        GeoDataFrame atualizado com as edições aplicadas.
    """
    import geopandas as gpd_sync
    gdf_updated = gdf.copy()

    colunas_editaveis = [
        "status_politico", "alinhamento_prefeito", "peso_lideranca",
        "tem_lideranca", "fixado_polo",
    ]

    for col in colunas_editaveis:
        if col in edited_df.columns and col in gdf_updated.columns:
            # Mapear por cod_ibge para garantir consistência
            mapa = edited_df.set_index("cod_ibge")[col].to_dict()
            gdf_updated[col] = gdf_updated["cod_ibge"].map(mapa).fillna(gdf_updated[col])

    return gdf_updated


def obter_gdf_atualizado() -> Optional["gpd.GeoDataFrame"]:
    """
    Retorna o GeoDataFrame atual da sessão, sincronizado com as últimas edições.
    """
    if SESSION_KEY_GDF not in st.session_state:
        return None
    return st.session_state[SESSION_KEY_GDF]


def obter_cor_status(status: str) -> str:
    """Retorna a cor hex para um status político."""
    return STATUS_COLORS.get(status, STATUS_COLORS["Neutro"])


def preparar_relatorio_match(matched_df: pd.DataFrame) -> Tuple[int, int, List[str]]:
    """
    Prepara um relatório de qualidade do fuzzy matching para exibição.

    Returns:
        Tupla (n_matched, n_failed, lista_de_nao_matchados).
    """
    matched = matched_df.dropna(subset=["cod_ibge_matched"])
    failed = matched_df[matched_df["cod_ibge_matched"].isna()]
    nao_matchados = failed["municipio_input"].tolist()
    return len(matched), len(failed), nao_matchados
