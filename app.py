"""
app.py
=======
Interface Principal – Inteligência Logística Eleitoral (Paraíba)

Arquitetura:
- Sidebar: Upload de dados, configurações de roteirização, controle de setores.
- Main Area (topo): st.data_editor com tabela editável de municípios.
- Main Area (fundo): Mapa Folium interativo com coloração dinâmica.
- KPI Panel: Eleitores na rota, km previstos, municípios cobertos.

O mapa escuta o st.data_editor. Qualquer edição de status político
(Aliado → Oposição, etc.) recalcula as cores e, se ativado, os setores.

Autor: Arquitetura MVP – Inteligência Logística Eleitoral
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from data_manager import (
    SESSION_KEY_EDITED,
    SESSION_KEY_GDF,
    SESSION_KEY_ORS,
    SESSION_KEY_SETORES,
    STATUS_OPTIONS,
    aplicar_status_ao_gdf,
    inicializar_session_state,
    ler_pdf,
    ler_planilha,
    match_city_names,
    obter_cor_status,
    preparar_relatorio_match,
    sincronizar_editor_com_gdf,
    _gdf_para_editor_df,
)
from etl_pipeline import carregar_base_mestra
from political_algorithm import (
    STATUS_COLORS,
    Setor,
    calcular_score_polo,
    clusterizar_municipios,
    filtrar_satelites_no_raio,
)
from routing_engine import (
    ORSClient,
    RouteResult,
    calcular_isocronas_com_fallback,
    calcular_rota_custo_beneficio,
    gerar_url_google_maps,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ Intel Eleitoral – Paraíba",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS Global
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Fundos */
  .stApp {
    background: #f8fafc;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #f1f5f9;
    border-right: 1px solid #e2e8f0;
  }

  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4f46e5;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  /* Cards de KPI */
  .kpi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 0.5rem;
  }
  .kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.25);
  }
  .kpi-value {
    font-size: 1.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .kpi-label {
    font-size: 0.7rem;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.15rem;
  }

  /* Títulos */
  h1, h2, h3 {
    color: #1e293b !important;
  }

  /* Mapa container */
  .map-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  }

  /* Status badges */
  .badge-aliado   { background:#14532d; color:#4ade80; border-radius:8px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
  .badge-oposicao { background:#7f1d1d; color:#f87171; border-radius:8px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
  .badge-neutro   { background:#e2e8f0; color:#475569; border-radius:8px; padding:2px 10px; font-size:0.8rem; font-weight:600; }

  /* Botões */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
  }

  /* Data editor */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* Divider */
  hr { border-color: rgba(99,102,241,0.2); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Card de KPI
# ---------------------------------------------------------------------------
def kpi_card(valor: str, label: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-value">{valor}</div>
      <div class="kpi-label">{label}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Construção do Mapa Folium
# ---------------------------------------------------------------------------
def construir_mapa(
    gdf: gpd.GeoDataFrame,
    setores: List[Setor],
    setor_selecionado_id: Optional[int],
    isochrone_geojson: Optional[Dict],
    rotas: List[RouteResult],
    mostrar_clusters: bool = True,
    zoom_start: int = 8,
) -> folium.Map:
    """
    Constrói e retorna o mapa Folium com todas as camadas.

    Camadas:
    1. Marcadores coloridos por status político (Aliado/Neutro/Oposição).
    2. Polos marcados com estrela dourada.
    3. Polígono de isócrona (ORS ou fallback circular).
    4. Rota do dia selecionada.
    5. Tooltips com KPIs por município.
    """
    # Centro da Paraíba
    centro_lat = gdf["lat"].mean()
    centro_lon = gdf["lon"].mean()

    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom_start,
        tiles=None,
        prefer_canvas=True,
    )

    # Camada base: OpenStreetMap (estilo mapa de ruas clássico, semelhante ao Google Maps)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Mapa de Ruas",
        max_zoom=19,
    ).add_to(mapa)

    # Agrupar municípios por setor para coloring
    setor_cor_map: Dict[int, str] = {}
    polo_ids: set = set()
    for setor in setores:
        setor_cor_map[setor.id_setor] = setor.cor_hex
        polo_ids.add(setor.polo.get("cod_ibge"))

    # ------------------------------------------------------------------
    # Camada 1: Marcadores de municípios
    # ------------------------------------------------------------------
    cluster_group = folium.FeatureGroup(name="Municípios", show=mostrar_clusters)

    for _, row in gdf.iterrows():
        status = str(row.get("status_politico", "Neutro"))
        cod = int(row.get("cod_ibge", 0))
        nome = str(row.get("municipio", ""))
        lat = float(row.get("lat", 0))
        lon = float(row.get("lon", 0))
        eleitorado = int(row.get("eleitorado_total", 0))
        alinhamento = int(row.get("alinhamento_prefeito", 0))
        peso = int(row.get("peso_lideranca", 0))
        fixado = bool(row.get("fixado_polo", False))
        setor_id = int(row.get("setor", -1)) if "setor" in gdf.columns else -1

        # Determinar cor
        if fixado:
            cor_marcador = "#f59e0b"   # Âmbar – polo fixado
            icon_color = "orange"
            icon_type = "star"
        elif cod in polo_ids:
            cor_marcador = "#f59e0b"
            icon_color = "orange"
            icon_type = "star"
        else:
            cor_marcador = STATUS_COLORS.get(status, "#94a3b8")
            icon_color = (
                "green" if status == "Aliado"
                else "red" if status == "Oposição"
                else "gray"
            )
            icon_type = "circle"

        # Cor do setor (borda do marcador)
        setor_cor = setor_cor_map.get(setor_id, "#3b82f6")

        tooltip_html = f"""
        <div style="font-family:Inter,sans-serif;min-width:180px">
          <b style="font-size:1rem;color:{cor_marcador}">{nome}</b>
          <hr style="margin:4px 0;border-color:#334155">
          <span style="color:#94a3b8">Status: </span>
          <b style="color:{cor_marcador}">{status}</b><br>
          <span style="color:#94a3b8">Eleitorado: </span>
          <b style="color:#e2e8f0">{eleitorado:,}</b><br>
          <span style="color:#94a3b8">Alinhamento Prefeito: </span>
          <b style="color:#e2e8f0">{alinhamento}/5</b><br>
          <span style="color:#94a3b8">Peso Liderança: </span>
          <b style="color:#e2e8f0">{peso}/5</b><br>
          <span style="color:#94a3b8">Setor: </span>
          <b style="color:{setor_cor}">#{setor_id}</b>
          {"<br><b style='color:#f59e0b'>⭐ POLO FIXADO</b>" if fixado else ""}
          {"<br><b style='color:#f59e0b'>📍 CIDADE-POLO</b>" if cod in polo_ids and not fixado else ""}
        </div>
        """

        if cod in polo_ids or fixado:
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color=icon_color, icon='star'),
                tooltip=folium.Tooltip(tooltip_html, max_width=250),
            ).add_to(mapa)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=setor_cor,
                fill=True,
                fill_color=cor_marcador,
                fill_opacity=0.85,
                weight=2,
                tooltip=folium.Tooltip(tooltip_html, max_width=250),
            ).add_to(cluster_group)

    cluster_group.add_to(mapa)

    # ------------------------------------------------------------------
    # Camada 2: Isócrona
    # ------------------------------------------------------------------
    if isochrone_geojson:
        def iso_style(feature: Dict) -> Dict:
            return {
                "fillColor": "#6366f1",
                "fillOpacity": 0.18,
                "color": "#818cf8",
                "weight": 2.5,
                "dashArray": "6, 4",
            }

        folium.GeoJson(
            isochrone_geojson,
            name="Isócrona de Tempo",
            style_function=iso_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["value"],
                aliases=["Tempo (s):"],
                localize=True,
            ),
        ).add_to(mapa)

    # ------------------------------------------------------------------
    # Camada 3: Linhas de rota do dia
    # ------------------------------------------------------------------
    if rotas:
        rota_group = folium.FeatureGroup(name="Rota do Dia")
        polo_rota = rotas[0]
        polo_lat_r = float(gdf[gdf["municipio"] == polo_rota.origem]["lat"].iloc[0]) if len(gdf[gdf["municipio"] == polo_rota.origem]) > 0 else centro_lat
        polo_lon_r = float(gdf[gdf["municipio"] == polo_rota.origem]["lon"].iloc[0]) if len(gdf[gdf["municipio"] == polo_rota.origem]) > 0 else centro_lon

        for rota in rotas:
            dest_row = gdf[gdf["municipio"] == rota.destino]
            if dest_row.empty:
                continue
            dest_lat = float(dest_row["lat"].iloc[0])
            dest_lon = float(dest_row["lon"].iloc[0])

            # Gradiente de cor por custo-benefício
            cb_normalizado = min(rota.custo_beneficio / 1000.0, 1.0)
            r = int(255 * (1 - cb_normalizado))
            g = int(255 * cb_normalizado)
            linha_cor = f"#{r:02x}{g:02x}64"

            folium.PolyLine(
                locations=[[polo_lat_r, polo_lon_r], [dest_lat, dest_lon]],
                color=linha_cor,
                weight=2.5,
                opacity=0.8,
                tooltip=f"→ {rota.destino} | {rota.distancia_km}km | {rota.tempo_minutos:.0f}min | {rota.eleitores_destino:,} eleitores",
            ).add_to(rota_group)

        rota_group.add_to(mapa)

    # ------------------------------------------------------------------
    # Camada 4: Contornos dos setores (se há setores definidos)
    # ------------------------------------------------------------------
    if setores and len(setores) > 0:
        setores_group = folium.FeatureGroup(name="Setores", show=True)
        for setor in setores:
            polo = setor.polo
            sats = setor.satelites
            todos = [polo] + sats

            try:
                mask = gdf["setor"] == setor.id_setor
                if mask.sum() > 0:
                    poligono_unificado = gdf[mask].geometry.unary_union
                    
                    if poligono_unificado and not poligono_unificado.is_empty:
                        # Previne crash do Leaflet se for um tipo de geometria inválido para fill
                        if poligono_unificado.geom_type in ['Polygon', 'MultiPolygon']:
                            folium.GeoJson(
                            poligono_unificado.__geo_interface__,
                            style_function=lambda f, cor=setor.cor_hex: {
                                "fillColor": cor,
                                "fillOpacity": 0.25,
                                "color": cor,
                                "weight": 2,
                            },
                            tooltip=f"Setor #{setor.id_setor} | Polo: {setor.polo.get('municipio')} | {setor.total_eleitorado:,} eleitores",
                        ).add_to(setores_group)
            except Exception as e:
                logger.error(f"Erro ao desenhar setor {setor.id_setor}: {e}")

        setores_group.add_to(mapa)

    folium.LayerControl(collapsed=False).add_to(mapa)
    folium.plugins.Fullscreen(
        position="topright",
        title="Tela Cheia",
        title_cancel="Sair",
        force_separate_button=True,
    ).add_to(mapa)

    return mapa


# ---------------------------------------------------------------------------
# MAIN – Interface Streamlit
# ---------------------------------------------------------------------------
def main() -> None:

    # -------- Header --------
    st.markdown("""
    <div style="
      background: linear-gradient(90deg, rgba(99,102,241,0.2) 0%, rgba(139,92,246,0.15) 100%);
      border: 1px solid rgba(99,102,241,0.4);
      border-radius: 16px;
      padding: 1.2rem 2rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    ">
      <div style="font-size:2.5rem;">🗺️</div>
      <div>
        <h1 style="margin:0;font-size:1.6rem;font-weight:900;background:linear-gradient(90deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
          Inteligência Logística Eleitoral
        </h1>
        <p style="margin:0;color:#64748b;font-size:0.85rem;">
          Paraíba · Deputado Federal · Roteirização Estratégica com IA
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # -------- Carregar dados base --------
    with st.spinner("⚙️ Carregando base de municípios da Paraíba..."):
        gdf_base = carregar_base_mestra(usar_geobr=False)

    inicializar_session_state(gdf_base)

    # ======================================================
    # SIDEBAR
    # ======================================================
    with st.sidebar:
        with st.expander("📖 Guia de Preenchimento (Como usar)"):
            st.markdown("""
            **Passo a Passo:**
            1. **Importar Dados:** Suba uma planilha com o mapeamento político das cidades.
            2. **Clusterização:** Escolha a lógica (ex: geográfica ou afinidade) e quantos setores quer criar. Clique em 'Gerar Setores'.
            3. **Matriz (Tabela Central):** Edite ali mesmo se uma cidade virou 'Aliado' ou se a liderança mudou. O mapa e as cores atualizam na hora!
            4. **Painel de Rota:** Após gerar os setores, escolha o Setor do Dia e defina o tempo de viagem (Isócrona). O algoritmo mostrará as cidades alcançáveis e a melhor rota em custo-benefício.

            **Legenda de Status:**
            - 🟢 **Aliado:** Base consolidada do candidato.
            - 🔴 **Oposição:** Dominado por adversários.
            - ⚫ **Neutro:** Campo aberto para negociação.
            """)
        st.markdown("### 🔑 API & Conexão")
        ors_key = st.text_input(
            "API Key – OpenRouteService",
            type="password",
            placeholder="Insira sua chave ORS aqui",
            help="Obtenha gratuitamente em: openrouteservice.org/dev/#/signup",
        )

        if ors_key:
            if st.session_state.get(SESSION_KEY_ORS) is None or st.session_state.get("_ors_key") != ors_key:
                st.session_state[SESSION_KEY_ORS] = ORSClient(api_key=ors_key)
                st.session_state["_ors_key"] = ors_key
                st.success("✅ ORS conectado!")
        else:
            st.info("💡 Sem API Key: distâncias estimadas por Haversine.")
            st.session_state[SESSION_KEY_ORS] = None

        st.divider()

        # ------ Upload de Dados ------
        st.markdown("### 📂 Importar Dados de Campo")
        uploaded_file = st.file_uploader(
            "Upload CSV/XLSX/PDF",
            type=["csv", "xlsx", "xls", "pdf"],
            help="Planilha ou PDF com colunas: Município, Status (Aliado/Neutro/Oposição), Peso",
        )

        if uploaded_file:
            with st.spinner("🔍 Processando e aplicando Fuzzy Matching..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    input_df = ler_pdf(uploaded_file)
                else:
                    input_df = ler_planilha(uploaded_file)

                if input_df is not None:
                    matched_df = match_city_names(
                        input_df=input_df,
                        base_gdf_names=st.session_state[SESSION_KEY_GDF]["municipio"],
                        base_gdf_codes=st.session_state[SESSION_KEY_GDF]["cod_ibge"],
                    )
                    n_ok, n_fail, nao_matched = preparar_relatorio_match(matched_df)
                    st.success(f"✅ {n_ok} municípios mapeados com sucesso.")
                    if n_fail > 0:
                        st.warning(f"⚠️ {n_fail} não mapeados: {', '.join(nao_matched[:5])}{'...' if n_fail > 5 else ''}")

                    # Aplicar ao GDF
                    gdf_updated = aplicar_status_ao_gdf(
                        st.session_state[SESSION_KEY_GDF],
                        matched_df,
                    )
                    st.session_state[SESSION_KEY_GDF] = gdf_updated
                    st.session_state[SESSION_KEY_EDITED] = _gdf_para_editor_df(gdf_updated)
                    st.rerun()

        st.divider()

        # ------ Configurações de Setorização ------
        st.markdown("### 🧠 Clusterização Política")
        n_setores = st.slider(
            "Número de Setores",
            min_value=5,
            max_value=25,
            value=15,
            step=1,
            help="Define em quantos setores eleitorais o estado será dividido.",
        )

        # Estratégias do algoritmo
        estrategias_opcoes = [
            "Geográfica (Padrão)",
            "Equilíbrio de Eleitores",
            "Afinidade Política"
        ]
        estrategia_selecionada = st.selectbox(
            "Lógica da Divisão",
            options=estrategias_opcoes,
            help="Geográfica agrupa cidades vizinhas. Eleitores tenta manter o tamanho populacional parecido. Afinidade junta aliados."
        )

        # Polos fixados manualmente
        st.markdown("### 📌 Override Manual (Polos Fixados)")
        gdf_atual = st.session_state[SESSION_KEY_GDF]
        municipios_lista = sorted(gdf_atual["municipio"].tolist())

        polos_fixados_nomes = st.multiselect(
            "Fixar Cidades-Polo",
            options=municipios_lista,
            default=[],
            help="Estas cidades serão obrigatoriamente polos. O restante do estado é recalculado ao redor delas.",
        )

        polos_fixados_ids = gdf_atual[gdf_atual["municipio"].isin(polos_fixados_nomes)]["cod_ibge"].tolist()

        if st.button("⚡ Gerar / Atualizar Setores", use_container_width=True):
            with st.spinner("🔄 Executando K-Means político..."):
                gdf_clusterizado, setores = clusterizar_municipios(
                    gdf=st.session_state[SESSION_KEY_GDF],
                    n_setores=n_setores,
                    estrategia=estrategia_selecionada,
                    polos_fixados=polos_fixados_ids,
                )
                st.session_state[SESSION_KEY_GDF] = gdf_clusterizado
                st.session_state[SESSION_KEY_SETORES] = setores
                st.session_state[SESSION_KEY_EDITED] = _gdf_para_editor_df(gdf_clusterizado)
                st.success(f"✅ {len(setores)} setores gerados!")
                st.rerun()

        st.divider()

        # ------ Configurações de Roteamento ------
        st.markdown("### 🛣️ Roteirização")
        tempo_isocronos = st.slider(
            "Raio de Ação (tempo de viagem)",
            min_value=30,
            max_value=240,
            value=90,
            step=15,
            format="%d min",
        )

        setores_disponiveis = st.session_state.get(SESSION_KEY_SETORES, [])
        setor_selecionado_id: Optional[int] = None
        polo_selecionado = None
        satelites_no_raio: List[Dict] = []
        cidades_alvo_rota: List[Dict] = []

        if not setores_disponiveis:
            st.info("👆 Gere os setores primeiro para habilitar a roteirização.")
            st.button("🗺️ Traçar Rotas", disabled=True, use_container_width=True)
        else:
            st.markdown("### 🎯 Setor do Dia")
            opcoes_setores = {
                f"Setor #{s.id_setor} – {s.polo.get('municipio', '')} ({s.total_eleitorado:,} eleit.)": s.id_setor
                for s in setores_disponiveis
            }
            setor_escolhido_label = st.selectbox(
                "Selecione o Setor",
                options=list(opcoes_setores.keys()),
            )
            setor_selecionado_id = opcoes_setores[setor_escolhido_label]
            setor_obj = next((s for s in setores_disponiveis if s.id_setor == setor_selecionado_id), None)

            if setor_obj:
                polo_selecionado = setor_obj.polo
                st.markdown(f"**🏙️ Polo:** {polo_selecionado.get('municipio')}")
                st.markdown(f"**👥 Eleitorado Polo:** {int(polo_selecionado.get('eleitorado_total', 0)):,}")

                # Filtrar satélites pelo raio (convertendo tempo → km estimado a 65km/h)
                raio_km_estimado = (tempo_isocronos / 60.0) * 65.0
                satelites_no_raio, fora_raio = filtrar_satelites_no_raio(
                    polo_lat=float(polo_selecionado["lat"]),
                    polo_lon=float(polo_selecionado["lon"]),
                    satelites=setor_obj.satelites,
                    raio_km=raio_km_estimado,
                )

                if satelites_no_raio:
                    st.markdown(f"**📍 Cidades alcançáveis:** {len(satelites_no_raio)}")
                    cidades_alvo_rota = satelites_no_raio
                else:
                    st.warning("Nenhuma cidade alcançável neste tempo de viagem.")

            # Botão sempre visível, desabilitado se não houver cidades alvo
            pode_tracar = bool(polo_selecionado and cidades_alvo_rota)
            if st.button("🗺️ Traçar Rotas", disabled=not pode_tracar, use_container_width=True):
                with st.spinner("📐 Traçando as melhores rotas do dia..."):
                    ors_client = st.session_state.get(SESSION_KEY_ORS)
                    rotas_resultado = calcular_rota_custo_beneficio(
                        polo_nome=str(polo_selecionado.get("municipio", "")),
                        polo_lat=float(polo_selecionado["lat"]),
                        polo_lon=float(polo_selecionado["lon"]),
                        cidades_alvo=cidades_alvo_rota,
                        ors_client=ors_client,
                    )
                    st.session_state["rotas_resultado"] = rotas_resultado
                    st.rerun()

        st.divider()
        if st.button("🔄 Resetar Dados", use_container_width=True):
            for key in [SESSION_KEY_GDF, SESSION_KEY_EDITED, SESSION_KEY_SETORES, "rotas_resultado"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ======================================================
    # MAIN AREA
    # ======================================================
    gdf_atual = st.session_state[SESSION_KEY_GDF]
    setores_atuais = st.session_state.get(SESSION_KEY_SETORES, [])
    rotas_resultado: List[RouteResult] = st.session_state.get("rotas_resultado", [])

    # ------ KPIs ------
    total_eleitorado = int(gdf_atual["eleitorado_total"].sum())
    aliados = int((gdf_atual["status_politico"] == "Aliado").sum())
    neutros = int((gdf_atual["status_politico"] == "Neutro").sum())
    opositores = int((gdf_atual["status_politico"] == "Oposição").sum())
    n_setores_gerados = len(setores_atuais)

    eleitores_rota = sum(r.eleitores_destino for r in rotas_resultado)
    km_rota = sum(r.distancia_km for r in rotas_resultado)

    col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
    with col_k1:
        st.markdown(kpi_card(f"{total_eleitorado:,}", "Eleitorado Total PB"), unsafe_allow_html=True)
    with col_k2:
        st.markdown(kpi_card(str(aliados), "Municípios Aliados"), unsafe_allow_html=True)
    with col_k3:
        st.markdown(kpi_card(str(neutros), "Municípios Neutros"), unsafe_allow_html=True)
    with col_k4:
        st.markdown(kpi_card(str(opositores), "Municípios Oposição"), unsafe_allow_html=True)
    with col_k5:
        st.markdown(kpi_card(str(n_setores_gerados), "Setores Gerados"), unsafe_allow_html=True)
    with col_k6:
        st.markdown(kpi_card(f"{eleitores_rota:,}", "Eleit. na Rota do Dia"), unsafe_allow_html=True)

    st.divider()

    # ------ Data Editor ------
    st.markdown("#### 📋 Matriz de Municípios (Editável em Tempo Real)")
    st.caption("Edite o Status Político, Alinhamento e Peso diretamente na tabela. O mapa atualiza automaticamente.")

    editor_df = st.session_state.get(SESSION_KEY_EDITED, _gdf_para_editor_df(gdf_atual))

    edited_result = st.data_editor(
        editor_df,
        use_container_width=True,
        num_rows="fixed",
        height=300,
        column_config={
            "cod_ibge": st.column_config.NumberColumn("Cód. IBGE", disabled=True, width="small"),
            "municipio": st.column_config.TextColumn("Município", disabled=True, width="medium"),
            "eleitorado_total": st.column_config.NumberColumn("Eleitorado", format="%d", disabled=True),
            "votos_partido_2022": st.column_config.NumberColumn("Votos 2022", format="%d", disabled=True),
            "status_politico": st.column_config.SelectboxColumn(
                "Status Político",
                options=STATUS_OPTIONS,
                required=True,
                width="medium",
            ),
            "alinhamento_prefeito": st.column_config.NumberColumn(
                "Alinhamento (1-5)",
                min_value=1,
                max_value=5,
                step=1,
                width="small",
            ),
            "peso_lideranca": st.column_config.NumberColumn(
                "Peso Liderança (1-5)",
                min_value=1,
                max_value=5,
                step=1,
                width="small",
            ),
            "tem_lideranca": st.column_config.CheckboxColumn("Tem Liderança?", width="small"),
            "fixado_polo": st.column_config.CheckboxColumn("📌 Polo Fixado", width="small"),
            "indice_infraestrutura": st.column_config.NumberColumn(
                "Infra (0-1)",
                format="%.2f",
                disabled=True,
                width="small",
            ),
            "lat": st.column_config.NumberColumn("Lat", format="%.4f", disabled=True, width="small"),
            "lon": st.column_config.NumberColumn("Lon", format="%.4f", disabled=True, width="small"),
        },
        hide_index=True,
        key="data_editor_municipios",
    )

    # Sincronizar edições com o GDF
    if edited_result is not None:
        gdf_sincronizado = sincronizar_editor_com_gdf(edited_result, gdf_atual)

        # Verificar se houve mudança real (para evitar re-renders desnecessários)
        cols_check = ["status_politico", "alinhamento_prefeito", "peso_lideranca", "fixado_polo"]
        houve_mudanca = False
        for col in cols_check:
            if col in gdf_sincronizado.columns and col in gdf_atual.columns:
                if not gdf_sincronizado[col].equals(gdf_atual[col]):
                    houve_mudanca = True
                    break

        if houve_mudanca:
            st.session_state[SESSION_KEY_GDF] = gdf_sincronizado
            st.session_state[SESSION_KEY_EDITED] = edited_result
            gdf_atual = gdf_sincronizado

    st.divider()

    # ------ Mapa ------
    col_mapa, col_painel = st.columns([3, 1])

    with col_mapa:
        st.markdown("#### 🗺️ Mapa Estratégico Interativo")

        # Calcular isócrona se polo selecionado
        isochrone_geojson = None
        if polo_selecionado:
            ors_client = st.session_state.get(SESSION_KEY_ORS)
            isochrone_geojson = calcular_isocronas_com_fallback(
                lat=float(polo_selecionado["lat"]),
                lon=float(polo_selecionado["lon"]),
                time_minutes=tempo_isocronos,
                ors_client=ors_client,
            )

        mapa_folium = construir_mapa(
            gdf=gdf_atual,
            setores=setores_atuais,
            setor_selecionado_id=setor_selecionado_id,
            isochrone_geojson=isochrone_geojson,
            rotas=rotas_resultado,
            mostrar_clusters=True,
        )

        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(mapa_folium, use_container_width=True, height=580, returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)

        # Legenda
        st.markdown("""
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;font-size:0.75rem;color:#475569">
          <span>🟢 Aliado</span>
          <span>🔴 Oposição</span>
          <span>⚫ Neutro</span>
          <span>⭐ Cidade-Polo</span>
          <span>🟣 Isócrona de Tempo</span>
        </div>
        """, unsafe_allow_html=True)

    with col_painel:
        st.markdown("#### 📊 Painel de Rota")

        if polo_selecionado and cidades_alvo_rota:
            st.markdown(f"**📍 Municípios no Raio de {tempo_isocronos} min:**")
            st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:10px'>A partir do polo: {polo_selecionado.get('municipio')}</div>", unsafe_allow_html=True)
            for cidade in cidades_alvo_rota:
                status = cidade.get('status_politico', 'Neutro')
                cor_dot = STATUS_COLORS.get(status, '#64748b')
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.03);border-left:3px solid {cor_dot};padding:4px 8px;margin-bottom:4px;border-radius:0 4px 4px 0;font-size:0.8rem;display:flex;justify-content:space-between;align-items:center;">
                  <span>{cidade['municipio']}</span>
                  <span style="color:#334155;font-weight:600;font-size:0.75rem;">{int(cidade.get('eleitorado_total', 0)):,} el.</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

        if rotas_resultado:
            total_km = sum(r.distancia_km for r in rotas_resultado)
            total_min = sum(r.tempo_minutos for r in rotas_resultado)
            total_el_rota = sum(r.eleitores_destino for r in rotas_resultado)

            st.markdown(kpi_card(f"{total_km:.0f} km", "KM Totais na Rota"), unsafe_allow_html=True)
            st.markdown(kpi_card(f"{total_min:.0f} min", "Tempo Total Estimado"), unsafe_allow_html=True)
            st.markdown(kpi_card(f"{total_el_rota:,}", "Eleitores Cobertos"), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Ranking Custo-Benefício**")

            for i, rota in enumerate(rotas_resultado[:8]):
                cb_pct = min(rota.custo_beneficio / max(r.custo_beneficio for r in rotas_resultado), 1.0)
                bar_width = int(cb_pct * 100)
                st.markdown(f"""
                <div style="margin-bottom:0.6rem">
                  <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#334155">
                    <span>#{i+1} {rota.destino[:20]}</span>
                    <span style="color:#4f46e5">{rota.custo_beneficio:.0f} el/min</span>
                  </div>
                  <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-top:3px">
                    <div style="background:linear-gradient(90deg,#6366f1,#22c55e);width:{bar_width}%;height:100%;border-radius:4px"></div>
                  </div>
                  <div style="font-size:0.65rem;color:#64748b">{rota.distancia_km:.1f}km · {rota.tempo_minutos:.0f}min · {rota.eleitores_destino:,} eleit.</div>
                </div>
                """, unsafe_allow_html=True)

            # Botão Google Maps
            if rotas_resultado:
                todos_pontos = [(float(polo_selecionado["lat"]), float(polo_selecionado["lon"]))] + [
                    (float(gdf_atual[gdf_atual["municipio"] == r.destino]["lat"].iloc[0]),
                     float(gdf_atual[gdf_atual["municipio"] == r.destino]["lon"].iloc[0]))
                    for r in rotas_resultado
                    if not gdf_atual[gdf_atual["municipio"] == r.destino].empty
                ]
                google_url = gerar_url_google_maps(todos_pontos, [polo_selecionado.get("municipio", "")] + [r.destino for r in rotas_resultado])
                if google_url:
                    st.link_button("📱 Abrir Rota no Google Maps", url=google_url, use_container_width=True)

            # Tabela detalhada
            with st.expander("📋 Tabela Detalhada"):
                df_rotas = pd.DataFrame([r.to_dict() for r in rotas_resultado])
                st.dataframe(df_rotas, use_container_width=True, hide_index=True)

        else:
            st.info("👈 Gere os setores e calcule a rota do dia para ver o painel de custo-benefício.")

            # Sugestões de polos
            if setores_atuais:
                st.markdown("**Sugestões de Cidades-Polo:**")
                for setor in setores_atuais[:5]:
                    polo = setor.polo
                    score = calcular_score_polo(polo)
                    st.markdown(f"""
                    <div style="background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.15);border-radius:10px;padding:0.5rem 0.75rem;margin-bottom:0.4rem;font-size:0.8rem">
                      <b style="color:#2563eb">{polo.get('municipio')}</b>
                      <div style="color:#64748b">Setor #{setor.id_setor} · {setor.total_eleitorado:,} eleit.</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()

    # ------ Tabela de Setores ------
    if setores_atuais:
        st.markdown("#### 🗂️ Resumo dos Setores Gerados")
        resumo_setores = []
        for s in setores_atuais:
            aliados_s = sum(1 for sat in s.satelites if sat.get("status_politico") == "Aliado")
            resumo_setores.append({
                "Setor": f"#{s.id_setor}",
                "Cidade-Polo": s.polo.get("municipio", ""),
                "Eleitorado Polo": f"{int(s.polo.get('eleitorado_total', 0)):,}",
                "Municípios": s.total_municipios,
                "Eleitorado Total": f"{s.total_eleitorado:,}",
                "Aliados no Setor": aliados_s,
                "Alinhamento Polo": f"{int(s.polo.get('alinhamento_prefeito', 0))}/5",
                "Polo Fixado?": "✅" if s.polo.get("fixado_polo") else "🤖 IA",
            })

        df_resumo = pd.DataFrame(resumo_setores)
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)

        # Download CSV
        csv_bytes = df_resumo.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exportar Setores (CSV)",
            data=csv_bytes,
            file_name="setores_campanha_pb.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
