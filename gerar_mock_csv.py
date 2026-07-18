"""
gerar_mock_csv.py
==================
Script standalone para gerar o arquivo dados_paraiba.csv para testes imediatos.

Uso:
    python gerar_mock_csv.py

Output:
    dados_paraiba.csv  (na pasta atual)
    dados_paraiba_status_exemplo.xlsx  (para testar upload com status político)
    dados_paraiba_status_exemplo.pdf   (REQUER reportlab, opcional)
"""

import os
import sys

# Garante que o diretório do script está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etl_pipeline import carregar_base_mestra, salvar_base_mestra
import pandas as pd
import numpy as np


def gerar_csv_completo() -> None:
    print("📦 Gerando Base Mestra sintética...")
    gdf = carregar_base_mestra(usar_geobr=False, forcar_regerar=True)
    path = salvar_base_mestra(gdf, "dados_paraiba.csv")
    print(f"✅ dados_paraiba.csv gerado com {len(gdf)} municípios em → {path}")


def gerar_xlsx_status_exemplo() -> None:
    """
    Gera uma planilha de exemplo simulando um levantamento de campo,
    com erros propositais de digitação para testar o Fuzzy Matching.
    """
    dados_campo = [
        # Corretos
        {"Municipio": "João Pessoa",    "Status": "Aliado",    "Peso": 5},
        {"Municipio": "Campina Grande", "Status": "Aliado",    "Peso": 5},
        {"Municipio": "Patos",          "Status": "Aliado",    "Peso": 4},
        {"Municipio": "Sousa",          "Status": "Aliado",    "Peso": 4},
        {"Municipio": "Cajazeiras",     "Status": "Neutro",    "Peso": 3},
        # Com erros de digitação (para testar Fuzzy Matching)
        {"Municipio": "Joao Pessoa",    "Status": "Aliado",    "Peso": 5},  # sem acento
        {"Municipio": "Campina  Grande","Status": "Aliado",    "Peso": 5},  # espaço duplo
        {"Municipio": "Guarabira",      "Status": "Oposição",  "Peso": 2},
        {"Municipio": "santa rita",     "Status": "Aliado",    "Peso": 3},  # minúsculo
        {"Municipio": "Bayeoux",        "Status": "Neutro",    "Peso": 2},  # typo
        {"Municipio": "Monteiroo",      "Status": "Oposição",  "Peso": 1},  # typo
        {"Municipio": "Pombal",         "Status": "Aliado",    "Peso": 4},
        {"Municipio": "Princesa Isabel","Status": "Neutro",    "Peso": 3},
        {"Municipio": "Cabedelo",       "Status": "Aliado",    "Peso": 3},
        {"Municipio": "Sumé",           "Status": "Oposição",  "Peso": 2},
        {"Municipio": "Taperoá",        "Status": "Neutro",    "Peso": 2},
        {"Municipio": "Picui",          "Status": "Aliado",    "Peso": 3},  # sem acento
        {"Municipio": "Alagoa Grande",  "Status": "Aliado",    "Peso": 3},
        {"Municipio": "Mamanguapé",     "Status": "Neutro",    "Peso": 2},
        {"Municipio": "Soledade",       "Status": "Aliado",    "Peso": 4},
    ]

    df = pd.DataFrame(dados_campo)
    output_path = "dados_paraiba_status_exemplo.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Planilha de exemplo gerada → {output_path}")
    print("   ⚠️  Contém erros propositais para validar o Fuzzy Matching!")


def gerar_csv_status_exemplo() -> None:
    """Gera também em CSV para facilitar o upload."""
    dados_campo = [
        {"Municipio": "João Pessoa",    "Status": "Aliado",   "Peso": 5},
        {"Municipio": "Campina Grande", "Status": "Aliado",   "Peso": 5},
        {"Municipio": "Patos",          "Status": "Aliado",   "Peso": 4},
        {"Municipio": "Guarabira",      "Status": "Oposição", "Peso": 2},
        {"Municipio": "Bayeoux",        "Status": "Neutro",   "Peso": 2},
        {"Municipio": "Monteiroo",      "Status": "Oposição", "Peso": 1},
        {"Municipio": "santa rita",     "Status": "Aliado",   "Peso": 3},
        {"Municipio": "Pombal",         "Status": "Aliado",   "Peso": 4},
        {"Municipio": "Sousa",          "Status": "Aliado",   "Peso": 4},
        {"Municipio": "Picui",          "Status": "Aliado",   "Peso": 3},
    ]
    df = pd.DataFrame(dados_campo)
    output_path = "dados_paraiba_status_exemplo.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ CSV de status gerado → {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  GERADOR DE DADOS MOCK – Intel Eleitoral Paraíba")
    print("=" * 60)

    gerar_csv_completo()
    gerar_xlsx_status_exemplo()
    gerar_csv_status_exemplo()

    print("\n🎯 Próximo passo:")
    print("   1. Ative seu ambiente virtual: venv\\Scripts\\activate")
    print("   2. pip install -r requirements.txt")
    print("   3. streamlit run app.py")
    print("\n📌 Arquivos de teste gerados:")
    print("   • dados_paraiba.csv              (base completa)")
    print("   • dados_paraiba_status_exemplo.xlsx  (upload na sidebar)")
    print("   • dados_paraiba_status_exemplo.csv   (upload na sidebar)")
