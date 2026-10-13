# -*- coding: utf-8 -*-
"""
EXPLORATÓRIO -- Volcano (limiar exploratório) + Enrichr + DGIdb
            para o cruzado idoso vs jovem (4 datasets, pós-ComBat)
======================================================================
Reaproveita o CSV bruto salvo pelo 04_sensibilidade_limiares.py
(DE_CRUZADO_idoso_vs_jovem_BRUTO_sem_limiar.csv) -- não recarrega o
checkpoint nem refaz o teste estatístico. Usa só pandas, matplotlib e
requests.

Ordem das etapas (valida se o sinal faz sentido biológico antes de mapear
para fármacos):
  1) Volcano com o limiar exploratório (|log2FC| >= 0,15; p-ajustado <= 0,01).
  2) Lista dos genes significativos, com a categoria funcional de cada um.
  3) Enrichr (Ma'ayan Lab) -- enriquecimento dos genes UP e DOWN
     separadamente, em GO Biological Process, KEGG e MSigDB Hallmark.
  4) DGIdb -- mapeamento gene -> fármaco.

Entrada:  resultados/DE_CRUZADO_idoso_vs_jovem_BRUTO_sem_limiar.csv
Saída:    resultados/ (lista de genes, ENRICHR_*.csv, FARMACOS_*.csv)
          resultados/figuras/exploratorio_limiar_015/

Gera a Figura 9 do TCC (análise de sensibilidade, 14 genes).
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DIR_RESULTADOS, DIR_FIGURAS, GENES_CATEGORIAS

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
OUTPUT_DIR = DIR_RESULTADOS
PLOTS_DIR = os.path.join(DIR_FIGURAS, "exploratorio_limiar_015")
os.makedirs(PLOTS_DIR, exist_ok=True)

BRUTO_PATH = os.path.join(OUTPUT_DIR, "DE_CRUZADO_idoso_vs_jovem_BRUTO_sem_limiar.csv")

LOG2FC_THRESHOLD = 0.15  # limiar exploratório, escolhido pela tabela de sensibilidade do passo 04
P_THRESHOLD = 0.01       # mesmo rigor estatístico do critério confirmatório (0.5)

PALETA = plt.cm.viridis
DGIDB_URL = 'https://dgidb.org/api/graphql'

ENRICHR_ADD_URL = 'https://maayanlab.cloud/Enrichr/addList'
ENRICHR_ENRICH_URL = 'https://maayanlab.cloud/Enrichr/enrich'
BIBLIOTECAS_ENRICHR = [
    'GO_Biological_Process_2023',
    'KEGG_2021_Human',
    'MSigDB_Hallmark_2020',
]

def categorizar_gene(gene):
    categorias = [cat for cat, lista in GENES_CATEGORIAS.items() if gene in lista]
    return ', '.join(categorias) if categorias else '-'


def plot_volcano(df_de, titulo, caminho_saida, log2fc_threshold, p_threshold, n_rotular=15):
    df = df_de.dropna(subset=['p_ajustado']).copy()
    if df.empty:
        print(f"   ⚠️ Sem genes com p-valor válido pra plotar ({titulo})")
        return

    menor_p_nao_zero = df.loc[df['p_ajustado'] > 0, 'p_ajustado'].min()
    df['p_plot'] = df['p_ajustado'].replace(0, menor_p_nao_zero)
    df['neg_log10_p'] = -np.log10(df['p_plot'])

    cores = {'UP': PALETA(0.85), 'DOWN': PALETA(0.15), 'ns': '#c7c7c7'}

    fig, ax = plt.subplots(figsize=(9, 7))
    for regulacao, cor in cores.items():
        subset = df[df['regulacao'] == regulacao]
        ax.scatter(subset['log2FC'], subset['neg_log10_p'], c=[cor], s=14, alpha=0.7,
                   label=f"{regulacao} ({len(subset)})")

    ax.axhline(-np.log10(p_threshold), color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(log2fc_threshold, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-log2fc_threshold, color='gray', linestyle='--', linewidth=0.8)

    destaques = df[df['regulacao'] != 'ns'].nsmallest(n_rotular, 'p_ajustado')
    for _, row in destaques.iterrows():
        ax.annotate(row['gene'], (row['log2FC'], row['neg_log10_p']),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('log2(Fold Change)')
    ax.set_ylabel('-log10(p-valor ajustado)')
    ax.set_title(titulo)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Volcano salvo: {os.path.basename(caminho_saida)}")


# ============================================================================
# ENRICHR -- validação biológica (Ma'ayan Lab)
# ============================================================================
def enrichr_add_list(genes, description='lista'):
    if not genes:
        return None
    genes_str = '\n'.join(genes)
    payload = {'list': (None, genes_str), 'description': (None, description)}
    try:
        response = requests.post(ENRICHR_ADD_URL, files=payload, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"   ❌ Erro ao enviar lista pro Enrichr: {e}")
        return None
    return response.json().get('userListId')


def enrichr_get_enrichment(user_list_id, library):
    if user_list_id is None:
        return pd.DataFrame()
    try:
        response = requests.get(
            ENRICHR_ENRICH_URL,
            params={'userListId': user_list_id, 'backgroundType': library},
            timeout=30
        )
        response.raise_for_status()
    except Exception as e:
        print(f"   ❌ Erro ao consultar enriquecimento ({library}): {e}")
        return pd.DataFrame()

    resultados = response.json().get(library, [])
    # formato de cada linha: [rank, termo, p_valor, z_score, combined_score,
    #                          genes_sobrepostos, p_ajustado, old_p, old_p_adj]
    linhas = []
    for r in resultados:
        linhas.append({
            'rank': r[0], 'termo': r[1], 'p_valor': r[2], 'z_score': r[3],
            'combined_score': r[4], 'genes_sobrepostos': ', '.join(r[5]),
            'p_ajustado': r[6]
        })
    return pd.DataFrame(linhas)


def rodar_enrichr(genes, nome_lista):
    print(f"\n🧬 Enrichr -- {nome_lista} ({len(genes)} genes)")
    if not genes:
        print("   ⚠️ Lista vazia -- pulando.")
        return {}
    if len(genes) < 5:
        print(f"   ⚠️ Só {len(genes)} genes -- Enrichr roda mesmo assim, mas resultado "
              "é estatisticamente frágil com listas tão pequenas (poucos termos vão "
              "sobreviver à correção de múltiplas comparações). Leia como indício, "
              "não como confirmação.")

    user_list_id = enrichr_add_list(genes, description=nome_lista)
    if user_list_id is None:
        print("   ❌ Não consegui enviar a lista pro Enrichr.")
        return {}
    time.sleep(1)  # Enrichr recomenda um respiro entre addList e enrich

    resultados = {}
    for lib in BIBLIOTECAS_ENRICHR:
        df_lib = enrichr_get_enrichment(user_list_id, lib)
        resultados[lib] = df_lib
        if df_lib.empty:
            print(f"\n   {lib}: (sem resultados)")
        else:
            top5 = df_lib.sort_values('p_ajustado').head(5)
            print(f"\n   📊 {lib} -- top termos (por p-ajustado):")
            for _, row in top5.iterrows():
                print(f"      [padj={row['p_ajustado']:.4f}] {row['termo']} "
                      f"-- genes: {row['genes_sobrepostos']}")
            caminho = os.path.join(OUTPUT_DIR, f"ENRICHR_{nome_lista}_{lib}.csv")
            df_lib.to_csv(caminho, index=False)
        time.sleep(0.5)  # evita bater rate limit entre bibliotecas

    return resultados


# ============================================================================
# DGIDB
# ============================================================================
def consultar_dgidb(genes):
    if not genes:
        return pd.DataFrame()
    query = """
    { genes(names: %s) { nodes { name interactions {
        drug { name conceptId } interactionScore
        interactionTypes { type directionality } sources { sourceDbName }
    } } } }
    """ % str(genes).replace("'", '"')

    try:
        response = requests.post(DGIDB_URL, json={'query': query}, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"   ❌ Erro na consulta ao DGIdb: {e}")
        return pd.DataFrame()

    data = response.json()
    linhas = []
    for gene_node in data.get('data', {}).get('genes', {}).get('nodes', []):
        for interacao in gene_node.get('interactions', []):
            drug = interacao.get('drug') or {}
            tipos = interacao.get('interactionTypes') or []
            fontes = interacao.get('sources') or []
            linhas.append({
                'gene': gene_node['name'],
                'farmaco': drug.get('name'),
                'interaction_score': interacao.get('interactionScore'),
                'tipos_interacao': ', '.join(f"{t.get('type')} ({t.get('directionality')})" for t in tipos if t),
                'fontes': ', '.join(f.get('sourceDbName', '') for f in fontes if f),
            })
    return pd.DataFrame(linhas)


def main():
    print("=" * 80)
    print(f"VOLCANO + ENRICHR + DGIdb -- CRUZADO IDOSO VS JOVEM, log2FC >= {LOG2FC_THRESHOLD}")
    print("=" * 80)

    if not os.path.exists(BRUTO_PATH):
        print(f"❌ Não encontrei: {BRUTO_PATH}")
        print("   Rode 04_sensibilidade_limiares.py primeiro -- ele gera esse CSV.")
        return

    df_de = pd.read_csv(BRUTO_PATH)
    print(f"\n📥 Carregado: {len(df_de)} genes (log2FC + p_ajustado já calculados)")

    # log2FC = mean_idoso - mean_jovem -> UP = sobe com a idade, DOWN = cai com a idade
    significativo = (df_de['log2FC'].abs() >= LOG2FC_THRESHOLD) & (df_de['p_ajustado'] <= P_THRESHOLD)
    df_de['regulacao'] = 'ns'
    df_de.loc[significativo & (df_de['log2FC'] > 0), 'regulacao'] = 'UP'
    df_de.loc[significativo & (df_de['log2FC'] < 0), 'regulacao'] = 'DOWN'

    # --- 1) Volcano com o limiar exploratório ---
    thr_str = str(LOG2FC_THRESHOLD).replace('.', '')
    caminho_volcano = os.path.join(PLOTS_DIR, f"volcano_cruzado_idoso_vs_jovem_thr{thr_str}.png")
    plot_volcano(
        df_de,
        f"Idoso vs Jovem (cruzado, batch-corrigido)\nlimiar exploratório log2FC≥{LOG2FC_THRESHOLD}",
        caminho_volcano, LOG2FC_THRESHOLD, P_THRESHOLD
    )

    # --- 2) Lista de genes significativos, com categoria funcional ---
    df_sig = df_de[df_de['regulacao'] != 'ns'].copy()
    df_sig['categoria'] = df_sig['gene'].apply(categorizar_gene)
    df_sig = df_sig.sort_values('p_ajustado')

    print(f"\n📋 {len(df_sig)} genes significativos (log2FC≥{LOG2FC_THRESHOLD}, padj≤{P_THRESHOLD}):")
    print(df_sig[['gene', 'log2FC', 'p_ajustado', 'regulacao', 'categoria']].to_string(index=False))

    caminho_lista = os.path.join(OUTPUT_DIR, f"DE_CRUZADO_idoso_vs_jovem_thr{thr_str}_lista.csv")
    df_sig.to_csv(caminho_lista, index=False)
    print(f"\n💾 Lista salva: {os.path.basename(caminho_lista)}")

    genes_up = sorted(df_sig[df_sig['regulacao'] == 'UP']['gene'].tolist())
    genes_down = sorted(df_sig[df_sig['regulacao'] == 'DOWN']['gene'].tolist())
    print(f"\n🔴 Genes UP com idade ({len(genes_up)}): {genes_up}")
    print(f"🟢 Genes DOWN com idade ({len(genes_down)}): {genes_down}")

    # --- 3) VALIDAÇÃO BIOLÓGICA -- Enrichr, ANTES do DGIdb ---
    print("\n" + "=" * 80)
    print("VALIDAÇÃO BIOLÓGICA -- Enrichr (Ma'ayan Lab)")
    print("=" * 80)
    print("Objetivo: checar se os termos enriquecidos têm cara de músculo/exercício/")
    print("envelhecimento antes de investir tempo no mapeamento farmacológico.")

    resultados_enrichr_up = rodar_enrichr(genes_up, f"UP_idade_thr{thr_str}")
    resultados_enrichr_down = rodar_enrichr(genes_down, f"DOWN_idade_thr{thr_str}")

    # --- 4) DGIdb -- só depois da checagem acima ---
    print("\n" + "=" * 80)
    print("MAPEAMENTO GENE -> FÁRMACO (DGIdb)")
    print("=" * 80)

    print("\n🔎 Consultando DGIdb -- genes UP com idade...")
    df_farm_up = consultar_dgidb(genes_up)
    if not df_farm_up.empty:
        df_farm_up = df_farm_up.sort_values('interaction_score', ascending=False)
        df_farm_up.to_csv(os.path.join(OUTPUT_DIR, f"FARMACOS_CRUZADO_UP_idade_thr{thr_str}.csv"), index=False)
        print(df_farm_up.to_string(index=False))
    else:
        print("   (sem resultados)")

    print("\n🔎 Consultando DGIdb -- genes DOWN com idade...")
    df_farm_down = consultar_dgidb(genes_down)
    if not df_farm_down.empty:
        df_farm_down = df_farm_down.sort_values('interaction_score', ascending=False)
        df_farm_down.to_csv(os.path.join(OUTPUT_DIR, f"FARMACOS_CRUZADO_DOWN_idade_thr{thr_str}.csv"), index=False)
        print(df_farm_down.to_string(index=False))
    else:
        print("   (sem resultados)")

    if not df_farm_up.empty and not df_farm_down.empty:
        f_up = set(df_farm_up['farmaco'].dropna())
        f_down = set(df_farm_down['farmaco'].dropna())
        ambos = f_up & f_down
        if ambos:
            print(f"\n⚠️ Fármacos com alvo nos dois eixos (efeito potencialmente ambíguo): {sorted(ambos)}")

    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()