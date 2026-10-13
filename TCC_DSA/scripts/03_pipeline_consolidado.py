# -*- coding: utf-8 -*-
"""
================================================================================
PASSO 03 -- Correção de lote (ComBat) e análises pós-correção
================================================================================
Autor: Julio Kazuaki

Pré-requisito: ter rodado 02_clustering_nao_supervisionado.py, que gera
MATRIZ_LOG2CPM.csv. Este script parte da matriz log2CPM e não refaz a
normalização.

Etapas, na ordem em que o main() as executa:
  A) DEGs dentro de cada dataset (grupos experimentais próprios de cada um,
     incluindo idade no GSE157585 e oleuropeína no GSE318937), com teste t
     independente ou pareado (mesmo sujeito) -> volcano plot por comparação.
  B) Correção de efeito de lote (ComBat) e comparação idoso vs jovem
     cruzando os datasets -> volcano, PCA, K-means e UMAP.
  D) Clusterização por categoria funcional (muscular, neuromuscular,
     oxigenação), antes e depois do ComBat, com ARI.
  C) Mapeamento gene -> fármaco (DGIdb) para genes do eixo "benéfico"
     (resposta ao treino) e "maléfico" (envelhecimento/inflamação).

Cada etapa pode ser ligada/desligada em MASTER_SWITCHES. Ao final da Etapa B
é salvo um checkpoint (_checkpoint_pos_etapaB.pkl), usado pelo
04_sensibilidade_limiares.py.

Entrada:  dados/processados/MATRIZ_LOG2CPM.csv
          metadados/METADADOS_ENRIQUECIDOS_v2.csv
Saída:    resultados/DE_*.csv, resultados/FARMACOS_*.csv
          resultados/figuras/03_pos_normalizacao/
          dados/processados/_checkpoint_pos_etapaB.pkl
================================================================================
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_DISPONIVEL = True
except ImportError:
    UMAP_DISPONIVEL = False

import requests

from config import (DIR_RESULTADOS, DIR_FIGURAS, MATRIZ_LOG2CPM, METADADOS,
                    CHECKPOINT_ETAPA_B, GENES_CATEGORIAS)

# ============================================================================
# 0. CONFIGURAÇÃO GERAL -- edite aqui, sem precisar mexer no resto do código
# ============================================================================
OUTPUT_DIR = DIR_RESULTADOS
PLOTS_DIR = os.path.join(DIR_FIGURAS, "03_pos_normalizacao")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Liga/desliga cada etapa do roteiro independentemente
MASTER_SWITCHES = {
    'A_degs_dentro_dataset': True,
    'B_cruzamento_datasets': True,
    'D_cluster_categoria': True,
    'C_mapeamento_farmaco': True,
}

# Datasets incluídos na ETAPA B (cruzamento entre datasets)
DATASETS_CRUZAMENTO = [
    'GSE277819',   # exercício (número de séries)
    'GSE157585',   # exercício (idoso/jovem) -- torna Grupo_Etario identificável
    'GSE318937',   # exercício em jovens -- oleuropeína vs placebo
    'GSE305038',   # exercício agudo (ativos/inativos)
]

# --- Critérios estatísticos ---
LOG2FC_THRESHOLD = 0.5
P_THRESHOLD = 0.01
CRITERIO_PREFERENCIAL = 'p_ajustado'
N_AMOSTRAL_MINIMO_PARA_AJUSTADO = 8

# Comparações DENTRO de cada dataset (Etapa A)
# formato: (dataset, coluna_grupo, grupo_referencia, grupo_teste, nome_comparacao)
COMPARACOES_DENTRO_DATASET = [
    # GSE157585 -- idoso vs jovem (alimenta FONTES_MALEFICO da Etapa C)
    ('GSE157585', 'Idade_x_Treino', 'Jovem adulto', 'Idoso_baseline', 'idoso_vs_jovem_157585'),
    ('GSE157585', 'Idade_x_Treino', 'Jovem adulto', 'Idoso_pos_treino', 'idoso_POS_TREINO_vs_jovem_157585'),

    # GSE305038 -- pré vs 4h pós-exercício
    ('GSE305038', 'Caracteristicas_grupo', 'Pre_treino', '4h_Pos_exercicio', 'pre_vs_pos_305038'),

    # GSE318937 -- oleuropeína vs placebo (tratamento nutracêutico)
    ('GSE318937', 'Tratamento', 'Placebo', 'Oleuropeina', 'OLE_vs_placebo_318937'),

    # GSE277819 -- comparações de número de séries
    ('GSE277819', 'Caracteristicas_grupo', 'Pre_treino', 'Serie_unica', 'pre_vs_1serie'),
    ('GSE277819', 'Caracteristicas_grupo', 'Pre_treino', 'quatro_series', 'pre_vs_4series'),
    ('GSE277819', 'Caracteristicas_grupo', 'Serie_unica', 'quatro_series', '1serie_vs_4series'),
]

# ----------------------------------------------------------------------
# Comparações PAREADAS (mesmo sujeito em ambos os grupos) -- usam
# Sujeito_ID do METADADOS_ENRIQUECIDOS_v2.csv. Rodam com ttest_rel em vez
# de ttest_ind, que é estatisticamente mais forte pra desenho
# within-subject (cancela a variância entre-sujeitos).
#
# formato: (dataset, coluna_grupo, valor1, valor2, nome_comp,
#           coluna_sujeito, coluna_chave_extra_ou_None)
#
# coluna_chave_extra: usada quando um sujeito pode ter MAIS DE UM par
# possível (ex: GSE305038 tem par Pré/Pós dentro de CADA condição
# Ativo/Inativo; GSE318937 tem par Placebo/Oleuropeína dentro de CADA
# timepoint Pre/Post/Post24h) -- sem isso, o pareamento juntaria
# condições/timepoints diferentes por engano.
COMPARACOES_PAREADAS = [
    ('GSE157585', 'Biopsia_Timepoint', 1, 3, 'baseline_vs_pos_treino_157585_PAREADO', 'Sujeito_ID', None),
    ('GSE277819', 'Caracteristicas_grupo', 'Pre_treino', 'Serie_unica', 'pre_vs_1serie_PAREADO', 'Sujeito_ID', None),
    ('GSE277819', 'Caracteristicas_grupo', 'Pre_treino', 'quatro_series', 'pre_vs_4series_PAREADO', 'Sujeito_ID', None),
    ('GSE277819', 'Caracteristicas_grupo', 'Serie_unica', 'quatro_series', '1serie_vs_4series_PAREADO', 'Sujeito_ID', None),
    ('GSE305038', 'Caracteristicas_grupo', 'Pre_treino', '4h_Pos_exercicio', 'pre_vs_pos_305038_PAREADO', 'Sujeito_ID', 'Condicao_Atividade'),
    # GSE318937: o mesmo sujeito recebeu Oleuropeína e Placebo (crossover).
    # A coluna extra evita parear Placebo-Pre com Oleuropeina-Post do mesmo
    # sujeito por engano.
    ('GSE318937', 'Tratamento', 'Placebo', 'Oleuropeina', 'OLE_vs_placebo_318937_PAREADO', 'Sujeito_ID', 'Caracteristicas_grupo'),
]

# GENES_CATEGORIAS (clusterização por categoria e mapeamento de fármaco) vem de config.py


# ============================================================================
# 1. CARREGAMENTO (robusto a nome de coluna com espaço vs underscore)
# ============================================================================
def carregar_matriz_e_metadados():
    print("\n📥 CARREGANDO DADOS...")
    expr_path = MATRIZ_LOG2CPM
    meta_path = METADADOS

    df_expr = pd.read_csv(expr_path, index_col=0)
    df_meta = pd.read_csv(meta_path, sep=';', index_col=0, encoding='latin1')

    for col in df_meta.select_dtypes(include='object').columns:
        df_meta[col] = df_meta[col].str.strip()

    if 'Caracteristicas do grupo' in df_meta.columns and 'Caracteristicas_grupo' not in df_meta.columns:
        df_meta = df_meta.rename(columns={'Caracteristicas do grupo': 'Caracteristicas_grupo'})
        print("   ℹ️ Coluna 'Caracteristicas do grupo' renomeada para 'Caracteristicas_grupo'")

    if 'Biopsia_Timepoint' in df_meta.columns and 'Grupo_Etario' in df_meta.columns:
        df_meta['Idade_x_Treino'] = df_meta['Grupo_Etario']
        mask_idoso_pos = (df_meta['Grupo_Etario'] == 'Idoso') & (df_meta['Biopsia_Timepoint'] == 3)
        mask_idoso_pre = (df_meta['Grupo_Etario'] == 'Idoso') & (df_meta['Biopsia_Timepoint'] == 1)
        df_meta.loc[mask_idoso_pos, 'Idade_x_Treino'] = 'Idoso_pos_treino'
        df_meta.loc[mask_idoso_pre, 'Idade_x_Treino'] = 'Idoso_baseline'

    # ------------------------------------------------------------------
    # Eixo Pré/Pós unificado entre os datasets de exercício -- diferente
    # de Grupo_Etario, aqui vários datasets contribuem tanto Pré quanto Pós,
    # então esse eixo não é confundido 1:1 com Dataset. Ressalva: mistura
    # escalas de tempo diferentes sob "Pós" (treino crônico x sessão aguda).
    #
    # Se a coluna já existir no CSV (dummy manual: 1=Pós, 0=Pré,
    # vazio=excluído -- Post imediato e Serie_unica ficam de fora), ela é
    # respeitada e só convertida para 'Pre'/'Pos'. Só é calculada do zero
    # se a coluna não existir no CSV.
    # ------------------------------------------------------------------
    if 'Momento_Treino_Cruzado' in df_meta.columns:
        print("   ℹ️ Coluna 'Momento_Treino_Cruzado' já existe no CSV (manual) -- usando ela, "
              "sem recalcular.")
        mapa_dummy = {1: 'Pos', 1.0: 'Pos', '1': 'Pos', 0: 'Pre', 0.0: 'Pre', '0': 'Pre'}
        df_meta['Momento_Treino_Cruzado'] = df_meta['Momento_Treino_Cruzado'].map(mapa_dummy)
    else:
        df_meta['Momento_Treino_Cruzado'] = pd.Series(np.nan, index=df_meta.index, dtype='object')
        if 'Biopsia_Timepoint' in df_meta.columns:
            df_meta.loc[df_meta['Biopsia_Timepoint'] == 1, 'Momento_Treino_Cruzado'] = 'Pre'
            df_meta.loc[df_meta['Biopsia_Timepoint'] == 3, 'Momento_Treino_Cruzado'] = 'Pos'
        if 'Caracteristicas_grupo' in df_meta.columns:
            df_meta.loc[df_meta['Caracteristicas_grupo'] == 'Pre_treino', 'Momento_Treino_Cruzado'] = 'Pre'
            df_meta.loc[df_meta['Caracteristicas_grupo'] == 'quatro_series', 'Momento_Treino_Cruzado'] = 'Pos'
            df_meta.loc[df_meta['Caracteristicas_grupo'] == 'Pre', 'Momento_Treino_Cruzado'] = 'Pre'
            df_meta.loc[df_meta['Caracteristicas_grupo'] == 'Post24h', 'Momento_Treino_Cruzado'] = 'Pos'
            # 'Post' (imediato) e 'Serie_unica' ficam de fora do binário

    n_pre = (df_meta['Momento_Treino_Cruzado'] == 'Pre').sum()
    n_pos = (df_meta['Momento_Treino_Cruzado'] == 'Pos').sum()
    print(f"   ℹ️ Momento_Treino_Cruzado (Pré/Pós unificado): {n_pre} Pré | {n_pos} Pós "
          f"| {df_meta['Momento_Treino_Cruzado'].isna().sum()} fora do binário (Post imediato/Serie_unica)")

    print(f"✅ Matriz: {df_expr.shape[0]} genes × {df_expr.shape[1]} amostras")
    print(f"✅ Metadados: {df_meta.shape[0]} amostras")
    print(f"   Datasets presentes: {', '.join(sorted(df_meta['Dataset'].dropna().unique()))}")

    return df_expr, df_meta


# ============================================================================
# 2. ENGINE DE DE VETORIZADA (compartilhada por todas as comparações)
# ============================================================================
def run_de_vetorizado(df_expr, amostras_c1, amostras_c2, nome_c1='c1', nome_c2='c2'):
    v1 = df_expr[amostras_c1].values
    v2 = df_expr[amostras_c2].values

    mean_c1 = v1.mean(axis=1)
    mean_c2 = v2.mean(axis=1)
    log2fc = mean_c2 - mean_c1
    t_stat, p_val = stats.ttest_ind(v2, v1, axis=1)

    df_de = pd.DataFrame({
        'gene': df_expr.index,
        'log2FC': log2fc,
        'FC': 2 ** np.abs(log2fc),
        f'mean_{nome_c1}': mean_c1,
        f'mean_{nome_c2}': mean_c2,
        'p_valor': p_val
    })

    p_vals = df_de['p_valor'].values
    mask_valido = ~np.isnan(p_vals)
    n_nan = (~mask_valido).sum()
    p_ajustado = np.full_like(p_vals, np.nan)
    if mask_valido.sum() > 0:
        p_ajustado[mask_valido] = multipletests(p_vals[mask_valido], method='fdr_bh')[1]
    df_de['p_ajustado'] = p_ajustado
    if n_nan > 0:
        print(f"   ℹ️ {n_nan} genes com variância zero -- excluídos da correção FDR")

    n_min = min(len(amostras_c1), len(amostras_c2))
    coluna_p = CRITERIO_PREFERENCIAL
    if n_min < N_AMOSTRAL_MINIMO_PARA_AJUSTADO:
        print(f"   ⚠️ Amostra pequena (n_min={n_min}) -- considere conferir também com 'p_valor' bruto")

    df_de['significant'] = (np.abs(df_de['log2FC']) >= LOG2FC_THRESHOLD) & (df_de[coluna_p] <= P_THRESHOLD)
    df_de['regulacao'] = 'ns'
    df_de.loc[df_de['significant'] & (df_de['log2FC'] > 0), 'regulacao'] = 'UP'
    df_de.loc[df_de['significant'] & (df_de['log2FC'] < 0), 'regulacao'] = 'DOWN'

    return df_de


def run_de_pareado(df_expr, df_meta, dataset, coluna, valor1, valor2,
                    coluna_sujeito='Sujeito_ID', coluna_chave_extra=None,
                    nome_c1='c1', nome_c2='c2'):
    df_ds = df_meta[df_meta['Dataset'] == dataset].copy()
    if coluna not in df_ds.columns or coluna_sujeito not in df_ds.columns:
        print(f"   ⚠️ Coluna '{coluna}' ou '{coluna_sujeito}' não existe -- pulando.")
        return None

    if coluna_chave_extra is not None and coluna_chave_extra in df_ds.columns:
        df_ds['_chave_pareamento'] = (df_ds[coluna_sujeito].astype(str) + '||' +
                                       df_ds[coluna_chave_extra].astype(str))
    else:
        df_ds['_chave_pareamento'] = df_ds[coluna_sujeito].astype(str)

    g1 = df_ds[df_ds[coluna] == valor1].dropna(subset=['_chave_pareamento'])
    g2 = df_ds[df_ds[coluna] == valor2].dropna(subset=['_chave_pareamento'])
    g1 = g1[g1.index.isin(df_expr.columns)]
    g2 = g2[g2.index.isin(df_expr.columns)]

    g1 = g1[~g1['_chave_pareamento'].duplicated(keep='first')].set_index('_chave_pareamento')
    g2 = g2[~g2['_chave_pareamento'].duplicated(keep='first')].set_index('_chave_pareamento')
    chaves_comuns = sorted(set(g1.index) & set(g2.index))

    print(f"   • {valor1}: {len(g1)} | {valor2}: {len(g2)} | pares completos: {len(chaves_comuns)}")
    if len(chaves_comuns) < 3:
        print("   ⚠️ Menos de 3 pares completos -- pulando (ttest_rel precisa de pares).")
        return None

    g1_idx = df_ds[df_ds[coluna] == valor1].dropna(subset=['_chave_pareamento'])
    g1_idx = g1_idx[g1_idx.index.isin(df_expr.columns)]
    g1_idx = g1_idx[~g1_idx['_chave_pareamento'].duplicated(keep='first')]
    g2_idx = df_ds[df_ds[coluna] == valor2].dropna(subset=['_chave_pareamento'])
    g2_idx = g2_idx[g2_idx.index.isin(df_expr.columns)]
    g2_idx = g2_idx[~g2_idx['_chave_pareamento'].duplicated(keep='first')]

    mapa_c1 = dict(zip(g1_idx['_chave_pareamento'], g1_idx.index))
    mapa_c2 = dict(zip(g2_idx['_chave_pareamento'], g2_idx.index))
    amostras_c1 = [mapa_c1[k] for k in chaves_comuns]
    amostras_c2 = [mapa_c2[k] for k in chaves_comuns]

    v1 = df_expr[amostras_c1].values
    v2 = df_expr[amostras_c2].values

    mean_c1 = v1.mean(axis=1)
    mean_c2 = v2.mean(axis=1)
    log2fc = mean_c2 - mean_c1
    t_stat, p_val = stats.ttest_rel(v2, v1, axis=1)

    df_de = pd.DataFrame({
        'gene': df_expr.index,
        'log2FC': log2fc,
        'FC': 2 ** np.abs(log2fc),
        f'mean_{nome_c1}': mean_c1,
        f'mean_{nome_c2}': mean_c2,
        'p_valor': p_val
    })

    p_vals = df_de['p_valor'].values
    mask_valido = ~np.isnan(p_vals)
    n_nan = (~mask_valido).sum()
    p_ajustado = np.full_like(p_vals, np.nan)
    if mask_valido.sum() > 0:
        p_ajustado[mask_valido] = multipletests(p_vals[mask_valido], method='fdr_bh')[1]
    df_de['p_ajustado'] = p_ajustado
    if n_nan > 0:
        print(f"   ℹ️ {n_nan} genes com variância zero -- excluídos da correção FDR")

    coluna_p = CRITERIO_PREFERENCIAL
    df_de['significant'] = (np.abs(df_de['log2FC']) >= LOG2FC_THRESHOLD) & (df_de[coluna_p] <= P_THRESHOLD)
    df_de['regulacao'] = 'ns'
    df_de.loc[df_de['significant'] & (df_de['log2FC'] > 0), 'regulacao'] = 'UP'
    df_de.loc[df_de['significant'] & (df_de['log2FC'] < 0), 'regulacao'] = 'DOWN'
    df_de.attrs['amostras_c1'] = amostras_c1
    df_de.attrs['amostras_c2'] = amostras_c2

    return df_de


# ============================================================================
# 3. CORREÇÃO DE BATCH (vetorizada)
# ============================================================================
def correct_batch_effect(df_expr, df_meta, batch_col='Dataset'):
    print(f"\n🔄 CORRIGINDO BATCH EFFECT -- mean-centering ingênuo ({batch_col})...")
    batch_labels = df_meta.loc[df_expr.columns, batch_col]
    batch_means_matrix = df_expr.T.groupby(batch_labels).transform('mean').T
    global_mean = df_expr.mean(axis=1)
    df_corrected = df_expr.sub(batch_means_matrix).add(global_mean, axis=0)
    print("   ✅ Correção concluída!")
    return df_corrected


try:
    from combat.pycombat import pycombat as pycombat_norm
    COMBAT_DISPONIVEL = True
except ImportError:
    COMBAT_DISPONIVEL = False
    print("⚠️ pacote 'combat' não encontrado -- rode: pip install combat")


def checar_confundimento(df_meta, amostras, batch_col, covar_col):
    batch = df_meta.loc[amostras, batch_col].astype(str)
    covar = df_meta.loc[amostras, covar_col].astype(str)
    tabela = pd.crosstab(batch, covar)
    n_niveis_por_batch = tabela.gt(0).sum(axis=1)
    identificavel = (n_niveis_por_batch > 1).any()
    if not identificavel:
        print(f"   🚫 '{covar_col}' é TOTALMENTE confundida com '{batch_col}':")
        print(tabela.to_string())
        print(f"      Cada nível de '{covar_col}' existe em um único batch --")
        print("      impossível distinguir efeito biológico de efeito técnico.")
    else:
        print(f"   ✅ '{covar_col}' varia dentro de pelo menos um '{batch_col}' -- identificável:")
        print(tabela.to_string())
    return identificavel, tabela


def correct_batch_effect_combat(df_expr, df_meta, batch_col='Dataset', proteger_col=None):
    if not COMBAT_DISPONIVEL:
        raise ImportError("Instale: pip install combat")

    amostras = list(df_expr.columns)
    batch = df_meta.loc[amostras, batch_col].astype(str)

    if proteger_col is not None and proteger_col in df_meta.columns:
        checar_confundimento(df_meta, amostras, batch_col, proteger_col)

    print(f"\n🔄 CORRIGINDO BATCH EFFECT -- ComBat/Empirical Bayes ({batch_col})...")
    df_corrected = pycombat_norm(df_expr, batch)
    print("   ✅ Correção concluída!")
    return df_corrected


# ============================================================================
# 4. FUNÇÕES DE PLOT COMPARTILHADAS (paleta viridis)
# ============================================================================
PALETA = plt.cm.viridis


def plot_volcano(df_de, titulo, caminho_saida, n_rotular=10):
    df = df_de.dropna(subset=['p_ajustado']).copy()
    if df.empty:
        print(f"   ⚠️ Sem genes com p-valor válido pra plotar volcano ({titulo})")
        return

    menor_p_nao_zero = df.loc[df['p_ajustado'] > 0, 'p_ajustado'].min()
    df['p_plot'] = df['p_ajustado'].replace(0, menor_p_nao_zero)
    df['neg_log10_p'] = -np.log10(df['p_plot'])

    cores = {'UP': PALETA(0.85), 'DOWN': PALETA(0.15), 'ns': '#c7c7c7'}

    fig, ax = plt.subplots(figsize=(9, 7))
    for regulacao, cor in cores.items():
        subset = df[df['regulacao'] == regulacao]
        ax.scatter(subset['log2FC'], subset['neg_log10_p'], c=[cor], s=12, alpha=0.7,
                   label=f"{regulacao} ({len(subset)})")

    ax.axhline(-np.log10(P_THRESHOLD), color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(LOG2FC_THRESHOLD, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-LOG2FC_THRESHOLD, color='gray', linestyle='--', linewidth=0.8)

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


def plot_elbow(k_range, inertias, melhor_k, titulo, caminho_saida):
    plt.figure(figsize=(7.5, 5))
    plt.plot(list(k_range), inertias, marker='o', linewidth=2, color=PALETA(0.3), markersize=8)
    plt.axvline(melhor_k, color=PALETA(0.85), linestyle='--', linewidth=1.5,
                label=f'k escolhido (silhouette) = {melhor_k}')
    plt.xlabel('k (número de clusters)')
    plt.ylabel('Inércia (soma das distâncias² intra-cluster)')
    plt.title(f'{titulo}\nMétodo do Cotovelo (Elbow)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {os.path.basename(caminho_saida)}")


def plot_boxplot_outliers(pca_df, df_meta, titulo, caminho_saida, n_destacar=3, escala_log=True):
    centroides = pca_df.groupby('Cluster')[['D1', 'D2']].transform('mean')
    dist = np.sqrt(((pca_df[['D1', 'D2']] - centroides) ** 2).sum(axis=1))
    df_dist = pd.DataFrame({'Cluster': pca_df['Cluster'].astype(str), 'Distancia': dist},
                            index=pca_df.index)
    df_dist['Dataset'] = df_meta.loc[df_dist.index, 'Dataset'] if 'Dataset' in df_meta.columns else 'N/A'

    clusters_ordenados = sorted(df_dist['Cluster'].unique(), key=lambda x: int(x))
    dados_por_cluster = [df_dist.loc[df_dist['Cluster'] == c, 'Distancia'].values for c in clusters_ordenados]

    plt.figure(figsize=(8.5, 5.5))
    bp = plt.boxplot(dados_por_cluster, labels=clusters_ordenados, patch_artist=True,
                      showfliers=False)
    for patch, c in zip(bp['boxes'], clusters_ordenados):
        patch.set_facecolor(PALETA(0.4))
        patch.set_alpha(0.5)

    rng = np.random.default_rng(42)
    datasets_unicos = sorted(df_dist['Dataset'].unique())
    cores_dataset = {ds: PALETA(i / max(len(datasets_unicos) - 1, 1)) for i, ds in enumerate(datasets_unicos)}
    for ds in datasets_unicos:
        sub = df_dist[df_dist['Dataset'] == ds]
        x_pos = [clusters_ordenados.index(c) + 1 for c in sub['Cluster']]
        x_jit = np.array(x_pos) + rng.normal(0, 0.06, size=len(sub))
        plt.scatter(x_jit, sub['Distancia'], s=16, color=cores_dataset[ds], alpha=0.65,
                    zorder=3, label=ds, edgecolor='white', linewidth=0.3)

    top_outliers = df_dist.sort_values('Distancia', ascending=False).head(n_destacar)
    for amostra, row in top_outliers.iterrows():
        x_pos = clusters_ordenados.index(row['Cluster']) + 1
        rotulo = f"{amostra}\n({row['Dataset']})"
        plt.annotate(rotulo, xy=(x_pos, row['Distancia']),
                     xytext=(x_pos + 0.25, row['Distancia']),
                     fontsize=7.5, color='#333333',
                     arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))

    if escala_log:
        plt.yscale('log')
    plt.xlabel('Cluster (K-means)')
    plt.ylabel('Distância euclidiana ao centróide do próprio cluster' + (' (escala log)' if escala_log else ''))
    plt.title(f'{titulo}\nDistância ao centróide por cluster (outliers identificados)')
    plt.legend(title='Dataset', fontsize=8, title_fontsize=9, loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {os.path.basename(caminho_saida)}")
    print(f"   ℹ️ Top {n_destacar} outliers:")
    for amostra, row in top_outliers.iterrows():
        print(f"      {amostra} (Dataset: {row['Dataset']}, Cluster {row['Cluster']}, "
              f"distância={row['Distancia']:.2f})")
    return df_dist


def gerar_pacote_apresentacao_clustering(df_expr_fonte, df_meta_fonte, coluna_grupo,
                                          titulo_prefixo, plots_dir, max_k=8,
                                          n_genes_variaveis=3000):
    print(f"\n--- Pacote de apresentação -- {titulo_prefixo} ---")
    df_var = select_variable_genes(df_expr_fonte, n_genes_variaveis)
    X = StandardScaler().fit_transform(df_var.T.values)
    pca = PCA(n_components=min(10, X.shape[0] - 1))
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca[:, :2], columns=['D1', 'D2'], index=df_var.columns)

    k_range_elbow = range(1, min(max_k + 1, X_pca.shape[0]))
    inertias = []
    sil_scores = []
    for k in k_range_elbow:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca[:, :3])
        inertias.append(km.inertia_)
        if k >= 2:
            labels = km.labels_
            sil_scores.append(silhouette_score(X_pca[:, :3], labels) if len(set(labels)) > 1 else 0)
    melhor_k = list(k_range_elbow)[1:][int(np.argmax(sil_scores))] if sil_scores else 2

    plot_elbow(k_range_elbow, inertias, melhor_k, titulo_prefixo,
               os.path.join(plots_dir, f"elbow_{titulo_prefixo}.png"))

    kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=20)
    pca_df['Cluster'] = kmeans_final.fit_predict(X_pca[:, :3]).astype(str)

    plot_scatter_2d(pca_df, pca_df, 'Cluster',
                     f"{titulo_prefixo} -- clusters do K-means (k={melhor_k})",
                     os.path.join(plots_dir, f"scatter_clusters_{titulo_prefixo}.png"))

    ari, n = calcular_ari(pca_df, df_meta_fonte, coluna_grupo)
    plot_scatter_2d(pca_df, df_meta_fonte, coluna_grupo,
                     f"{titulo_prefixo} -- colorido por {coluna_grupo}\n(ARI={ari:.3f})",
                     os.path.join(plots_dir, f"scatter_{coluna_grupo}_{titulo_prefixo}.png"))

    if 'Dataset' in df_meta_fonte.columns and coluna_grupo != 'Dataset':
        ari_dataset, _ = calcular_ari(pca_df, df_meta_fonte, 'Dataset')
        plot_scatter_2d(pca_df, df_meta_fonte, 'Dataset',
                         f"{titulo_prefixo} -- colorido por Dataset\n(ARI={ari_dataset:.3f})",
                         os.path.join(plots_dir, f"scatter_Dataset_{titulo_prefixo}.png"))

    plot_composicao_cluster(pca_df, df_meta_fonte, coluna_grupo, titulo_prefixo,
                             os.path.join(plots_dir, f"composicao_{titulo_prefixo}.png"))

    plot_boxplot_outliers(pca_df, df_meta_fonte, titulo_prefixo,
                           os.path.join(plots_dir, f"boxplot_outliers_{titulo_prefixo}.png"))

    return pca_df, melhor_k, ari


def plot_scatter_2d(coords_df, df_meta, coluna_cor, titulo, caminho_saida, eixo_x='D1', eixo_y='D2'):
    rotulos = df_meta.loc[coords_df.index, coluna_cor].fillna('N/A').astype(str)
    valores_unicos = sorted(rotulos.unique())
    cores = PALETA(np.linspace(0, 1, max(len(valores_unicos), 2)))

    plt.figure(figsize=(8, 6))
    for cor, valor in zip(cores, valores_unicos):
        mask = rotulos == valor
        plt.scatter(coords_df.loc[mask, eixo_x], coords_df.loc[mask, eixo_y],
                    color=cor, label=valor, alpha=0.8, s=45)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.title(f'{titulo}\ncolorido por: {coluna_cor}')
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {os.path.basename(caminho_saida)}")


def plot_composicao_cluster(coords_df, df_meta, coluna_grupo, titulo, caminho_saida):
    rotulos = df_meta.loc[coords_df.index, coluna_grupo].fillna('N/A').astype(str)
    tabela = pd.crosstab(coords_df['Cluster'], rotulos, normalize='index') * 100
    tabela = tabela.sort_index()

    cores = PALETA(np.linspace(0, 1, max(tabela.shape[1], 2)))
    plt.figure(figsize=(8, 5))
    baixo = np.zeros(len(tabela))
    for cor, grupo in zip(cores, tabela.columns):
        plt.bar(tabela.index.astype(str), tabela[grupo], bottom=baixo,
                color=cor, label=grupo, edgecolor='white')
        baixo += tabela[grupo].values

    plt.xlabel('Cluster (K-means)')
    plt.ylabel('% de amostras')
    plt.title(f'{titulo}\ncomposição de cada cluster por: {coluna_grupo}')
    plt.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {os.path.basename(caminho_saida)}")


def plot_ari_resumo(resultados_ari, titulo, caminho_saida):
    itens = [(k, v) for k, v in resultados_ari.items() if v is not None and not np.isnan(v)]
    if not itens:
        print("   ⚠️ Nenhum ARI válido pra plotar.")
        return
    nomes = [k for k, v in itens]
    valores = [v for k, v in itens]

    cores = [PALETA(0.7) if v >= 0 else PALETA(0.2) for v in valores]
    plt.figure(figsize=(max(8, len(nomes) * 1.1), 6))
    plt.bar(nomes, valores, color=cores, edgecolor='white')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.ylabel('ARI (quanto maior, mais a clusterização reflete o grupo biológico)')
    plt.title(titulo)
    plt.xticks(rotation=35, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {os.path.basename(caminho_saida)}")


def plot_silhouette(k_range, sil_scores, titulo, caminho_saida):
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), sil_scores, 'o-', color=PALETA(0.3))
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Silhouette salvo: {os.path.basename(caminho_saida)}")


# ============================================================================
# 5. PCA + K-MEANS + UMAP (compartilhada)
# ============================================================================
def select_variable_genes(df_expr, n_genes=3000):
    variancias = df_expr.var(axis=1)
    return df_expr.loc[variancias.sort_values(ascending=False).head(n_genes).index]


def select_category_genes(df_expr, categoria):
    genes_cat = GENES_CATEGORIAS[categoria]
    genes_presentes = [g for g in genes_cat if g in df_expr.index]
    genes_faltantes = [g for g in genes_cat if g not in df_expr.index]
    print(f"   ℹ️ Categoria '{categoria}': {len(genes_presentes)}/{len(genes_cat)} genes presentes na matriz"
          + (f" -- não encontrados: {genes_faltantes}" if genes_faltantes else ""))
    return df_expr.loc[genes_presentes]


def run_pca_kmeans_umap(df_expr, max_k=8, usar_yeo_johnson=False):
    """
    Remove genes com NaN logo no início -- protege contra ValueError caso
    algum gene fique com variância zero em algum grupo após o ComBat.

    usar_yeo_johnson: se True, aplica PowerTransformer(method='yeo-johnson')
    por gene ANTES do StandardScaler -- corrige assimetria da distribuição
    de cada gene. Comparar ARI com/sem é uma forma honesta de checar se a
    separação fraca entre grupos é artefato de distribuição ou sinal
    biológico real.
    """
    n_genes_antes = df_expr.shape[0]
    df_expr = df_expr.dropna(axis=0, how='any')
    n_removidos = n_genes_antes - df_expr.shape[0]
    if n_removidos > 0:
        print(f"   ℹ️ {n_removidos} genes removidos por NaN pós-ComBat (variância zero em algum grupo)")

    X = df_expr.T.values  # amostras em linhas, genes em colunas
    if usar_yeo_johnson:
        X = PowerTransformer(method='yeo-johnson').fit_transform(X)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(10, X.shape[0] - 1))
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca[:, :2], columns=['D1', 'D2'], index=df_expr.columns)

    elbow_k_range = range(1, min(max_k + 1, len(X)))
    inertias = []
    for k in elbow_k_range:
        km_elbow = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca[:, :3])
        inertias.append(km_elbow.inertia_)

    k_range = range(2, min(max_k + 1, len(X)))
    sil_scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pca[:, :3])
        sil_scores.append(silhouette_score(X_pca[:, :3], labels) if len(set(labels)) > 1 else 0)
    melhor_k = list(k_range)[int(np.argmax(sil_scores))]

    kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=20)
    pca_df['Cluster'] = kmeans_final.fit_predict(X_pca[:, :3])

    print(f"   • k ótimo (silhouette): {melhor_k} | variância explicada (2 PCs): "
          f"{pca.explained_variance_ratio_[:2].sum()*100:.1f}%")

    umap_df = None
    if UMAP_DISPONIVEL:
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)
        umap_df = pd.DataFrame(X_umap, columns=['D1', 'D2'], index=df_expr.columns)
        umap_df['Cluster'] = pca_df['Cluster'].values
    else:
        print("   ⚠️ umap-learn não instalado (pip install umap-learn) -- pulando UMAP, mantendo só PCA")

    return pca_df, umap_df, melhor_k, list(k_range), sil_scores, list(elbow_k_range), inertias


def calcular_ari(coords_df, df_meta, coluna_rotulo):
    rotulos = df_meta.loc[coords_df.index, coluna_rotulo]
    mask = rotulos.notna()
    if mask.sum() < 2 or rotulos[mask].nunique() < 2:
        return np.nan, mask.sum()
    return adjusted_rand_score(rotulos[mask], coords_df.loc[mask, 'Cluster']), mask.sum()


# ============================================================================
# ETAPA A -- DEGs DENTRO DE CADA DATASET (+ volcano)
# ============================================================================
def etapa_A_degs_dentro_dataset(df_expr, df_meta):
    print("\n" + "="*80)
    print("ETAPA A -- DEGs DENTRO DE CADA DATASET")
    print("="*80)

    resultados = {}
    plots_dir_a = os.path.join(PLOTS_DIR, "A_dentro_dataset")
    os.makedirs(plots_dir_a, exist_ok=True)

    for dataset, coluna, valor1, valor2, nome_comp in COMPARACOES_DENTRO_DATASET:
        print(f"\n🔬 {dataset} | {coluna}: {valor1} vs {valor2} ({nome_comp})")

        df_ds = df_meta[df_meta['Dataset'] == dataset]
        if coluna not in df_ds.columns:
            print(f"   ⚠️ Coluna '{coluna}' não existe -- pulando.")
            continue

        amostras_c1 = [a for a in df_ds[df_ds[coluna] == valor1].index if a in df_expr.columns]
        amostras_c2 = [a for a in df_ds[df_ds[coluna] == valor2].index if a in df_expr.columns]

        print(f"   • {valor1}: {len(amostras_c1)} | {valor2}: {len(amostras_c2)}")
        if len(amostras_c1) < 2 or len(amostras_c2) < 2:
            print("   ⚠️ Amostras insuficientes -- pulando.")
            continue

        df_de = run_de_vetorizado(df_expr, amostras_c1, amostras_c2, valor1, valor2)
        df_de['dataset'] = dataset
        df_de['comparacao'] = nome_comp

        n_sig = (df_de['regulacao'] != 'ns').sum()
        n_up = (df_de['regulacao'] == 'UP').sum()
        n_down = (df_de['regulacao'] == 'DOWN').sum()
        print(f"   ✅ {n_sig} genes (↑{n_up} ↓{n_down})")

        df_de.to_csv(os.path.join(OUTPUT_DIR, f"DE_{dataset}_{nome_comp}.csv"), index=False)

        plot_volcano(df_de, f"{dataset} -- {nome_comp}",
                     os.path.join(plots_dir_a, f"volcano_{dataset}_{nome_comp}.png"))

        resultados[nome_comp] = df_de

    print("\n" + "-"*80)
    print("Comparações PAREADAS (mesmo sujeito, ttest_rel)")
    print("-"*80)
    for dataset, coluna, valor1, valor2, nome_comp, col_sujeito, col_extra in COMPARACOES_PAREADAS:
        print(f"\n🔬 [PAREADO] {dataset} | {coluna}: {valor1} vs {valor2} ({nome_comp})")

        df_de = run_de_pareado(df_expr, df_meta, dataset, coluna, valor1, valor2,
                                coluna_sujeito=col_sujeito, coluna_chave_extra=col_extra,
                                nome_c1=str(valor1), nome_c2=str(valor2))
        if df_de is None:
            continue
        df_de['dataset'] = dataset
        df_de['comparacao'] = nome_comp

        n_sig = (df_de['regulacao'] != 'ns').sum()
        n_up = (df_de['regulacao'] == 'UP').sum()
        n_down = (df_de['regulacao'] == 'DOWN').sum()
        print(f"   ✅ {n_sig} genes (↑{n_up} ↓{n_down})")

        df_de.to_csv(os.path.join(OUTPUT_DIR, f"DE_{dataset}_{nome_comp}.csv"), index=False)

        plot_volcano(df_de, f"{dataset} -- {nome_comp}",
                     os.path.join(plots_dir_a, f"volcano_{dataset}_{nome_comp}.png"))

        resultados[nome_comp] = df_de

    return resultados


# ============================================================================
# ETAPA B -- CRUZAMENTO ENTRE DATASETS (batch correction + K-means/PCA/UMAP)
# ============================================================================
def etapa_B_cruzamento_datasets(df_expr, df_meta):
    print("\n" + "="*80)
    print("ETAPA B -- CRUZAMENTO ENTRE DATASETS (com correção de batch)")
    print("="*80)

    plots_dir_b = os.path.join(PLOTS_DIR, "B_cruzamento")
    os.makedirs(plots_dir_b, exist_ok=True)

    df_meta_b = df_meta[df_meta['Dataset'].isin(DATASETS_CRUZAMENTO)]
    common = sorted(set(df_expr.columns) & set(df_meta_b.index))
    df_expr_b = df_expr[common]
    df_meta_b = df_meta_b.loc[common]
    print(f"✅ {len(common)} amostras nos datasets: {', '.join(DATASETS_CRUZAMENTO)}")

    print("\n--- Diagnóstico: variável biológica x batch é identificável? ---")
    for col_biologica in ['Grupo_Etario', 'Momento_Treino_Cruzado']:
        if col_biologica in df_meta_b.columns:
            print(f"\n[{col_biologica}]")
            checar_confundimento(df_meta_b, common, 'Dataset', col_biologica)

    USAR_COMBAT = True
    if USAR_COMBAT and COMBAT_DISPONIVEL:
        df_corrected = correct_batch_effect_combat(df_expr_b, df_meta_b, proteger_col='Grupo_Etario')
    else:
        if USAR_COMBAT and not COMBAT_DISPONIVEL:
            print("\n⚠️ 'combat' não instalado -- usando mean-centering ingênuo como fallback")
        df_corrected = correct_batch_effect(df_expr_b, df_meta_b)

    resultados = {}

    print("\n--- B1: idoso vs jovem (cruzando datasets) ---")
    amostras_idoso = [a for a in df_meta_b[df_meta_b['Grupo_Etario'] == 'Idoso'].index if a in df_corrected.columns]
    amostras_jovem = [a for a in df_meta_b[df_meta_b['Grupo_Etario'] == 'Jovem adulto'].index if a in df_corrected.columns]
    print(f"   Idoso: {len(amostras_idoso)} | Jovem adulto: {len(amostras_jovem)}")

    if len(amostras_idoso) >= 2 and len(amostras_jovem) >= 2:
        df_de_idade = run_de_vetorizado(df_corrected, amostras_jovem, amostras_idoso, 'jovem', 'idoso')
        n_sig = (df_de_idade['regulacao'] != 'ns').sum()
        print(f"   ✅ {n_sig} genes significativos")
        df_de_idade.to_csv(os.path.join(OUTPUT_DIR, "DE_CRUZADO_idoso_vs_jovem.csv"), index=False)
        plot_volcano(df_de_idade, "Idoso vs Jovem (cruzado, batch-corrigido)",
                     os.path.join(plots_dir_b, "volcano_cruzado_idoso_vs_jovem.png"))
        resultados['idoso_vs_jovem_cruzado'] = df_de_idade

    print("\n--- Clustering pós-correção (PCA + K-means + UMAP) ---")
    df_variaveis = select_variable_genes(df_corrected, 3000)
    pca_df, umap_df, melhor_k, k_range, sil_scores, elbow_k_range, inertias = run_pca_kmeans_umap(df_variaveis)

    plot_elbow(elbow_k_range, inertias, melhor_k, "Cruzamento pós-batch",
               os.path.join(plots_dir_b, "elbow_cruzamento.png"))

    for coluna_cor in ['Dataset', 'Grupo_Etario']:
        ari, n = calcular_ari(pca_df, df_meta_b, coluna_cor)
        print(f"   ARI vs {coluna_cor}: {ari:.3f} (n={n})" if not np.isnan(ari) else f"   ARI vs {coluna_cor}: n/a")
        plot_scatter_2d(pca_df, df_meta_b, coluna_cor, f"PCA pós-batch (k={melhor_k})",
                         os.path.join(plots_dir_b, f"pca_{coluna_cor}.png"))
        if umap_df is not None:
            plot_scatter_2d(umap_df, df_meta_b, coluna_cor, "UMAP pós-batch",
                             os.path.join(plots_dir_b, f"umap_{coluna_cor}.png"))

    plot_scatter_2d(pca_df, pca_df, 'Cluster', f"PCA pós-batch -- clusters do K-means (k={melhor_k})",
                     os.path.join(plots_dir_b, "pca_Cluster.png"))
    if umap_df is not None:
        plot_scatter_2d(umap_df, umap_df, 'Cluster', f"UMAP pós-batch -- clusters do K-means (k={melhor_k})",
                         os.path.join(plots_dir_b, "umap_Cluster.png"))

    plot_silhouette(k_range, sil_scores, "Silhouette -- cruzamento pós-batch",
                     os.path.join(plots_dir_b, "silhouette_cruzamento.png"))

    return resultados, df_expr_b, df_corrected, df_meta_b


# ============================================================================
# ETAPA D -- CLUSTERIZAÇÃO POR CATEGORIA FUNCIONAL (pós-ComBat)
# ============================================================================
CATEGORIAS_CLUSTER = ['musculares', 'neuromuscular', 'oxigenacao']
TESTAR_YEO_JOHNSON = True
NOMES_AMIGAVEIS_CATEGORIA = {
    'musculares': 'Muscular',
    'neuromuscular': 'Neuromuscular (proxy neuronal)',
    'oxigenacao': 'Oxigenação/Circulatório',
}


def _cluster_e_plot_categoria(df_expr_fonte, df_meta_fonte, categoria, coluna_cor,
                               prefixo_titulo, plots_dir, sufixo_arquivo,
                               usar_yeo_johnson=False, gerar_plots=True):
    nome_bonito = NOMES_AMIGAVEIS_CATEGORIA.get(categoria, categoria)
    print(f"\n--- {prefixo_titulo} | categoria: {nome_bonito}"
          f"{' [Yeo-Johnson]' if usar_yeo_johnson else ''} ---")

    df_cat = select_category_genes(df_expr_fonte, categoria)
    if df_cat.shape[0] < 5:
        print("   ⚠️ Menos de 5 genes disponíveis nessa categoria -- pulando clusterização.")
        return None

    pca_df, _, melhor_k, _, _, _, _ = run_pca_kmeans_umap(df_cat, max_k=6, usar_yeo_johnson=usar_yeo_johnson)
    ari, n = calcular_ari(pca_df, df_meta_fonte, coluna_cor)
    print(f"   • k ótimo: {melhor_k} | ARI vs {coluna_cor}: "
          f"{ari:.3f} (n={n})" if not np.isnan(ari) else f"   ARI vs {coluna_cor}: n/a")

    if not gerar_plots:
        return ari

    caminho = os.path.join(plots_dir, f"pca_{sufixo_arquivo}_{categoria}"
                            f"{'_YJ' if usar_yeo_johnson else ''}.png")
    plot_scatter_2d(pca_df, df_meta_fonte, coluna_cor,
                     f"{nome_bonito} -- {prefixo_titulo}"
                     f"{' (Yeo-Johnson)' if usar_yeo_johnson else ''}\n(k={melhor_k}, ARI={ari:.3f})",
                     caminho)

    caminho_comp = os.path.join(plots_dir, f"composicao_{sufixo_arquivo}_{categoria}"
                                 f"{'_YJ' if usar_yeo_johnson else ''}.png")
    plot_composicao_cluster(pca_df, df_meta_fonte, coluna_cor,
                             f"{nome_bonito} -- {prefixo_titulo}"
                             f"{' (Yeo-Johnson)' if usar_yeo_johnson else ''}",
                             caminho_comp)

    if 'Dataset' in df_meta_fonte.columns and coluna_cor != 'Dataset':
        caminho_dataset = os.path.join(plots_dir, f"pca_{sufixo_arquivo}_{categoria}_porDataset"
                                        f"{'_YJ' if usar_yeo_johnson else ''}.png")
        ari_dataset, _ = calcular_ari(pca_df, df_meta_fonte, 'Dataset')
        plot_scatter_2d(pca_df, df_meta_fonte, 'Dataset',
                         f"{nome_bonito} -- {prefixo_titulo} (colorido por Dataset)"
                         f"{' (Yeo-Johnson)' if usar_yeo_johnson else ''}\n(k={melhor_k}, ARI={ari_dataset:.3f})",
                         caminho_dataset)
    return ari


def testar_multiplos_k(df_expr_fonte, df_meta_fonte, coluna_grupo, titulo_prefixo,
                        plots_dir, k_list=(2, 3, 4, 5, 6, 8), n_genes_variaveis=3000):
    print(f"\n--- {titulo_prefixo}: testando k={list(k_list)} (sem filtro de categoria) ---")
    df_var = select_variable_genes(df_expr_fonte, n_genes_variaveis)
    X = StandardScaler().fit_transform(df_var.T.values)
    pca = PCA(n_components=min(10, X.shape[0] - 1))
    X_pca = pca.fit_transform(X)
    pca_df_base = pd.DataFrame(X_pca[:, :2], columns=['D1', 'D2'], index=df_var.columns)

    aris = []
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca[:, :min(3, X_pca.shape[1])])
        pca_df = pca_df_base.copy()
        pca_df['Cluster'] = km.labels_
        ari, n = calcular_ari(pca_df, df_meta_fonte, coluna_grupo)
        aris.append(ari)
        print(f"   k={k} | ARI vs {coluna_grupo}: {ari:.3f} (n={n})" if not np.isnan(ari) else f"   k={k}: n/a")

        plot_composicao_cluster(
            pca_df, df_meta_fonte, coluna_grupo,
            f"{titulo_prefixo} -- k={k} (sem filtro de categoria)",
            os.path.join(plots_dir, f"composicao_{titulo_prefixo}_k{k}.png")
        )

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(list(k_list), aris, marker='o', linewidth=2, color=PALETA(0.4), markersize=8)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xlabel('k (número de clusters forçado)')
    plt.ylabel(f'ARI vs {coluna_grupo}')
    plt.title(f'{titulo_prefixo}\nARI por número de clusters (sem filtro de categoria)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    caminho = os.path.join(plots_dir, f"ARI_por_k_{titulo_prefixo}.png")
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {os.path.basename(caminho)}")

    return dict(zip(k_list, aris))


def etapa_D_cluster_por_categoria(df_expr, df_meta, df_expr_b_bruto, df_corrected_b, df_meta_b):
    print("\n" + "="*80)
    print("ETAPA D -- CLUSTERIZAÇÃO POR CATEGORIA FUNCIONAL (pós-ComBat)")
    print("="*80)

    plots_dir_d = os.path.join(PLOTS_DIR, "D_cluster_categoria")
    os.makedirs(plots_dir_d, exist_ok=True)

    resultados_ari = {}

    # ------------------------------------------------------------------
    # Genes musculares e neuromusculares, ANTES e DEPOIS do ComBat, lado a
    # lado -- testa se o efeito de lote ainda domina um recorte
    # biologicamente coerente, ou se o ComBat resolve isso.
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Cruzado musculares/neuromuscular: ANTES vs DEPOIS do ComBat")
    print("-"*80)
    EIXOS_COR_CRUZADO = ['Grupo_Etario', 'Momento_Treino_Cruzado']
    for eixo_cor in EIXOS_COR_CRUZADO:
        print(f"\n>>> Eixo de cor: {eixo_cor}")
        for categoria in ['musculares', 'neuromuscular']:
            ari_pre = _cluster_e_plot_categoria(
                df_expr_b_bruto, df_meta_b, categoria, eixo_cor,
                f"Cruzado ({len(DATASETS_CRUZAMENTO)} datasets, PRÉ-ComBat, {eixo_cor})",
                plots_dir_d, f"cruzado_PREcombat_{eixo_cor}"
            )
            resultados_ari[f"cruzado_PREcombat_{eixo_cor}_{categoria}"] = ari_pre

            ari_pos = _cluster_e_plot_categoria(
                df_corrected_b, df_meta_b, categoria, eixo_cor,
                f"Cruzado ({len(DATASETS_CRUZAMENTO)} datasets, PÓS-ComBat, {eixo_cor})",
                plots_dir_d, f"cruzado_POScombat_{eixo_cor}"
            )
            resultados_ari[f"cruzado_POScombat_{eixo_cor}_{categoria}"] = ari_pos

            delta = (ari_pos - ari_pre) if (ari_pre is not None and ari_pos is not None
                                             and not np.isnan(ari_pre) and not np.isnan(ari_pos)) else np.nan
            print(f"   📊 {categoria} | {eixo_cor}: ARI pré-ComBat={ari_pre:.3f} -> pós-ComBat={ari_pos:.3f} "
                  f"(Δ={delta:+.3f})" if not np.isnan(delta) else f"   📊 {categoria} | {eixo_cor}: Δ n/a")

    print("\n" + "-"*80)
    print("Pacote de apresentação -- genes completos (Elbow + boxplot de outliers)")
    print("-"*80)
    gerar_pacote_apresentacao_clustering(
        df_corrected_b, df_meta_b, 'Grupo_Etario',
        "cruzado_generico", plots_dir_d
    )
    if 'Idade_x_Treino' in df_meta.columns:
        df_157585_pkg = df_meta[df_meta['Dataset'] == 'GSE157585']
        amostras_157585_pkg = [a for a in df_157585_pkg.index if a in df_expr.columns]
        gerar_pacote_apresentacao_clustering(
            df_expr[amostras_157585_pkg], df_meta, 'Idade_x_Treino',
            "GSE157585_generico", plots_dir_d
        )

    print("\n" + "-"*80)
    print("Opção A -- clusterização SEM filtro de categoria (genes de maior variância)")
    print("-"*80)
    if 'Idade_x_Treino' in df_meta.columns:
        df_157585_semfiltro = df_meta[df_meta['Dataset'] == 'GSE157585']
        amostras_157585_sf = [a for a in df_157585_semfiltro.index if a in df_expr.columns]
        testar_multiplos_k(df_expr[amostras_157585_sf], df_meta, 'Idade_x_Treino',
                            "GSE157585_semfiltro", plots_dir_d)
    testar_multiplos_k(df_corrected_b, df_meta_b, 'Grupo_Etario',
                        "cruzado_semfiltro", plots_dir_d)

    if 'Idade_x_Treino' in df_meta.columns:
        df_157585 = df_meta[df_meta['Dataset'] == 'GSE157585']
        amostras_157585 = [a for a in df_157585.index if a in df_expr.columns]
        df_expr_157585 = df_expr[amostras_157585]

        for categoria in CATEGORIAS_CLUSTER:
            ari = _cluster_e_plot_categoria(
                df_expr_157585, df_meta, categoria, 'Idade_x_Treino',
                "GSE157585 (idoso baseline/pós-treino/jovem)",
                plots_dir_d, "GSE157585_IdadeTreino"
            )
            resultados_ari[f"GSE157585_{categoria}"] = ari
            if TESTAR_YEO_JOHNSON:
                ari_yj = _cluster_e_plot_categoria(
                    df_expr_157585, df_meta, categoria, 'Idade_x_Treino',
                    "GSE157585 (idoso baseline/pós-treino/jovem)",
                    plots_dir_d, "GSE157585_IdadeTreino",
                    usar_yeo_johnson=True, gerar_plots=False
                )
                resultados_ari[f"GSE157585_{categoria}_YJ"] = ari_yj

    for categoria in CATEGORIAS_CLUSTER:
        ari = _cluster_e_plot_categoria(
            df_corrected_b, df_meta_b, categoria, 'Grupo_Etario',
            f"Cruzado ({len(DATASETS_CRUZAMENTO)} datasets, pós-ComBat)",
            plots_dir_d, "cruzado_GrupoEtario"
        )
        resultados_ari[f"cruzado_{categoria}"] = ari
        if TESTAR_YEO_JOHNSON:
            ari_yj = _cluster_e_plot_categoria(
                df_corrected_b, df_meta_b, categoria, 'Grupo_Etario',
                f"Cruzado ({len(DATASETS_CRUZAMENTO)} datasets, pós-ComBat)",
                plots_dir_d, "cruzado_GrupoEtario",
                usar_yeo_johnson=True, gerar_plots=False
            )
            resultados_ari[f"cruzado_{categoria}_YJ"] = ari_yj

    print("\n--- Resumo ARI por categoria (quanto maior, mais a clusterização" \
          " reflete o grupo biológico) ---")
    for k, v in resultados_ari.items():
        print(f"   {k}: {v:.3f}" if v is not None and not np.isnan(v) else f"   {k}: n/a")

    plot_ari_resumo(resultados_ari, "Resumo -- separação idoso/jovem por categoria funcional",
                     os.path.join(plots_dir_d, "resumo_ARI_por_categoria.png"))

    return resultados_ari


# ============================================================================
# ETAPA C -- MAPEAMENTO GENE -> FÁRMACO (DGIdb), BENÉFICO x MALÉFICO
# ============================================================================
DGIDB_URL = 'https://dgidb.org/api/graphql'

FONTES_BENEFICO = [('baseline_vs_pos_treino_157585_PAREADO', 'UP')]
FONTES_MALEFICO = [('idoso_vs_jovem_157585', 'UP')]
CATEGORIAS_FARMACO = ['musculares', 'inflamacao']


def _coletar_genes(resultados_etapa_a, fontes):
    genes = set()
    for nome_comp, direcao in fontes:
        df = resultados_etapa_a.get(nome_comp)
        if df is None:
            print(f"   ⚠️ Comparação '{nome_comp}' não encontrada nos resultados da Etapa A "
                  f"-- confira se ela está ativa em COMPARACOES_DENTRO_DATASET/PAREADAS.")
            continue
        for categoria in CATEGORIAS_FARMACO:
            genes_cat = GENES_CATEGORIAS[categoria]
            genes.update(df[(df['gene'].isin(genes_cat)) & (df['regulacao'] == direcao)]['gene'].tolist())
    return sorted(genes)


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


def etapa_C_mapeamento_farmaco(resultados_etapa_a):
    print("\n" + "="*80)
    print("ETAPA C -- MAPEAMENTO GENE -> FÁRMACO (DGIdb): BENÉFICO x MALÉFICO")
    print("="*80)

    genes_beneficos = _coletar_genes(resultados_etapa_a, FONTES_BENEFICO)
    genes_maleficos = _coletar_genes(resultados_etapa_a, FONTES_MALEFICO)

    print(f"\n🟢 Genes 'benéficos' (exercício/hipertrofia): {genes_beneficos}")
    print(f"🔴 Genes 'maléficos' (atrofia/inflamação crônica): {genes_maleficos}")

    print("\n🔎 Consultando DGIdb -- genes benéficos...")
    df_beneficos = consultar_dgidb(genes_beneficos)
    if not df_beneficos.empty:
        df_beneficos = df_beneficos.sort_values('interaction_score', ascending=False)
        df_beneficos.to_csv(os.path.join(OUTPUT_DIR, "FARMACOS_BENEFICOS.csv"), index=False)
        print(df_beneficos.head(15).to_string(index=False))
    else:
        print("   (sem resultados)")

    print("\n🔎 Consultando DGIdb -- genes maléficos...")
    df_maleficos = consultar_dgidb(genes_maleficos)
    if not df_maleficos.empty:
        df_maleficos = df_maleficos.sort_values('interaction_score', ascending=False)
        df_maleficos.to_csv(os.path.join(OUTPUT_DIR, "FARMACOS_MALEFICOS.csv"), index=False)
        print(df_maleficos.head(15).to_string(index=False))
    else:
        print("   (sem resultados)")

    if not df_beneficos.empty and not df_maleficos.empty:
        farmacos_beneficos = set(df_beneficos['farmaco'].dropna())
        farmacos_maleficos = set(df_maleficos['farmaco'].dropna())
        ambos = farmacos_beneficos & farmacos_maleficos
        so_malefico = farmacos_maleficos - farmacos_beneficos
        print(f"\n⚠️ Fármacos com alvo nos DOIS eixos (efeito potencialmente ambíguo): {sorted(ambos)}")
        print(f"🚫 Fármacos SÓ no eixo maléfico (uso questionável pro ganho muscular): {sorted(so_malefico)}")


# ============================================================================
# MAIN -- orquestra as 4 etapas conforme MASTER_SWITCHES
# ============================================================================
def main():
    print("="*80)
    print("🔬 PIPELINE CONSOLIDADO DE ANÁLISE DE TRANSCRIPTOMA")
    print("="*80)

    df_expr, df_meta = carregar_matriz_e_metadados()

    resultados_a = {}
    if MASTER_SWITCHES['A_degs_dentro_dataset']:
        resultados_a = etapa_A_degs_dentro_dataset(df_expr, df_meta)

    df_expr_b_bruto, df_corrected_b, df_meta_b = None, None, None
    if MASTER_SWITCHES['B_cruzamento_datasets']:
        _, df_expr_b_bruto, df_corrected_b, df_meta_b = etapa_B_cruzamento_datasets(df_expr, df_meta)

        checkpoint = {'df_expr': df_expr, 'df_meta': df_meta,
                      'df_expr_b_bruto': df_expr_b_bruto,
                      'df_corrected_b': df_corrected_b, 'df_meta_b': df_meta_b}
        with open(CHECKPOINT_ETAPA_B, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"\n💾 Checkpoint salvo: {CHECKPOINT_ETAPA_B} "
              f"(usado pelo 04_sensibilidade_limiares.py)")

    if MASTER_SWITCHES['D_cluster_categoria']:
        if df_corrected_b is None:
            print("\n⚠️ Etapa D pulada -- precisa da Etapa B rodada (df_corrected) primeiro.")
        else:
            etapa_D_cluster_por_categoria(df_expr, df_meta, df_expr_b_bruto, df_corrected_b, df_meta_b)

    if MASTER_SWITCHES['C_mapeamento_farmaco']:
        if resultados_a:
            etapa_C_mapeamento_farmaco(resultados_a)
        else:
            print("\n⚠️ Etapa C precisa dos resultados da Etapa A -- ligue 'A_degs_dentro_dataset' também.")

    print("\n" + "="*80)
    print("✅ PIPELINE CONCLUÍDO")
    print(f"📂 Resultados em: {OUTPUT_DIR}")
    print(f"📊 Gráficos em: {PLOTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()