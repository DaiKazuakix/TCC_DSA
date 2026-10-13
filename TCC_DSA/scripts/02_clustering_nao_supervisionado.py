# -*- coding: utf-8 -*-
"""
PASSO 02 -- Normalização log2CPM e análise exploratória antes da correção de lote
==================================================================================
Normaliza a matriz integrada de contagens (log2(CPM+1)) e executa a análise
não supervisionada exploratória (z-score, PCA e K-means com escolha de k
por silhueta), sem nenhuma correção de efeito de lote.

A PCA é repetida com três critérios de seleção de genes, para comparação:
  1) top 3000 genes mais variáveis;
  2) top 3000 genes mais expressos (genes estáveis -- se até eles separarem
     as amostras por dataset, é forte indício de efeito técnico);
  3) painel curado de genes musculares (GENES_CATEGORIAS['musculares']).

Também calcula o índice de separação por dataset (distância média
inter-dataset / intra-dataset), usado para quantificar o efeito de lote.

Entrada:  dados/processados/MATRIZ_INTEGRADA_CONTAGENS.csv
          metadados/METADADOS_ENRIQUECIDOS_v2.csv
Saída:    dados/processados/MATRIZ_LOG2CPM.csv
          dados/processados/METADADOS_CLUSTERS_3CRITERIOS.csv
          resultados/figuras/02_exploratoria/
"""
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist

from config import DIR_PROCESSADOS, DIR_FIGURAS, MATRIZ_CONTAGENS, METADADOS, GENES_CATEGORIAS

warnings.filterwarnings('ignore')

print("="*80)
print("🧬 PASSO 02 -- z-score, PCA, K-means (3 critérios de seleção de genes)")
print(f"📅 Data de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
OUTPUT_DIR = DIR_PROCESSADOS
PLOTS_DIR = os.path.join(DIR_FIGURAS, "02_exploratoria")
METADATA_V2_PATH = METADADOS

os.makedirs(PLOTS_DIR, exist_ok=True)

PALETA = plt.cm.viridis


def get_viridis_map(categorias):
    cats = sorted({str(c) for c in categorias})
    cores = PALETA(np.linspace(0, 0.92, max(len(cats), 2)))
    return dict(zip(cats, cores))


# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
def load_and_prepare_data():
    print(f"\n📥 1. CARREGANDO DADOS INTEGRADOS...")

    matrix_path = MATRIZ_CONTAGENS
    if not os.path.exists(matrix_path):
        print(f"❌ Matriz de contagens não encontrada: {matrix_path}")
        print("   Rode 01_integracao_datasets.py primeiro.")
        return None, None

    print(f"📊 Carregando matriz: {os.path.basename(matrix_path)}")
    df_expr = pd.read_csv(matrix_path, index_col=0)
    print(f"   • Dimensões: {df_expr.shape[0]} genes × {df_expr.shape[1]} amostras")

    df_meta = None
    if os.path.exists(METADATA_V2_PATH):
        print(f"📋 Carregando metadado v2: {METADATA_V2_PATH}")
        df_meta = pd.read_csv(METADATA_V2_PATH, sep=';', index_col=0, encoding='latin1')
        for col in df_meta.select_dtypes(include='object').columns:
            df_meta[col] = df_meta[col].str.strip()
        df_meta = df_meta.reset_index().rename(columns={df_meta.index.name or 'index': 'Amostra'})
        if 'Amostra' not in df_meta.columns:
            df_meta = df_meta.rename(columns={df_meta.columns[0]: 'Amostra'})
    else:
        raise FileNotFoundError(
            f"❌ METADADOS_ENRIQUECIDOS_v2.csv não encontrado em {METADATA_V2_PATH} -- "
            "ele deve estar na pasta metadados/ do repositório."
        )

    matrix_samples = set(df_expr.columns)
    metadata_samples = set(df_meta['Amostra'])
    common_samples = matrix_samples & metadata_samples

    print(f"\n   📊 ALINHAMENTO:")
    print(f"   • Amostras na matriz: {len(matrix_samples)}")
    print(f"   • Amostras no metadado: {len(metadata_samples)}")
    print(f"   • Amostras em comum: {len(common_samples)}")

    if not common_samples:
        raise ValueError(
            "❌ Interseção entre colunas da matriz e 'Amostra' do metadado ficou vazia."
        )

    df_expr = df_expr[list(common_samples)]
    df_meta = df_meta[df_meta['Amostra'].isin(common_samples)]
    print(f"\n✅ Dados alinhados: {len(common_samples)} amostras em comum")

    return df_expr, df_meta


# ============================================================================
# 2. NORMALIZAÇÃO (log2CPM)
# ============================================================================
def normalize_rnaseq_correct(df_expr):
    print(f"\n📊 APLICANDO NORMALIZAÇÃO RNA-SEQ...")
    df_expr_clean = df_expr.copy()
    df_expr_clean = df_expr_clean.loc[~(df_expr_clean.isna().all(axis=1))]
    df_expr_clean = df_expr_clean.loc[~(df_expr_clean == 0).all(axis=1)]
    print(f"   • Genes mantidos: {df_expr_clean.shape[0]}")
    if df_expr_clean.isna().any().any():
        df_expr_clean = df_expr_clean.fillna(0)
    df_expr_clean = df_expr_clean.astype(float)

    library_sizes = df_expr_clean.sum(axis=0).replace(0, 1)
    cpm_matrix = df_expr_clean.div(library_sizes, axis=1) * 1_000_000
    log2cpm_matrix = np.log2(cpm_matrix + 1)
    df_transformed = pd.DataFrame(log2cpm_matrix, index=df_expr_clean.index, columns=df_expr_clean.columns)
    print(f"   ✅ log2(CPM+1) aplicado")

    return df_transformed


# ============================================================================
# 3. BATCH EFFECTS
# ============================================================================
def analyze_batch_effects(df_transformed, df_meta):
    print(f"\n🔍 ANALISANDO BATCH EFFECTS...")
    if 'Dataset' not in df_meta.columns:
        print("⚠️ Coluna 'Dataset' não encontrada nos metadados")
        return {'recommendations': ["Sem informação de dataset"]}

    common_samples = set(df_transformed.columns) & set(df_meta['Amostra'])
    df_expr_aligned = df_transformed[list(common_samples)]
    df_meta_aligned = df_meta[df_meta['Amostra'].isin(common_samples)].copy()
    df_meta_aligned.set_index('Amostra', inplace=True)
    df_meta_aligned = df_meta_aligned.loc[df_expr_aligned.columns]

    gene_variance = df_expr_aligned.var(axis=1)
    top_genes = gene_variance.nlargest(min(2000, len(gene_variance))).index
    df_pca_input = df_expr_aligned.loc[top_genes].T

    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(df_pca_input)
    pca_df = pd.DataFrame(pca_result[:, :3], columns=['PC1', 'PC2', 'PC3'], index=df_pca_input.index)
    pca_df['Dataset'] = df_meta_aligned.loc[pca_df.index, 'Dataset']

    datasets = sorted(df_meta_aligned['Dataset'].unique(), key=str)
    cores_ds = get_viridis_map(datasets)

    batch_separation = {}
    for ds in datasets:
        mask = pca_df['Dataset'] == ds
        if mask.sum() > 1:
            intra_dist = pdist(pca_result[mask.values, :3]).mean()
            other_mask = ~mask
            if other_mask.sum() > 0:
                inter_dists = [np.linalg.norm(pca_result[i, :3] - pca_result[j, :3])
                                for i in np.where(mask.values)[0] for j in np.where(other_mask.values)[0]]
                inter_dist = np.mean(inter_dists) if inter_dists else 0
                separation_ratio = inter_dist / intra_dist if intra_dist > 0 else 1
                batch_separation[ds] = {'n_samples': int(mask.sum()), 'separation_ratio': separation_ratio}

    recommendations = []
    if batch_separation:
        avg_separation = np.mean([v['separation_ratio'] for v in batch_separation.values()])
        if avg_separation > 1.5:
            recommendations.append(f"⚠️ Forte batch effect (ratio: {avg_separation:.2f}).")
        elif avg_separation > 1.2:
            recommendations.append(f"⚠️ Batch effect moderado (ratio: {avg_separation:.2f}).")
        else:
            recommendations.append(f"✅ Batch effects mínimos (ratio: {avg_separation:.2f}).")

    plt.figure(figsize=(13, 5.5))
    plt.subplot(1, 2, 1)
    for ds in datasets:
        mask = pca_df['Dataset'] == ds
        n = int(mask.sum())
        if n > 0:
            plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                        color=cores_ds[str(ds)], label=f'{ds} (n={n})',
                        alpha=0.8, s=70, edgecolors='white', linewidth=0.4)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Batch Effects -- PCA por Dataset')
    plt.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    ds_names = list(batch_separation.keys())
    ratios = [batch_separation[ds]['separation_ratio'] for ds in ds_names]
    cores_barras = [cores_ds[str(ds)] for ds in ds_names]
    plt.bar(range(len(ratios)), ratios, color=cores_barras, alpha=0.9, edgecolor='black', linewidth=0.5)
    plt.axhline(y=1.2, color='orange', linestyle='--', label='Moderado (1.2)')
    plt.axhline(y=1.5, color='red', linestyle='--', label='Forte (1.5)')
    plt.xticks(range(len(ratios)), ds_names, rotation=45, ha='right')
    plt.ylabel('Separation Ratio (inter/intra)')
    plt.title('Batch Effect Intensity by Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    batch_plot_path = os.path.join(PLOTS_DIR, "batch_effects.png")
    plt.savefig(batch_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {batch_plot_path}")

    return {'recommendations': recommendations}


# ============================================================================
# 4. SELEÇÃO DE GENES -- 3 critérios (GENES_CATEGORIAS vem de config.py)
# ============================================================================
def select_variable_genes(df_transformed, n_genes=3000, method='var', lista_genes=None):
    if method == 'lista_curada':
        if not lista_genes:
            raise ValueError("method='lista_curada' exige `lista_genes`.")
        genes_presentes = [g for g in lista_genes if g in df_transformed.index]
        faltantes = set(lista_genes) - set(genes_presentes)
        print(f"\n🎯 Genes da lista curada: {len(genes_presentes)}/{len(lista_genes)} presentes na matriz")
        if faltantes:
            print(f"   ⚠️ Não encontrados: {sorted(faltantes)}")
        return df_transformed.loc[genes_presentes]

    print(f"\n🎯 Selecionando {n_genes} genes por '{method}'...")
    if method == 'var':
        gene_scores = df_transformed.var(axis=1)
    elif method == 'mean':
        gene_scores = df_transformed.mean(axis=1)
    else:
        mean = df_transformed.mean(axis=1)
        std = df_transformed.std(axis=1)
        gene_scores = std / mean.abs()
    gene_scores = gene_scores.replace([np.inf, -np.inf], np.nan).dropna()
    gene_scores = gene_scores[gene_scores > 0]
    n_genes = min(n_genes, len(gene_scores))
    top_genes = gene_scores.nlargest(n_genes).index
    df_selected = df_transformed.loc[top_genes]
    print(f"   • Genes selecionados: {len(top_genes)}")
    return df_selected


# ============================================================================
# 5. PCA (com sufixo pra não sobrescrever entre critérios)
# ============================================================================
def perform_pca_analysis(df_input, df_meta, use_zscore_for_viz=True, sufixo=''):
    print(f"\n📊 EXECUTANDO ANÁLISE PCA...")

    if use_zscore_for_viz:
        X_scaled = StandardScaler().fit_transform(df_input.T)
        pca_input = X_scaled
    else:
        pca_input = df_input.T.values
    sample_names = df_input.columns

    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(pca_input)
    pca_df = pd.DataFrame(pca_result[:, :3], columns=['PC1', 'PC2', 'PC3'], index=sample_names)

    if 'Amostra' in df_meta.columns:
        meta_subset = df_meta[df_meta['Amostra'].isin(sample_names)].set_index('Amostra')
        for col in ['Dataset', 'Grupo', 'Condicao_Atividade', 'Grupo_Etario', 'Tratamento']:
            if col in meta_subset.columns:
                pca_df[col] = meta_subset.loc[pca_df.index, col]

    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)
    print(f"   📈 Variância explicada PC1-PC3: {cum_var[2]*100:.2f}%")

    tem_dataset = 'Dataset' in pca_df.columns
    cores_ds = get_viridis_map(pca_df['Dataset'].unique()) if tem_dataset else None

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(1, 6), var_explained[:5], alpha=0.75, color=PALETA(0.4))
    plt.plot(range(1, 6), cum_var[:5], 'o-', color=PALETA(0.9), label='Cumulative')
    plt.xlabel('PC'); plt.ylabel('Variance Explained'); plt.title(f'Scree Plot{sufixo}')
    plt.legend(); plt.grid(True, alpha=0.3)

    for subplot_idx, pc_y, pc_y_idx in [(2, 'PC2', 2), (3, 'PC3', 3)]:
        plt.subplot(1, 3, subplot_idx)
        if tem_dataset:
            for ds in sorted(pca_df['Dataset'].unique(), key=str):
                mask = pca_df['Dataset'] == ds
                n = int(mask.sum())
                plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, pc_y],
                            color=cores_ds[str(ds)], label=f'{ds} (n={n})',
                            alpha=0.8, s=70, edgecolors='white', linewidth=0.4)
        else:
            plt.scatter(pca_df['PC1'], pca_df[pc_y], color=PALETA(0.5), alpha=0.8, s=70)
        plt.xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
        plt.ylabel(f'{pc_y} ({var_explained[pc_y_idx-1]*100:.1f}%)')
        plt.title(f'PCA -- PC1 vs {pc_y}{sufixo}')
        if tem_dataset and subplot_idx == 2:
            plt.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    pca_plot_path = os.path.join(PLOTS_DIR, f"pca_analysis{sufixo}.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {pca_plot_path}")

    return {'pca_df': pca_df, 'pca_model': pca, 'variance_explained': var_explained}


# ============================================================================
# 6. K-MEANS
# ============================================================================
def perform_kmeans_clustering(pca_df, max_k=10, sufixo=''):
    print(f"\n🔍 EXECUTANDO CLUSTERING K-MEANS...")
    X = pca_df[['PC1', 'PC2', 'PC3']].values

    k_range = range(2, min(max_k + 1, len(X)))
    silhouette_scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        silhouette_scores.append(sil)

    optimal_k = list(k_range)[int(np.argmax(silhouette_scores))]
    print(f"   ✅ Número ótimo de clusters: k={optimal_k}")

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    final_labels = kmeans_final.fit_predict(X)
    pca_df['Cluster'] = final_labels
    pca_df['Cluster_Label'] = [f'C{c+1}' for c in final_labels]

    cores_cluster = PALETA(np.linspace(0, 0.92, optimal_k))
    plt.figure(figsize=(13, 5.5))
    plt.subplot(1, 2, 1)
    plt.plot(list(k_range), silhouette_scores, 'o-', color=PALETA(0.3), linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color=PALETA(0.85), linestyle='--', label=f'Optimal k={optimal_k}')
    plt.xlabel('Número de clusters (k)'); plt.ylabel('Silhouette Score')
    plt.title(f'K-Means: Optimal k{sufixo}'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for cluster in range(optimal_k):
        mask = pca_df['Cluster'] == cluster
        n = int(mask.sum())
        comp = ''
        if 'Dataset' in pca_df.columns:
            top_ds = pca_df.loc[mask, 'Dataset'].value_counts().idxmax()
            comp = f' -- {top_ds} predom.'
        plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                    color=cores_cluster[cluster], label=f'Cluster {cluster+1} (n={n}){comp}',
                    alpha=0.8, s=100, edgecolors='white', linewidth=0.6)
    centroids_pc_space = kmeans_final.cluster_centers_
    plt.scatter(centroids_pc_space[:, 0], centroids_pc_space[:, 1],
                c='red', marker='X', s=200, label='Centroids', edgecolors='black', linewidth=2)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title(f'K-Means Clusters (k={optimal_k}){sufixo}')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    kmeans_plot_path = os.path.join(PLOTS_DIR, f"kmeans_clusters{sufixo}.png")
    plt.savefig(kmeans_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico salvo: {kmeans_plot_path}")

    return {'pca_df': pca_df, 'optimal_k': optimal_k}


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================
def main():
    print("\n🚀 INICIANDO PASSO 02")

    df_expr, df_meta = load_and_prepare_data()
    if df_expr is None:
        return

    print(f"\n{'='*80}\n📊 NORMALIZAÇÃO RNA-SEQ\n{'='*80}")
    df_log2cpm = normalize_rnaseq_correct(df_expr)

    print(f"\n{'='*80}\n🔍 BATCH EFFECTS (top 2000 mais variáveis, referência)\n{'='*80}")
    batch_results = analyze_batch_effects(df_log2cpm, df_meta)

    print(f"\n{'='*80}\n🎯 PCA + K-MEANS -- 3 CRITÉRIOS DE SELEÇÃO DE GENES\n{'='*80}")
    criterios = [
        ('_top3000_variaveis', dict(method='var', n_genes=3000)),
        ('_top3000_mais_expressos', dict(method='mean', n_genes=3000)),
        ('_painel_muscular', dict(method='lista_curada', lista_genes=GENES_CATEGORIAS['musculares'])),
    ]

    resultados = {}
    for sufixo, kwargs in criterios:
        print(f"\n--- Critério: {sufixo.strip('_')} ---")
        df_selecionado = select_variable_genes(df_log2cpm, **kwargs)
        pca_results = perform_pca_analysis(df_selecionado, df_meta, use_zscore_for_viz=True, sufixo=sufixo)
        kmeans_results = perform_kmeans_clustering(pca_results['pca_df'], sufixo=sufixo)
        resultados[sufixo] = {'pca': pca_results, 'kmeans': kmeans_results}

    print(f"\n{'='*80}\n💾 SALVANDO\n{'='*80}")
    df_log2cpm.to_csv(os.path.join(OUTPUT_DIR, "MATRIZ_LOG2CPM.csv"))

    # metadado enriquecido com o cluster de CADA critério (uma coluna por critério)
    meta_enriched = df_meta.copy().set_index('Amostra')
    for sufixo, r in resultados.items():
        meta_enriched[f'Cluster{sufixo}'] = r['kmeans']['pca_df']['Cluster_Label']
    meta_enriched.to_csv(os.path.join(OUTPUT_DIR, "METADADOS_CLUSTERS_3CRITERIOS.csv"), sep=';')

    print(f"\n✅ CONCLUÍDO. Gráficos em: {PLOTS_DIR}")
    for rec in batch_results['recommendations']:
        print(f"   {rec}")


if __name__ == "__main__":
    main()