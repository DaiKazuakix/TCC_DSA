# -*- coding: utf-8 -*-
"""
PASSO 05 -- DESeq2 (análise principal) + volcano plot da Figura 10
===================================================================
Três análises via DESeq2 (pydeseq2) sobre a matriz de contagens brutas:

  A) Idoso vs Jovem CRUZADO (4 datasets) -- análise principal do TCC.
     Usa o dataset como covariável no próprio modelo
     (~ Dataset + Grupo_Etario), sem ComBat, depois de checar se a idade
     não está totalmente confundida com o dataset.
  B) Metformina vs Placebo -- dentro do GSE157585.
  C) Oleuropeína vs Placebo -- dentro do GSE318937, com desenho pareado
     (Sujeito_ID na fórmula). Não gerou saída e não foi incorporada ao TCC;
     fica desligada por padrão (RODAR_OLEUROPEINA = False).

Ao final, gera o volcano plot de dois painéis usado na Figura 10 do TCC:
  A) Idoso vs jovem (cruzado, DESeq2)
  B) Metformina vs placebo (GSE157585, DESeq2)

O volcano lê os CSVs já salvos em resultados/, então pode ser refeito sem
rodar o DESeq2 de novo: basta colocar RODAR_DESEQ2 = False.

Requer: pip install pydeseq2  (só se RODAR_DESEQ2 = True)

Entrada:  dados/processados/MATRIZ_INTEGRADA_CONTAGENS.csv
          metadados/METADADOS_ENRIQUECIDOS_v2.csv
Saída:    resultados/DESeq2_CRUZADO_idoso_vs_jovem.csv
          resultados/DESeq2_metformina_vs_placebo_157585.csv
          resultados/DESeq2_OLE_vs_placebo_318937_PAREADO.csv  (só se RODAR_OLEUROPEINA)
          resultados/figuras/05_deseq2/volcano_deseq2_cruzado_e_metformina.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (DIR_RESULTADOS, DIR_FIGURAS, MATRIZ_CONTAGENS, METADADOS,
                    DESEQ2_CRUZADO, DESEQ2_METFORMINA)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
RODAR_DESEQ2 = True     # False = só refaz o volcano a partir dos CSVs salvos
GERAR_VOLCANO = True
RODAR_OLEUROPEINA = False   # comparação C -- não incorporada ao TCC

LOG2FC_THRESHOLD = 0.5
PADJ_THRESHOLD = 0.01

PLOTS_DIR = os.path.join(DIR_FIGURAS, "05_deseq2")
os.makedirs(PLOTS_DIR, exist_ok=True)
CAMINHO_VOLCANO = os.path.join(PLOTS_DIR, "volcano_deseq2_cruzado_e_metformina.png")

# Genes rotulados no painel A (os mesmos da Figura 10): os de menor
# p-ajustado em cada direção + genes discutidos no texto.
GENES_ROTULO_CRUZADO = ['UNC13C', 'TTN-AS1', 'SKAP2', 'ZFPM1', 'DMRT2',
                        'NDUFB9', 'PGAM2', 'CDKN1A', 'SAA1']
# No painel B são rotulados todos os genes significativos (são só 3).
GENES_ROTULO_METFORMINA = None

# Deslocamento de cada rótulo em relação ao ponto (em pontos tipográficos:
# x positivo = direita, y positivo = acima). Genes fora da lista usam o padrão.
DESLOCAMENTO_PADRAO = (10, 8)
DESLOCAMENTO_ROTULO = {
    'UNC13C': (-14, -4), 'TTN-AS1': (-16, -8), 'SKAP2': (6, 8),
    'ZFPM1': (10, 6), 'DMRT2': (4, 10), 'NDUFB9': (4, 2),
    'PGAM2': (10, 14), 'CDKN1A': (10, 10), 'SAA1': (8, 10),
    'PTPRCAP': (10, 6), 'ELAPOR2': (10, -14), 'SCTR-AS1': (-14, 2),
}

PALETA = plt.cm.viridis
COR_UP = PALETA(0.85)
COR_DOWN = PALETA(0.15)
COR_NS = '#c7c7c7'


# ============================================================================
# 1. DESeq2
# ============================================================================
def checar_confundimento(df_meta, amostras, col_batch, col_covar):
    """Confirma se a covariável varia dentro de pelo menos um batch antes de
    tentar protegê-la na fórmula do modelo."""
    tab = pd.crosstab(df_meta.loc[amostras, col_batch], df_meta.loc[amostras, col_covar])
    identificavel = (tab.gt(0).sum(axis=1) > 1).any()
    print(tab.to_string())
    print(f"   {'✅ identificável' if identificavel else '🚫 confundida -- NÃO prosseguir'}")
    return identificavel


def rodar_deseq2(df_counts, df_meta, amostras, formula_design, contraste, nome_saida,
                 covar_extra_cols=None):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts = df_counts[amostras].T.round().astype(int)
    meta = df_meta.loc[amostras, :].copy()
    if covar_extra_cols:
        for c in covar_extra_cols:
            meta[c] = meta[c].astype(str)  # fatores do design não aceitam tipos mistos
    dds = DeseqDataSet(counts=counts, metadata=meta, design=formula_design)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=contraste)
    stats.summary()
    res = stats.results_df
    res.to_csv(nome_saida)
    n_sig = ((res['padj'] <= PADJ_THRESHOLD) & (res['log2FoldChange'].abs() >= LOG2FC_THRESHOLD)).sum()
    print(f"   ✅ {n_sig} genes significativos -- salvo em {os.path.basename(nome_saida)}\n")
    return res


def executar_deseq2():
    df_counts = pd.read_csv(MATRIZ_CONTAGENS, index_col=0)
    df_meta = pd.read_csv(METADADOS, sep=';', encoding='latin1', index_col=0)

    # ------------------------------------------------------------------
    # A) Idoso vs Jovem CRUZADO (4 datasets)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("A) Idoso vs Jovem -- cruzado, 4 datasets")
    print("=" * 70)
    col_idade = 'Idade_x_Treino' if 'Idade_x_Treino' in df_meta.columns else 'Grupo_Etario'
    amostras_cruzado = df_meta[df_meta[col_idade].isin(
        ['Idoso_baseline', 'Idoso', 'Jovem adulto']
    )].index
    amostras_cruzado = [a for a in amostras_cruzado if a in df_counts.columns]

    print(f"n = {len(amostras_cruzado)} amostras")
    identificavel = checar_confundimento(df_meta, amostras_cruzado, 'Dataset', col_idade)

    if identificavel:
        nivel_idoso = 'Idoso' if 'Idoso' in df_meta[col_idade].unique() else 'Idoso_baseline'
        rodar_deseq2(
            df_counts, df_meta, amostras_cruzado,
            formula_design="~ Dataset + " + col_idade,
            contraste=[col_idade, nivel_idoso, 'Jovem adulto'],
            nome_saida=DESEQ2_CRUZADO,
            covar_extra_cols=['Dataset', col_idade],
        )
    else:
        print("   ⚠️ Pulando -- confundimento total entre Dataset e grupo etário "
              "inviabiliza a fórmula do modelo.\n")

    # ------------------------------------------------------------------
    # B) Metformina vs Placebo -- dentro do GSE157585
    # ------------------------------------------------------------------
    print("=" * 70)
    print("B) Metformina vs Placebo -- GSE157585")
    print("=" * 70)
    amostras_metf = df_meta[
        (df_meta['Dataset'] == 'GSE157585') & (df_meta['Tratamento'].isin(['Metformina', 'Placebo']))
    ].index
    amostras_metf = [a for a in amostras_metf if a in df_counts.columns]
    print(f"n = {len(amostras_metf)} amostras")

    rodar_deseq2(
        df_counts, df_meta, amostras_metf,
        formula_design="~ Tratamento",
        contraste=['Tratamento', 'Metformina', 'Placebo'],
        nome_saida=DESEQ2_METFORMINA,
        covar_extra_cols=['Tratamento'],
    )

    # ------------------------------------------------------------------
    # C) Oleuropeína vs Placebo -- dentro do GSE318937, pareado (Sujeito_ID)
    # ------------------------------------------------------------------
    if not RODAR_OLEUROPEINA:
        print("C) Oleuropeína vs Placebo -- desligada (RODAR_OLEUROPEINA = False)\n")
        return

    print("=" * 70)
    print("C) Oleuropeína vs Placebo -- GSE318937 (pareado por Sujeito_ID)")
    print("=" * 70)
    amostras_ole = df_meta[
        (df_meta['Dataset'] == 'GSE318937') & (df_meta['Tratamento'].isin(['Oleuropeina', 'Placebo']))
    ].index
    amostras_ole = [a for a in amostras_ole if a in df_counts.columns]
    print(f"n = {len(amostras_ole)} amostras")

    # só sujeitos com as duas condições entram no desenho pareado
    contagem_por_sujeito = df_meta.loc[amostras_ole].groupby('Sujeito_ID')['Tratamento'].nunique()
    sujeitos_completos = contagem_por_sujeito[contagem_por_sujeito == 2].index
    amostras_ole_pareado = [a for a in amostras_ole if df_meta.loc[a, 'Sujeito_ID'] in sujeitos_completos]
    print(f"   {len(amostras_ole_pareado)} amostras em sujeitos com as duas condições completas")

    try:
        rodar_deseq2(
            df_counts, df_meta, amostras_ole_pareado,
            formula_design="~ Sujeito_ID + Condicao_Atividade + Tratamento",
            contraste=['Tratamento', 'Oleuropeina', 'Placebo'],
            nome_saida=os.path.join(DIR_RESULTADOS, "DESeq2_OLE_vs_placebo_318937_PAREADO.csv"),
            covar_extra_cols=['Sujeito_ID', 'Condicao_Atividade', 'Tratamento'],
        )
    except Exception as e:
        print(f"   ⚠️ Comparação C não pôde ser ajustada: {e}\n")


# ============================================================================
# 2. VOLCANO PLOT (Figura 10)
# ============================================================================
def carregar_resultado_deseq2(caminho):
    """Lê o CSV do pydeseq2 (índice = símbolo do gene) e classifica cada gene
    em UP / DOWN / ns pelo critério confirmatório."""
    df = pd.read_csv(caminho, index_col=0)
    df.index.name = 'gene'
    df = df.reset_index().dropna(subset=['padj', 'log2FoldChange'])

    sig = (df['padj'] <= PADJ_THRESHOLD) & (df['log2FoldChange'].abs() >= LOG2FC_THRESHOLD)
    df['regulacao'] = 'ns'
    df.loc[sig & (df['log2FoldChange'] > 0), 'regulacao'] = 'UP'
    df.loc[sig & (df['log2FoldChange'] < 0), 'regulacao'] = 'DOWN'

    menor_p = df.loc[df['padj'] > 0, 'padj'].min()
    df['neg_log10_padj'] = -np.log10(df['padj'].replace(0, menor_p))
    return df


def plot_volcano_painel(ax, df, titulo, genes_rotulo=None):
    # ns primeiro (fica por baixo), depois DOWN e UP
    for regulacao, cor, tamanho in [('ns', COR_NS, 6), ('DOWN', COR_DOWN, 8), ('UP', COR_UP, 8)]:
        sub = df[df['regulacao'] == regulacao]
        ax.scatter(sub['log2FoldChange'], sub['neg_log10_padj'], color=cor, s=tamanho,
                   alpha=0.85, linewidths=0, label=f"{regulacao} ({len(sub)})", rasterized=True)

    ax.axhline(-np.log10(PADJ_THRESHOLD), color='gray', linestyle='--', linewidth=0.9)
    ax.axvline(LOG2FC_THRESHOLD, color='gray', linestyle='--', linewidth=0.9)
    ax.axvline(-LOG2FC_THRESHOLD, color='gray', linestyle='--', linewidth=0.9)

    if genes_rotulo is None:
        rotular = df[df['regulacao'] != 'ns']
    else:
        rotular = df[df['gene'].isin(genes_rotulo)]
        faltando = set(genes_rotulo) - set(rotular['gene'])
        if faltando:
            print(f"   ⚠️ Genes para rótulo não encontrados: {sorted(faltando)}")

    for _, row in rotular.iterrows():
        dx, dy = DESLOCAMENTO_ROTULO.get(row['gene'], DESLOCAMENTO_PADRAO)
        ax.annotate(row['gene'], (row['log2FoldChange'], row['neg_log10_padj']),
                    xytext=(dx, dy), textcoords='offset points', fontsize=8,
                    ha='right' if dx < 0 else 'left', va='center',
                    arrowprops=dict(arrowstyle='-', color='#444444', lw=0.6, shrinkA=0, shrinkB=2))

    y_max = df['neg_log10_padj'].max()
    ax.set_ylim(top=y_max * 1.06)

    ax.set_xlabel('log2(Fold Change)')
    ax.set_ylabel('-log10(p-valor ajustado)')
    ax.set_title(titulo)
    ax.legend(loc='upper left', fontsize=8.5, markerscale=1.6, frameon=True)
    ax.grid(alpha=0.25)


def gerar_volcano_figura10():
    for caminho in (DESEQ2_CRUZADO, DESEQ2_METFORMINA):
        if not os.path.exists(caminho):
            print(f"❌ Não encontrei {caminho} -- rode com RODAR_DESEQ2 = True primeiro.")
            return

    df_cruzado = carregar_resultado_deseq2(DESEQ2_CRUZADO)
    df_metf = carregar_resultado_deseq2(DESEQ2_METFORMINA)

    for nome, df in [('Cruzado idoso vs jovem', df_cruzado), ('Metformina vs placebo', df_metf)]:
        contagem = df['regulacao'].value_counts()
        print(f"   {nome}: UP={contagem.get('UP', 0)} | DOWN={contagem.get('DOWN', 0)} "
              f"| ns={contagem.get('ns', 0)}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    plot_volcano_painel(axes[0], df_cruzado, 'A  Idoso vs jovem (cruzado, DESeq2)',
                        GENES_ROTULO_CRUZADO)
    plot_volcano_painel(axes[1], df_metf, 'B  Metformina vs placebo (GSE157585, DESeq2)',
                        GENES_ROTULO_METFORMINA)
    plt.tight_layout()
    plt.savefig(CAMINHO_VOLCANO, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ Volcano salvo: {CAMINHO_VOLCANO}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    if RODAR_DESEQ2:
        executar_deseq2()
    if GERAR_VOLCANO:
        print("=" * 70)
        print("Volcano plot -- Figura 10 (DESeq2)")
        print("=" * 70)
        gerar_volcano_figura10()
    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()
