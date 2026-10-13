# -*- coding: utf-8 -*-
"""
EXPLORATÓRIO -- Validação por DESeq2 do contraste idoso vs jovem no GSE157585
==========================================================================
Repete o contraste idoso vs jovem dentro do GSE157585 usando DESeq2 (via
pydeseq2), como método estatístico independente do teste t sobre log2CPM
do passo 03, e mede a concordância entre os dois (interseção e Jaccard).

DESeq2 exige CONTAGENS BRUTAS (inteiras) -- o script verifica isso antes de
rodar e aborta se a matriz parecer já normalizada.

Requer: pip install pydeseq2

Entrada:  dados/processados/MATRIZ_INTEGRADA_CONTAGENS.csv
          metadados/METADADOS_ENRIQUECIDOS_v2.csv
          resultados/DE_GSE157585_idoso_vs_jovem_157585.csv  (passo 03)
Saída:    resultados/DESeq2_idoso_vs_jovem_157585.csv
"""
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DIR_RESULTADOS, MATRIZ_CONTAGENS, METADADOS

# ------------------------------------------------------------------
# 1. Carregar contagens brutas (genes x amostras) e metadados
# ------------------------------------------------------------------
CAMINHO_CONTAGENS_BRUTAS = MATRIZ_CONTAGENS  # genes x amostras, dos 4 datasets
CAMINHO_METADADOS = METADADOS
CAMINHO_TTEST = os.path.join(DIR_RESULTADOS, "DE_GSE157585_idoso_vs_jovem_157585.csv")
CAMINHO_SAIDA = os.path.join(DIR_RESULTADOS, "DESeq2_idoso_vs_jovem_157585.csv")


df_counts_genes_x_amostras = pd.read_csv(CAMINHO_CONTAGENS_BRUTAS, index_col=0)
df_meta = pd.read_csv(CAMINHO_METADADOS, sep=';', encoding='latin1', index_col=0)

# ------------------------------------------------------------------
# 1b. Checagem de sanidade -- confirma que isso É contagem bruta, e não
# algo já normalizado (log2CPM, por exemplo). DESeq2 exige contagens
# brutas; se os dados já estiverem em log2, o resultado sai sem sentido
# e SEM avisar sozinho -- por isso a checagem manual aqui.
# ------------------------------------------------------------------
amostra_teste = df_counts_genes_x_amostras.iloc[:, 0]
valor_max = amostra_teste.max()
tem_decimais = not (amostra_teste.dropna() % 1 == 0).all()
print(f"Diagnóstico da matriz '{os.path.basename(CAMINHO_CONTAGENS_BRUTAS)}':")
print(f"   valor máximo na 1ª coluna: {valor_max:.2f}")
print(f"   contém valores não-inteiros: {tem_decimais}")
if valor_max < 50 or tem_decimais:
    print("\n⚠️  ATENÇÃO: isso NÃO parece contagem bruta -- parece já normalizado/log-transformado")
    print("    (contagem bruta real costuma ter valores na casa de centenas/milhares e ser sempre inteira).")
    print("    Confirme a origem do arquivo antes de continuar. Abortando por segurança.")
    raise SystemExit(1)
else:
    print("   ✅ Parece contagem bruta (valores altos, inteiros) -- prosseguindo.\n")

# restringe a GSE157585, baseline (mesmo recorte do t-test original)
amostras_157585 = df_meta[
    (df_meta['Dataset'] == 'GSE157585') &
    (df_meta.get('Idade_x_Treino', df_meta.get('Grupo_Etario')).isin(
        ['Idoso_baseline', 'Jovem adulto'] if 'Idade_x_Treino' in df_meta.columns
        else ['Idoso', 'Jovem adulto']
    ))
].index
amostras_157585 = [a for a in amostras_157585 if a in df_counts_genes_x_amostras.columns]

# renomeia categorias sem espaço (pydeseq2/patsy não gostam de espaço no nível do fator)
meta_sub = df_meta.loc[amostras_157585, ['Dataset']].copy()
rotulo_idade = (df_meta.loc[amostras_157585, 'Idade_x_Treino']
                if 'Idade_x_Treino' in df_meta.columns
                else df_meta.loc[amostras_157585, 'Grupo_Etario'])
meta_sub['Grupo'] = rotulo_idade.replace({
    'Idoso_baseline': 'Idoso', 'Idoso': 'Idoso',
    'Jovem adulto': 'Jovem'
})

# pydeseq2 espera amostras nas LINHAS (oposto da convenção genes x amostras
# que usamos no resto do pipeline) -- por isso a transposição abaixo
counts_amostras_x_genes = df_counts_genes_x_amostras[amostras_157585].T
counts_amostras_x_genes = counts_amostras_x_genes.round().astype(int)  # DESeq2 exige inteiros

# ------------------------------------------------------------------
# 2. Rodar DESeq2
# ------------------------------------------------------------------
dds = DeseqDataSet(
    counts=counts_amostras_x_genes,
    metadata=meta_sub,
    design_factors="Grupo",
)
dds.deseq2()

stats = DeseqStats(dds, contrast=["Grupo", "Idoso", "Jovem"])
stats.summary()
resultado_deseq2 = stats.results_df  # colunas: baseMean, log2FoldChange, pvalue, padj...

resultado_deseq2.to_csv(CAMINHO_SAIDA)
print(f"DESeq2: {(resultado_deseq2['padj'] <= 0.01).sum()} genes com padj <= 0.01 "
      f"(sem filtro de log2FC ainda)")

# ------------------------------------------------------------------
# 3. Comparar com o resultado do teste t já existente
# ------------------------------------------------------------------
df_ttest = pd.read_csv(CAMINHO_TTEST)
genes_ttest_sig = set(df_ttest.loc[df_ttest['regulacao'] != 'ns', 'gene'])

sig_deseq2 = resultado_deseq2[
    (resultado_deseq2['padj'] <= 0.01) & (resultado_deseq2['log2FoldChange'].abs() >= 0.5)
]
genes_deseq2_sig = set(sig_deseq2.index)

intersecao = genes_ttest_sig & genes_deseq2_sig
uniao = genes_ttest_sig | genes_deseq2_sig
print(f"\nTeste t:  {len(genes_ttest_sig)} genes significativos")
print(f"DESeq2:   {len(genes_deseq2_sig)} genes significativos")
print(f"Interseção: {len(intersecao)} genes")
print(f"Jaccard (concordância): {len(intersecao)/len(uniao)*100:.1f}%")