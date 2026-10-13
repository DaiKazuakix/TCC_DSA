# -*- coding: utf-8 -*-
"""
PASSO 04 -- Teste de sensibilidade do limiar de log2FC (cruzado idoso vs jovem)
================================================================================
Reaproveita a matriz pós-ComBat salva no checkpoint do passo 03, roda o teste
de expressão diferencial (jovem vs idoso) uma única vez e reclassifica o mesmo
resultado em vários limiares de |log2FC|, mantendo p-ajustado <= 0,01.

Este script não importa combat/umap (evita o conflito de versões de numpy
dessas bibliotecas) -- por isso a função de DE é copiada aqui.

Entrada:  dados/processados/_checkpoint_pos_etapaB.pkl  (gerado pelo passo 03)
Saída:    resultados/DE_CRUZADO_idoso_vs_jovem_BRUTO_sem_limiar.csv
          resultados/RESUMO_limiares_cruzado_idoso_jovem.csv
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import DIR_RESULTADOS, CHECKPOINT_ETAPA_B

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
OUTPUT_DIR = DIR_RESULTADOS
CHECKPOINT_PATH = CHECKPOINT_ETAPA_B

P_THRESHOLD = 0.01
# 0.5 é o critério confirmatório; os demais são exploratórios
LOG2FC_THRESHOLDS_PARA_TESTAR = [0.5, 0.25, 0.2, 0.15, 0.1]


def run_de_vetorizado(df_expr, amostras_c1, amostras_c2, nome_c1='c1', nome_c2='c2'):
    """Mesma engine de DE do passo 03 (teste t + FDR de Benjamini-Hochberg),
    copiada aqui para não depender de importar combat/umap."""
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
    p_ajustado = np.full_like(p_vals, np.nan)
    if mask_valido.sum() > 0:
        p_ajustado[mask_valido] = multipletests(p_vals[mask_valido], method='fdr_bh')[1]
    df_de['p_ajustado'] = p_ajustado

    return df_de


def main():
    print("=" * 80)
    print("TESTE DE LIMIAR -- CRUZADO IDOSO VS JOVEM (pós-ComBat)")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint não encontrado em: {CHECKPOINT_PATH}")
        print("   Rode 03_pipeline_consolidado.py (pelo menos até a Etapa B) "
              "-- ele salva esse arquivo automaticamente.")
        return

    print(f"\n📥 Carregando checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)

    df_corrected_b = checkpoint['df_corrected_b']
    df_meta_b = checkpoint['df_meta_b']
    print(f"✅ df_corrected_b: {df_corrected_b.shape[0]} genes × {df_corrected_b.shape[1]} amostras")

    amostras_idoso = [a for a in df_meta_b[df_meta_b['Grupo_Etario'] == 'Idoso'].index
                       if a in df_corrected_b.columns]
    amostras_jovem = [a for a in df_meta_b[df_meta_b['Grupo_Etario'] == 'Jovem adulto'].index
                       if a in df_corrected_b.columns]
    print(f"   Idoso: {len(amostras_idoso)} | Jovem adulto: {len(amostras_jovem)}")

    print("\n🔬 Rodando DE (jovem vs idoso) uma única vez -- "
          "os limiares abaixo só reclassificam o mesmo resultado.")
    df_de = run_de_vetorizado(df_corrected_b, amostras_jovem, amostras_idoso, 'jovem', 'idoso')

    # Salva o resultado bruto (log2FC + p_ajustado de todos os genes) para
    # permitir testar outros limiares sem rodar o teste de novo.
    caminho_bruto = os.path.join(OUTPUT_DIR, "DE_CRUZADO_idoso_vs_jovem_BRUTO_sem_limiar.csv")
    df_de.to_csv(caminho_bruto, index=False)
    print(f"💾 Resultado bruto salvo em: {os.path.basename(caminho_bruto)}")

    print("\n" + "-" * 80)
    print(f"{'log2FC >=':>10} | {'UP':>6} | {'DOWN':>6} | {'total sig.':>10}")
    print("-" * 80)

    resumo = []
    for limiar in LOG2FC_THRESHOLDS_PARA_TESTAR:
        significativo = (df_de['log2FC'].abs() >= limiar) & (df_de['p_ajustado'] <= P_THRESHOLD)
        regulacao = pd.Series('ns', index=df_de.index)
        regulacao[significativo & (df_de['log2FC'] > 0)] = 'UP'
        regulacao[significativo & (df_de['log2FC'] < 0)] = 'DOWN'

        contagem = regulacao.value_counts()
        n_up = int(contagem.get('UP', 0))
        n_down = int(contagem.get('DOWN', 0))
        n_total = n_up + n_down

        print(f"{limiar:>10} | {n_up:>6} | {n_down:>6} | {n_total:>10}")
        resumo.append({'log2FC_threshold': limiar, 'p_threshold': P_THRESHOLD,
                        'UP': n_up, 'DOWN': n_down, 'total_significativo': n_total})

    print("-" * 80)

    df_resumo = pd.DataFrame(resumo)
    caminho_resumo = os.path.join(OUTPUT_DIR, "RESUMO_limiares_cruzado_idoso_jovem.csv")
    df_resumo.to_csv(caminho_resumo, index=False)
    print(f"\n💾 Resumo salvo em: {os.path.basename(caminho_resumo)}")
    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()
