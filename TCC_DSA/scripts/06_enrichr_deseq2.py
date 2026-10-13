# -*- coding: utf-8 -*-
"""
PASSO 06 -- Análise funcional (Enrichr) sobre os DEGs do DESeq2 cruzado
========================================================================
Roda o enriquecimento funcional (Enrichr, Ma'ayan Lab) separadamente para
os genes aumentados e reduzidos no grupo idoso (DESeq2, |log2FC| >= 0,5,
p-ajustado <= 0,01, n = 1.163 UP + 684 DOWN), nas quatro bibliotecas do
texto: GO Biological Process 2023, KEGG 2021 Human, Reactome 2022 e
MSigDB Hallmark 2020. Considera significativo padj <= 0,05 e sobreposição
mínima de 5 genes, conforme a Metodologia do TCC.

Entrada:  resultados/DESeq2_CRUZADO_idoso_vs_jovem.csv  (passo 05)
Saída:    resultados/enrichr_deseq2/ (ENRICHR_*.csv e TERMOS_PRONTOS_*.txt)
"""
import os
import sys
import time
import pandas as pd
import requests

from config import DIR_RESULTADOS, DESEQ2_CRUZADO

CSV_ENTRADA = DESEQ2_CRUZADO
OUT = os.path.join(DIR_RESULTADOS, 'enrichr_deseq2')
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(CSV_ENTRADA):
    print(f"❌ Não encontrei {CSV_ENTRADA} -- rode 05_deseq2_principal.py primeiro.")
    sys.exit(1)

LFC, PADJ_DEG = 0.5, 0.01          # limiar dos DEGs (conforme o TCC)
PADJ_TERMO, MIN_OVERLAP = 0.05, 5  # limiar dos termos (conforme o TCC)

BIBLIOTECAS = [
    'GO_Biological_Process_2023',
    'KEGG_2021_Human',
    'Reactome_2022',
    'MSigDB_Hallmark_2020',
]

ADD_URL = 'https://maayanlab.cloud/Enrichr/addList'
ENRICH_URL = 'https://maayanlab.cloud/Enrichr/enrich'


def enrichr_add_list(genes, description):
    payload = {'list': (None, '\n'.join(genes)), 'description': (None, description)}
    r = requests.post(ADD_URL, files=payload, timeout=30)
    r.raise_for_status()
    return r.json()['userListId']


def enrichr_get(user_list_id, library):
    r = requests.get(ENRICH_URL, params={'userListId': user_list_id, 'backgroundType': library}, timeout=30)
    r.raise_for_status()
    linhas = []
    for row in r.json().get(library, []):
        linhas.append({'termo': row[1], 'p_valor': row[2], 'z_score': row[3],
                       'combined_score': row[4], 'n_genes_sobrepostos': len(row[5]),
                       'genes_sobrepostos': ', '.join(row[5]), 'p_ajustado': row[6]})
    return pd.DataFrame(linhas)


def rodar(nome_lista, genes):
    print(f"\n{'='*80}\n{nome_lista} ({len(genes)} genes)\n{'='*80}")
    uid = enrichr_add_list(genes, nome_lista)
    time.sleep(1)
    termos_para_texto = []
    for lib in BIBLIOTECAS:
        df = enrichr_get(uid, lib)
        df.to_csv(os.path.join(OUT, f'ENRICHR_{nome_lista}_{lib}.csv'), index=False)
        sig = df[(df.p_ajustado <= PADJ_TERMO) & (df.n_genes_sobrepostos >= MIN_OVERLAP)].sort_values('p_ajustado')
        print(f"\n{lib}: {len(sig)} termos significativos (de {len(df)} testados)")
        for r in sig.head(15).itertuples():
            print(f"   [padj={r.p_ajustado:.2e}, n={r.n_genes_sobrepostos}] {r.termo}")
        termos_para_texto += [f"{r.termo} ({lib.split('_')[0]}, padj={r.p_ajustado:.1e}, n={r.n_genes_sobrepostos})"
                              for r in sig.head(8).itertuples()]
        time.sleep(0.5)
    with open(os.path.join(OUT, f'TERMOS_PRONTOS_{nome_lista}.txt'), 'w', encoding='utf-8') as f:
        f.write('; '.join(termos_para_texto))
    print(f"\n>>> Resumo dos principais termos salvo em: TERMOS_PRONTOS_{nome_lista}.txt")
    return termos_para_texto


def main():
    d = pd.read_csv(CSV_ENTRADA)
    sig = d[(d.log2FoldChange.abs() >= LFC) & (d.padj <= PADJ_DEG)]
    up = sig[sig.log2FoldChange > 0].sort_values('padj').Gene_Symbol.tolist()
    down = sig[sig.log2FoldChange < 0].sort_values('padj').Gene_Symbol.tolist()
    print(f"DEGs (|log2FC|>={LFC}, padj<={PADJ_DEG}): {len(sig)} total | UP={len(up)} | DOWN={len(down)}")

    rodar('UP_idoso_DESeq2', up)
    rodar('DOWN_idoso_DESeq2', down)
    print(f"\n{'='*80}\nConcluído. Resultados em: {OUT}\n" + '='*80)


if __name__ == '__main__':
    main()