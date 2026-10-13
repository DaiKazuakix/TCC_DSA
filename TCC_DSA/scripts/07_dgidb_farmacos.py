# -*- coding: utf-8 -*-
"""
PASSO 07 -- Mapeamento gene -> fármaco (DGIdb) dos DEGs do DESeq2 cruzado
==========================================================================
Lê o resultado do DESeq2 idoso vs jovem (passo 05), separa os genes
aumentados (UP) e reduzidos (DOWN) no grupo idoso pelo critério
confirmatório (|log2FC| >= 0,5; p-ajustado < 0,01) e consulta a API
GraphQL do DGIdb em lotes.

Entrada:  resultados/DESeq2_CRUZADO_idoso_vs_jovem.csv
Saída:    resultados/dgidb/ (genes_UP.txt, genes_DOWN.txt, DGIDB_UP.csv,
          DGIDB_DOWN.csv, DGIDB_AMBOS.csv)
"""

import os
import json
import time
import requests
import pandas as pd

from config import DIR_RESULTADOS, DESEQ2_CRUZADO

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
OUTPUT_DIR = os.path.join(DIR_RESULTADOS, "dgidb")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Limiares
LOG2FC_CORTE = 0.5
PADJ_CORTE = 0.01

# DGIdb
DGIDB_URL = 'https://dgidb.org/api/graphql'
TAMANHO_CHUNK = 100
PAUSA_ENTRE_CHUNKS = 1.5

# ============================================================================
# 1. CARREGAR E SEPARAR GENES
# ============================================================================
def carregar_e_separar():
    caminho = DESEQ2_CRUZADO
    if not os.path.exists(caminho):
        print(f"❌ Não encontrei {caminho} -- rode 05_deseq2_principal.py primeiro.")
        return None
    print(f"📥 Carregando {caminho}...")

    # pydeseq2 salva com index = gene name
    df = pd.read_csv(caminho, index_col=0)
    df.index.name = 'gene'
    df = df.reset_index()

    print(f"✅ {len(df)} genes carregados")
    print(f"   Colunas: {list(df.columns)}")

    # Verificar colunas necessárias
    for col in ['gene', 'log2FoldChange', 'padj']:
        if col not in df.columns:
            print(f"❌ Coluna '{col}' não encontrada. Abortando.")
            return None

    # Filtrar significativos
    mask_sig = (df['padj'] < PADJ_CORTE) & (df['log2FoldChange'].abs() >= LOG2FC_CORTE)
    df_sig = df[mask_sig].copy()

    genes_up = df_sig[df_sig['log2FoldChange'] > 0]['gene'].dropna().unique().tolist()
    genes_down = df_sig[df_sig['log2FoldChange'] < 0]['gene'].dropna().unique().tolist()

    print(f"\n✅ Genes UP (aumentados no idoso): {len(genes_up)}")
    print(f"✅ Genes DOWN (reduzidos no idoso): {len(genes_down)}")
    print(f"✅ Total significativo: {len(genes_up) + len(genes_down)}")

    # Salvar listas
    pd.Series(genes_up).to_csv(os.path.join(OUTPUT_DIR, "genes_UP.txt"),
                                index=False, header=False)
    pd.Series(genes_down).to_csv(os.path.join(OUTPUT_DIR, "genes_DOWN.txt"),
                                  index=False, header=False)
    print(f"💾 Listas salvas em {OUTPUT_DIR}")

    return genes_up, genes_down


# ============================================================================
# 2. DGIDB
# ============================================================================
QUERY_DGIDB = """
{
  genes(names: %s) {
    nodes {
      name
      interactions {
        drug { name conceptId }
        interactionScore
        interactionTypes { type directionality }
        sources { sourceDbName }
      }
    }
  }
}
"""


def consultar_dgidb_chunk(genes_chunk):
    genes_json = json.dumps(genes_chunk)
    query = QUERY_DGIDB % genes_json
    try:
        r = requests.post(DGIDB_URL, json={'query': query}, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return []

    linhas = []
    for gene_node in data.get('data', {}).get('genes', {}).get('nodes', []):
        for interacao in gene_node.get('interactions', []):
            drug = interacao.get('drug') or {}
            tipos = interacao.get('interactionTypes') or []
            fontes = interacao.get('sources') or []
            linhas.append({
                'gene': gene_node['name'],
                'farmaco': drug.get('name'),
                'farmaco_id': drug.get('conceptId'),
                'interaction_score': interacao.get('interactionScore'),
                'tipos_interacao': '; '.join(
                    f"{t.get('type')} ({t.get('directionality')})" for t in tipos if t
                ),
                'fontes': '; '.join(f.get('sourceDbName', '') for f in fontes if f),
            })
    return linhas


def consultar_dgidb_lista(genes, rotulo):
    if not genes:
        print(f"   ⚠️ Lista {rotulo} vazia -- pulando.")
        return pd.DataFrame()

    print(f"\n🔎 DGIdb: {len(genes)} {rotulo}, chunks de {TAMANHO_CHUNK}...")
    todas = []
    n_chunks = (len(genes) + TAMANHO_CHUNK - 1) // TAMANHO_CHUNK

    for i in range(0, len(genes), TAMANHO_CHUNK):
        chunk = genes[i:i + TAMANHO_CHUNK]
        idx = i // TAMANHO_CHUNK + 1
        print(f"   [{idx}/{n_chunks}] {len(chunk)} genes...")
        todas.extend(consultar_dgidb_chunk(chunk))
        time.sleep(PAUSA_ENTRE_CHUNKS)

    df = pd.DataFrame(todas)
    if not df.empty:
        df = df.sort_values('interaction_score', ascending=False).reset_index(drop=True)
    print(f"   ✅ {len(df)} interações encontradas")
    return df


# ============================================================================
# 3. MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("DGIdb -- GENES DO DESeq2 (idoso vs jovem cruzado)")
    print("=" * 80)

    resultado = carregar_e_separar()
    if resultado is None:
        return
    genes_up, genes_down = resultado

    # UP = aumentados no idoso (candidatos a fármacos que reduzam)
    print("\n" + "=" * 80)
    print("EIXO UP (aumentados no idoso)")
    print("=" * 80)
    df_up = consultar_dgidb_lista(genes_up, "UP")
    if not df_up.empty:
        caminho = os.path.join(OUTPUT_DIR, "DGIDB_UP.csv")
        df_up.to_csv(caminho, index=False)
        print(f"   💾 {caminho}")
        print("\n   TOP 30:")
        print(df_up.head(30).to_string(index=False))

    # DOWN = reduzidos no idoso (candidatos a fármacos que aumentem)
    print("\n" + "=" * 80)
    print("EIXO DOWN (reduzidos no idoso)")
    print("=" * 80)
    df_down = consultar_dgidb_lista(genes_down, "DOWN")
    if not df_down.empty:
        caminho = os.path.join(OUTPUT_DIR, "DGIDB_DOWN.csv")
        df_down.to_csv(caminho, index=False)
        print(f"   💾 {caminho}")
        print("\n   TOP 30:")
        print(df_down.head(30).to_string(index=False))

    # Resumo
    if not df_up.empty and not df_down.empty:
        f_up = set(df_up['farmaco'].dropna())
        f_down = set(df_down['farmaco'].dropna())
        ambos = f_up & f_down
        so_up = f_up - f_down
        so_down = f_down - f_up

        print("\n" + "=" * 80)
        print("RESUMO")
        print("=" * 80)
        print(f"Fármacos eixo UP:   {len(f_up)}")
        print(f"Fármacos eixo DOWN: {len(f_down)}")
        print(f"Ambos os eixos:     {len(ambos)}")
        print(f"Só UP:              {len(so_up)}")
        print(f"Só DOWN:            {len(so_down)}")

        # Salvar interseção
        if ambos:
            df_ambos = pd.concat([
                df_up[df_up['farmaco'].isin(ambos)],
                df_down[df_down['farmaco'].isin(ambos)]
            ]).drop_duplicates()
            df_ambos.to_csv(os.path.join(OUTPUT_DIR, "DGIDB_AMBOS.csv"), index=False)
            print(f"\n💾 {os.path.join(OUTPUT_DIR, 'DGIDB_AMBOS.csv')}")

    print("\n" + "=" * 80)
    print(f"✅ CONCLUÍDO -- resultados em {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()