# -*- coding: utf-8 -*-
"""
PASSO 01 -- Integração dos datasets de RNA-seq (contagens brutas)
==================================================================
Lê os arquivos de contagem baixados do GEO, converte os identificadores
Ensembl para símbolo de gene (HGNC) e monta uma única matriz com os genes
comuns a todos os datasets.

Etapas:
  1) Lê cada arquivo e extrai os IDs brutos (sem conversão ainda).
  2) Une os IDs de todos os datasets num conjunto único e consulta o
     mygene uma única vez, com cache em disco (mapa_ensembl_symbol_CACHE.csv),
     de modo que novas execuções só consultam a API para IDs ainda não vistos.
  3) Aplica o mapeamento em cada dataset; IDs duplicados após a conversão
     são resolvidos mantendo a linha de maior expressão média.
  4) Concatena os datasets pelos genes comuns.

Entrada:  dados/brutos/  (arquivos listados em SEUS_ARQUIVOS)
Saída:    dados/processados/MATRIZ_INTEGRADA_CONTAGENS.csv
          dados/processados/<dataset>_PROCESSADO.csv
"""
import pandas as pd
import os
import re
from datetime import datetime
import warnings
from mygene import MyGeneInfo

from config import DIR_BRUTOS, DIR_PROCESSADOS, MATRIZ_CONTAGENS

warnings.filterwarnings('ignore')

print("="*80)
print("🧬 PASSO 01 -- INTEGRAÇÃO DE DADOS DE RNA-seq")
print(f"📅 Data de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
INPUT_DIR = DIR_BRUTOS
OUTPUT_DIR = DIR_PROCESSADOS
CACHE_PATH = os.path.join(OUTPUT_DIR, "mapa_ensembl_symbol_CACHE.csv")

SEUS_ARQUIVOS = [
    {'nome': 'GSE157585', 'caminho': os.path.join(INPUT_DIR, 'GSE157585_raw_counts_metadata.txt'),
     'tipo': 'txt', 'gene_id_type': 'ensembl'},
    {'nome': 'GSE305038_Inactive', 'caminho': os.path.join(INPUT_DIR, 'GSE305038_20250714_M008795_edgeRglm_Counts_Inactive_Post-Inactive_Pre.xlsx'),
     'tipo': 'excel', 'sheet_name': 0, 'gene_id_type': 'ensembl'},
    {'nome': 'GSE305038_Normal', 'caminho': os.path.join(INPUT_DIR, 'GSE305038_20250714_M008795_edgeRglm_Counts_Normal_Post-Normal_Pre.xlsx'),
     'tipo': 'excel', 'sheet_name': 0, 'gene_id_type': 'ensembl'},
    {'nome': 'GSE277819', 'caminho': os.path.join(INPUT_DIR, 'GSE277819_raw_counts.txt'),
     'tipo': 'txt', 'gene_id_type': 'ensembl'},
    {'nome': 'GSE318937', 'caminho': os.path.join(INPUT_DIR, 'GSE318937_raw_counts.csv'),
     'tipo': 'csv', 'gene_id_type': 'ensembl'},
]

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================
def detectar_separador(caminho):
    with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
        primeira_linha = f.readline()
    if '\t' in primeira_linha:
        return '\t'
    elif ',' in primeira_linha:
        return ','
    elif ';' in primeira_linha:
        return ';'
    elif '|' in primeira_linha:
        return '|'
    return r'\s+'


def extrair_id_gene(valor):
    if pd.isna(valor):
        return None
    valor = str(valor).strip()
    padroes = [r'ENSG\d{11}', r'ENSG\d{11}\.\d+', r'ENSG\d{11}_\d+', r'^\d+$']
    for padrao in padroes:
        match = re.search(padrao, valor)
        if match:
            return match.group(0)
    return valor


# ============================================================================
# ETAPA 1 -- LER SÓ OS IDs BRUTOS DE CADA ARQUIVO (sem converter ainda)
# ============================================================================
def carregar_dataframe_bruto(arquivo_info):
    """Lê o arquivo e devolve o DataFrame já com Gene_ID_Clean e colunas de
    contagem identificadas -- mas SEM fazer nenhuma conversão de símbolo."""
    nome = arquivo_info['nome']
    caminho = arquivo_info['caminho']
    tipo = arquivo_info['tipo']

    if not os.path.exists(caminho):
        print(f"⚠️ {nome}: arquivo não encontrado, pulando.")
        return None

    print(f"\n📥 Lendo {nome} ({tipo})...")

    if tipo == 'excel':
        df = pd.read_excel(caminho, sheet_name=arquivo_info.get('sheet_name', 0))
        id_col = df.columns[0]
        if not any('ENSG' in str(x) for x in df[id_col].head(10)):
            df = pd.read_excel(caminho, sheet_name=arquivo_info.get('sheet_name', 0), header=1)
            id_col = df.columns[0]
    elif tipo == 'txt':
        sep = detectar_separador(caminho)
        df = pd.read_csv(caminho, sep=sep, engine='python')
        id_col = df.columns[0]
    elif tipo == 'csv':
        # separador explícito de vírgula; cai pra auto-detecção se a
        # primeira linha não tiver vírgula (alguns "csv" na prática são ; )
        with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
            primeira_linha = f.readline()
        sep = ',' if ',' in primeira_linha else detectar_separador(caminho)
        df = pd.read_csv(caminho, sep=sep, engine='python')
        id_col = df.columns[0]
    elif tipo == 'tsv':
        df = pd.read_csv(caminho, sep='\t', engine='python')
        id_col = df.columns[0]
    else:
        print(f"⚠️ Tipo não suportado: {tipo}")
        return None

    df['Gene_ID_Clean'] = df[id_col].apply(extrair_id_gene)
    df = df[df['Gene_ID_Clean'].notna()].copy()

    colunas_contagem = [c for c in df.columns if c not in [id_col, 'Gene_ID_Clean']]
    print(f"   • {df.shape[0]} genes brutos x {len(colunas_contagem)} amostras")

    return {'nome': nome, 'df': df, 'colunas_contagem': colunas_contagem,
            'gene_id_type': arquivo_info.get('gene_id_type', 'ensembl')}


# ============================================================================
# ETAPA 2 -- CONVERSÃO UNIFICADA (uma chamada só, com cache)
# ============================================================================
def carregar_cache():
    if os.path.exists(CACHE_PATH):
        cache_df = pd.read_csv(CACHE_PATH, dtype=str)
        print(f"💾 Cache encontrado: {len(cache_df)} IDs já convertidos anteriormente")
        return dict(zip(cache_df['id'], cache_df['symbol']))
    return {}


def salvar_cache(mapeamento):
    pd.DataFrame({'id': list(mapeamento.keys()), 'symbol': list(mapeamento.values())}) \
        .to_csv(CACHE_PATH, index=False)
    print(f"💾 Cache salvo: {len(mapeamento)} IDs em {CACHE_PATH}")


def converter_ids_unificado(brutos):
    """Recebe a lista de dicts de carregar_dataframe_bruto, monta o conjunto
    único de IDs de TODOS os datasets, e consulta o mygene uma única vez
    (só para os IDs que ainda não estão no cache)."""
    print(f"\n{'='*80}\n🔗 UNIFICANDO IDs DE TODOS OS DATASETS\n{'='*80}")

    todos_ids = set()
    for b in brutos:
        todos_ids.update(b['df']['Gene_ID_Clean'].dropna().unique())
    print(f"🔍 Total de IDs únicos entre todos os datasets: {len(todos_ids):,}")

    mapeamento = carregar_cache()
    ids_faltantes = sorted(todos_ids - set(mapeamento.keys()))
    print(f"🔎 IDs a consultar no mygene (não estão no cache): {len(ids_faltantes):,}")

    if ids_faltantes:
        mg = MyGeneInfo()
        tamanho_lote = 500
        lotes = [ids_faltantes[i:i + tamanho_lote] for i in range(0, len(ids_faltantes), tamanho_lote)]

        for i, lote in enumerate(lotes, 1):
            print(f"   Lote {i}/{len(lotes)} ({len(lote)} IDs)...", end=" ")
            try:
                resultados = mg.querymany(
                    lote,
                    scopes=['ensembl.gene', 'ensembl.transcript', 'entrezgene', 'symbol'],
                    fields=['symbol'],
                    species='human',
                    returnall=False,
                    verbose=False,
                )
                convertidos = 0
                for r in resultados:
                    query = r.get('query', '')
                    symbol = r.get('symbol')
                    if symbol:
                        mapeamento[query] = symbol
                        convertidos += 1
                    elif query not in mapeamento:
                        mapeamento[query] = query  # fallback pontual, só pro que não bateu
                print(f"{convertidos}/{len(lote)} convertidos")
            except Exception as e:
                print(f"⚠️ erro no lote: {str(e)[:100]} -- usando ID como símbolo pra esse lote")
                for gene_id in lote:
                    mapeamento.setdefault(gene_id, gene_id)

        salvar_cache(mapeamento)
    else:
        print("✅ Todos os IDs já estavam no cache -- nenhuma chamada à API necessária.")

    taxa = sum(1 for k, v in mapeamento.items() if k in todos_ids and k != v) / max(len(todos_ids), 1)
    print(f"🎯 Taxa de conversão real (símbolo != ID original): {taxa*100:.1f}%")

    return mapeamento


# ============================================================================
# ETAPA 3 -- APLICAR O MAPEAMENTO E FINALIZAR CADA DATASET (sem API)
# ============================================================================
def finalizar_dataset(bruto, mapeamento):
    nome = bruto['nome']
    df = bruto['df']
    colunas_contagem = bruto['colunas_contagem']

    df = df.copy()
    df['Gene_Symbol'] = df['Gene_ID_Clean'].map(mapeamento)
    df['Gene_Symbol'] = df['Gene_Symbol'].fillna(df['Gene_ID_Clean'])

    df_final = pd.DataFrame({'Gene_Symbol': df['Gene_Symbol']})
    for col in colunas_contagem:
        df_final[col] = df[col].values

    if df_final['Gene_Symbol'].duplicated().any():
        df_final['mean_expr'] = df_final[colunas_contagem].mean(axis=1)
        df_final = df_final.sort_values('mean_expr', ascending=False)
        df_final = df_final.drop_duplicates(subset=['Gene_Symbol'], keep='first')
        df_final = df_final.drop(columns=['mean_expr'])

    print(f"✅ {nome}: {len(df_final)} genes únicos (símbolo) x {len(colunas_contagem)} amostras")

    caminho_saida = os.path.join(OUTPUT_DIR, f"{nome}_PROCESSADO.csv")
    df_final.set_index('Gene_Symbol').to_csv(caminho_saida)

    return df_final.set_index('Gene_Symbol')


# ============================================================================
# ETAPA 4 -- INTEGRAÇÃO
# ============================================================================
def integrar_datasets(datasets_processados):
    print(f"\n{'='*80}\n🔗 INTEGRANDO DATASETS\n{'='*80}")
    genes_por_dataset = {nome: set(df.index) for nome, df in datasets_processados.items()}
    genes_comuns = set.intersection(*genes_por_dataset.values())
    print(f"🔍 Genes comuns a todos os datasets: {len(genes_comuns):,}")

    partes = []
    for nome, df in datasets_processados.items():
        df_filtrado = df.loc[df.index.isin(genes_comuns)]
        df_filtrado = df_filtrado.rename(columns={c: f"{nome}_{c}" for c in df_filtrado.columns})
        partes.append(df_filtrado)

    matriz_integrada = pd.concat(partes, axis=1).sort_index()
    print(f"🎯 MATRIZ INTEGRADA: {matriz_integrada.shape[0]:,} genes x {matriz_integrada.shape[1]:,} amostras")

    matriz_integrada.to_csv(MATRIZ_CONTAGENS)
    print(f"💾 Matriz integrada salva em: {MATRIZ_CONTAGENS}")

    return matriz_integrada


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    # 1) Ler todos os arquivos brutos (sem converter símbolo ainda)
    brutos = [carregar_dataframe_bruto(a) for a in SEUS_ARQUIVOS]
    brutos = [b for b in brutos if b is not None]

    # 2) Converter TODOS os IDs de TODOS os datasets numa passada só
    mapeamento_global = converter_ids_unificado(brutos)

    # 3) Aplicar o mapeamento em cada dataset (nenhuma chamada de API aqui)
    print(f"\n{'='*80}\n📊 FINALIZANDO DATASETS INDIVIDUAIS\n{'='*80}")
    datasets_processados = {b['nome']: finalizar_dataset(b, mapeamento_global) for b in brutos}

    # 4) Integrar
    if len(datasets_processados) > 1:
        matriz_integrada = integrar_datasets(datasets_processados)
        print(f"\n✅ CONCLUÍDO -- {matriz_integrada.shape[0]:,} genes x {matriz_integrada.shape[1]:,} amostras")

    print(f"\n🏁 Fim: {datetime.now().strftime('%H:%M:%S')}")