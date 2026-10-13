# -*- coding: utf-8 -*-
"""
Configuração compartilhada por todos os scripts do pipeline.

Fica na pasta scripts/. Por padrão, as pastas de dados e resultados são
criadas na raiz do repositório (um nível acima de scripts/). Para usar outra pasta (por exemplo, onde os dados já estão
no seu computador), defina a variável de ambiente TCC_BASE_DIR antes de
rodar os scripts:

    Windows (Anaconda Prompt):  set TCC_BASE_DIR=C:\\caminho\\para\\pasta
    Linux / macOS:              export TCC_BASE_DIR=/caminho/para/pasta

Estrutura esperada dentro de TCC_BASE_DIR:

    dados/brutos/        arquivos de contagem baixados do GEO
    dados/processados/   matrizes integradas, log2CPM e checkpoint
    resultados/          tabelas de expressão diferencial, Enrichr e DGIdb
    resultados/figuras/  gráficos gerados pelos scripts

O arquivo de metadados fica em metadados/, na raiz do repositório.
"""
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
BASE_DIR = os.environ.get("TCC_BASE_DIR", REPO_DIR)

DIR_BRUTOS = os.path.join(BASE_DIR, "dados", "brutos")
DIR_PROCESSADOS = os.path.join(BASE_DIR, "dados", "processados")
DIR_RESULTADOS = os.path.join(BASE_DIR, "resultados")
DIR_FIGURAS = os.path.join(DIR_RESULTADOS, "figuras")

for _pasta in (DIR_BRUTOS, DIR_PROCESSADOS, DIR_RESULTADOS, DIR_FIGURAS):
    os.makedirs(_pasta, exist_ok=True)

# Arquivos centrais do pipeline
MATRIZ_CONTAGENS = os.path.join(DIR_PROCESSADOS, "MATRIZ_INTEGRADA_CONTAGENS.csv")
MATRIZ_LOG2CPM = os.path.join(DIR_PROCESSADOS, "MATRIZ_LOG2CPM.csv")
# Metadados curados manualmente -- versionados junto com o código, sempre
# lidos da pasta do repositório (mesmo que TCC_BASE_DIR aponte para outro lugar)
METADADOS = os.path.join(REPO_DIR, "metadados", "METADADOS_ENRIQUECIDOS_v2.csv")
CHECKPOINT_ETAPA_B = os.path.join(DIR_PROCESSADOS, "_checkpoint_pos_etapaB.pkl")
DESEQ2_CRUZADO = os.path.join(DIR_RESULTADOS, "DESeq2_CRUZADO_idoso_vs_jovem.csv")
DESEQ2_METFORMINA = os.path.join(DIR_RESULTADOS, "DESeq2_metformina_vs_placebo_157585.csv")

# Genes por categoria funcional (clusterização por categoria e anotação dos DEGs)
GENES_CATEGORIAS = {
    'musculares': [
        'BDNF', 'FGF2', 'FGF21', 'IGF1', 'IGF2', 'IGFBP3', 'IGFBP5',
        'ACTA1', 'ACTA2', 'ACTC1', 'DES', 'MYH1', 'MYH2', 'MYH3', 'MYH4', 'MYH6', 'MYH7',
        'MYH8', 'MYL1', 'MYL2', 'MYL3', 'MYLPF', 'TNNC1', 'TNNC2', 'TNNI1', 'TNNI2', 'TNNT1',
        'TNNT2', 'TNNT3', 'TPM1', 'TPM2', 'TPM3', 'TPM4',
        'MSTN', 'MYOD1', 'MYOG', 'PAX3', 'PAX7', 'MYF5', 'MYF6',
        'FOXO1', 'FOXO3', 'MURF1', 'FBXO32', 'TRIM63', 'MTOR', 'AKT1', 'AKT2',
        'PPARGC1A', 'PPARGC1B', 'ESRRA', 'ESRRB', 'ESRRG', 'NR4A1', 'NR4A3',
        'GDF8', 'GDF11', 'FST', 'FSTL1', 'FSTL3'
    ],
    'oxigenacao': [
        'EPO', 'EPOR', 'HIF1A', 'HIF3A', 'VEGFA', 'VEGFB', 'VEGFC',
        'ANGPT1', 'ANGPT2', 'ANGPTL4', 'TEK', 'TIE1',
        'NOS1', 'NOS2', 'NOS3', 'EDN1', 'EDNRA', 'EDNRB',
        'HBA1', 'HBA2', 'HBB', 'HBD', 'HBE1', 'HBG1', 'HBG2', 'HBZ', 'MB', 'CYGB',
        'CA9', 'CA12', 'LDHA', 'PDK1', 'SLC2A1', 'SLC2A3', 'BNIP3', 'BNIP3L',
        'MMP2', 'MMP9', 'TIMP1', 'TIMP2', 'TIMP3', 'TIMP4', 'COL18A1', 'THBS1', 'THBS2'
    ],
    'neuromuscular': [
        'CHRNA1', 'CHRNB1', 'CHRND', 'CHRNE', 'CHRNG', 'CHAT', 'ACHE', 'BCHE',
        'MUSK', 'LRP4', 'DOK7', 'AGRN', 'RAPSN', 'LAMA4', 'LAMA5', 'LAMB2',
        'NTRK1', 'NTRK2', 'NTRK3', 'NGFR', 'BDNF', 'NTF3', 'NTF4',
        'SYT1', 'SYT2', 'SNAP25', 'STX1A', 'VAMP1', 'VAMP2', 'SYN1', 'SYN2', 'SYN3',
        'SCN4A', 'SCN4B', 'CACNA1S', 'CACNB1', 'CACNG1', 'RYR1', 'RYR3', 'ATP2A1',
        'GDNF', 'NGF', 'NTN1', 'NTN4', 'ARTN', 'PSPN', 'CNTF', 'LIF', 'CTF1'
    ],
    'metabolicos': [
        'CPT1A', 'CPT1B', 'CPT2', 'ACADM', 'ACADVL', 'ACADL', 'ACADS',
        'PDK4', 'LDHA', 'LDHB', 'LDHC',
        'SLC2A1', 'SLC2A2', 'SLC2A3', 'SLC2A4', 'SLC2A5',
        'FABP1', 'FABP2', 'FABP3', 'FABP4', 'FABP5', 'FABP7',
        'LPL', 'LIPE', 'PNPLA2',
        'INSR', 'IRS1', 'IRS2', 'PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2'
    ],
    'inflamacao': [
        'IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12A', 'IL12B', 'IL15',
        'TNF', 'TNFSF10', 'TNFSF11', 'TNFSF12', 'TNFSF13',
        'CCL2', 'CCL3', 'CCL4', 'CCL5', 'CCL7', 'CCL8', 'CCL11', 'CCL13',
        'CXCL1', 'CXCL2', 'CXCL3', 'CXCL5', 'CXCL9', 'CXCL10',
        'CCR1', 'CCR2', 'CCR3', 'CCR4', 'CCR5', 'CCR6', 'CCR7', 'CCR8',
        'CXCR1', 'CXCR2', 'CXCR3', 'CXCR4', 'CXCR5', 'CXCR6',
        'CRP', 'SAA1', 'SAA2', 'SAA4', 'HP', 'ORM1', 'ORM2',
        'NFKB1', 'NFKB2', 'RELA', 'RELB', 'STAT1', 'STAT3', 'STAT5A', 'STAT5B'
    ],
    'senescencia': [
        'CDKN1A', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'CDKN2D', 'TP53', 'RB1',
        'SIRT1', 'SIRT2', 'SIRT3', 'SIRT4', 'SIRT5', 'SIRT6', 'SIRT7',
        'TERC', 'TERT', 'DKC1', 'NOP10', 'NHP2', 'POT1', 'TINF2',
        'SOD1', 'SOD2', 'SOD3', 'CAT', 'GPX1', 'GPX2', 'GPX3', 'GPX4', 'GSTP1',
        'GSTM1', 'GSTT1', 'PRDX1', 'PRDX2', 'PRDX3', 'PRDX4', 'PRDX5', 'PRDX6',
        'ATG3', 'ATG5', 'ATG7', 'ATG10', 'ATG12', 'ATG16L1', 'BECN1',
        'SQSTM1', 'LAMP1', 'LAMP2', 'ULK1', 'ULK2', 'PINK1'
    ]
}
