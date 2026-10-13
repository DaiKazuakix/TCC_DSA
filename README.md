Aprendizado de máquina aplicado ao transcriptoma de exercício físico e avaliação de fármacos adjuvantes

Orientador: Renato Máximo Instituição: USP Esalq Curso: Data Science & Analytics Ano: 2026

Este trabalho contou com a integração de quatro matrizes de dados de RNA-seq em humanos obtidos do músculo esquelético em jovens e idosos para avaliação do efeito do exercício físico em cada grupo. O objetivo do trabalho foi identificar genes diferencialmente expressos e relacioná-los com ações de fármacos já conhecidos na literatura, para que possam ser tratados como alvo terapêutico de indução ao mimetismo do exercício e/ou como adjuvantes à atividade física.

Estrutura do Repositório

Metadados brutos/: metadados de cada GSE, fornecidos pelos pesquisadores originais.
metadados ajustados/: metadados julgados pertinentes para o trabalho.
scripts/: códigos para reprodutibilidade da análise.
Dataset	Link de acesso	Arquivo utilizado
GSE157585	https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157585	contagens brutas (RNA-seq)
GSE277819	https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277819	contagens brutas (RNA-seq)
GSE305038	https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305038	..._edgeRglm_Counts_Inactive_Post-Inactive_Pre.xlsx e ..._edgeRglm_Counts_Normal_Post-Normal_Pre.xlsx
GSE318937	https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE318937	contagens brutas (RNA-seq)

Passo a passo, para cada dataset:

Acesse o link de acesso da tabela acima.
Role a página até a seção "Supplementary file", no final.
Baixe o(s) arquivo(s) de contagem indicado(s) na coluna "Arquivo utilizado" (clique em (http) ou (ftp) ao lado do nome do arquivo).
Salve o arquivo baixado na pasta scripts/ (ou na pasta de onde o script correspondente for executado) — cada script indica, no cabeçalho, o nome de arquivo que espera encontrar.

TCC_DSA/
├── README.md
├── requirements.txt
├── metadados/
│   └── METADADOS_ENRIQUECIDOS_v2.csv   anotação das amostras (montada manualmente a partir do GEO)
├── scripts/
│   ├── config.py                       caminhos e listas de genes compartilhados
│   ├── 01_ ... 07_*.py                 fluxo principal
│   └── exploratorios/                  análises complementares
├── dados/                              criada na execução (não versionada)
│   ├── brutos/                         contagens baixadas do GEO
│   └── processados/                    matrizes integradas, log2CPM e checkpoint
└── resultados/                         criada na execução
    ├── *.csv                           expressão diferencial, Enrichr e DGIdb
    └── figuras/                        gráficos de cada etapa
requests — consulta às APIs do Enrichr e do DGIdb
📜 Licença

Trabalho acadêmico (TCC). Direitos reservados ao autor. Para reuso do código, entre em contato: jkazukaa@gmail.com

Ordem de execução dos scripts
Ordem	Script	O que faz
1	01_integracao_datasets.py	Lê os 4 datasets, converte Ensembl → símbolo do gene (via mygene.info, com cache local) e gera a matriz integrada de contagens com os genes comuns
2	02_clustering_nao_supervisionado.py	Normalização log2(CPM+1), padronização z-score, PCA e K-means (3 critérios de seleção de genes) antes da correção de lote; calcula o índice de separação por dataset
3	03_pipeline_consolidado.py	Correção de lote por ComBat, nova clusterização (PCA, K-means, UMAP, ARI por categoria funcional) e análise diferencial por teste t
4	04_sensibilidade_limiares.py	Testa diferentes limiares de log2FC sobre o resultado da etapa anterior (análise de sensibilidade)
5	05_deseq2_principal.py	Análise diferencial principal via DESeq2: idoso vs. jovem cruzando os 4 datasets, com o dataset como covariável (resultado principal), e metformina vs. placebo (GSE157585, discutido nas Limitações). Gera também o volcano plot da Figura 10
6	06_enrichr_deseq2.py	Enriquecimento funcional (Enrichr) sobre os genes diferencialmente expressos pelo DESeq2
7	07_dgidb_farmacos.py	Mapeamento gene → fármaco (DGIdb) sobre os genes do DESeq2

No 05_deseq2_principal.py, a opção RODAR_DESEQ2 = False refaz apenas o volcano a partir dos CSVs já salvos. A terceira comparação do script (oleuropeína vs. placebo, GSE318937) não gerou saída e não foi incorporada ao TCC; ela fica desligada por padrão (RODAR_OLEUROPEINA = False).

Cada script indica, no próprio cabeçalho, os arquivos de entrada que espera encontrar e onde salva as saídas.

Scripts exploratórios (scripts/exploratorios/)

Mantidos para documentar o percurso metodológico; não fazem parte do fluxo principal.

Script	O que faz
limiar_015_enrichr_dgidb.py	Volcano, Enrichr e DGIdb no limiar exploratório (abs(log2FC) ≥ 0,15) sobre os dados corrigidos por ComBat (Figura 9). Requer a etapa 4
validacao_deseq2_GSE157585.py	DESeq2 idoso vs. jovem restrito ao GSE157585 e concordância com o teste t da etapa 3
Critérios estatísticos

Genes diferencialmente expressos: abs(log2FC) ≥ 0,5 e p-ajustado (Benjamini-Hochberg) ≤ 0,01. Termos do Enrichr: p-ajustado ≤ 0,05 e sobreposição mínima de 5 genes.

Tecnologias Utilizadas
Python 3 (ambiente Anaconda / Spyder)
pandas, numpy — manipulação de dados
scikit-learn — PCA, K-means, padronização (z-score)
scipy, statsmodels — testes estatísticos (teste t, correção de Benjamini-Hochberg)
pyComBat — correção de efeito de lote
pydeseq2 — análise diferencial (DESeq2)
umap-learn — projeção UMAP
mygene — conversão de identificadores de genes
matplotlib — visualização
requests — consulta às APIs do Enrichr e do DGIdb

As dependências podem ser instaladas com pip install -r requirements.txt. O pacote combat (pyComBat) pode exigir uma versão específica do numpy; se houver conflito, rode a etapa 3 em um ambiente separado.

📜 Licença

Trabalho acadêmico (TCC). Direitos reservados ao autor. Para reuso do código, entre em contato: jkazukaa@gmail.com
## 📧 Contato

