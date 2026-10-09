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

Passo a passo de execução; 
Ordem de execução dos scripts

Os scripts em scripts/ devem ser executados na ordem abaixo. Os nomes de arquivo e as saídas geradas ainda estão sendo organizados — a tabela indica o nome sugerido e, entre parênteses, o nome atual do arquivo, quando diferente.

Ordem	Script

O que faz

1	integracao_datasets.py (Integrador de Datasets)	
Busca os 4 datasets, converte Ensembl → símbolo do gene (via mygene.info) e gera a matriz integrada

2	clustering_nao_supervisionado.py (Analise sem normalizacao)	
Padronização z-score, PCA e K-means (3 critérios de seleção de genes), antes da correção de lote

3	pipeline_consolidado.py (Analise pos normalizacao)	
Normalização log2(CPM+1), correção de lote por ComBat, nova clusterização e análise diferencial por teste t

4	sensibilidade_limiares.py (Threshold)	
Testa diferentes limiares de log2FC sobre o resultado da etapa anterior (análise de sensibilidade)

5	deseq2_principal.py (Analise deseq2 completa)	
Análise diferencial principal via DESeq2 — idoso vs. jovem cruzado (4 datasets, resultado principal) e metformina vs. placebo (GSE157585, discutido nas Limitações). O script também tenta uma terceira comparação (oleuropeína vs. placebo, GSE318937), que não gerou saída e não foi incorporada ao TCC.

6	enrichr_deseq2.py (07_enrich)	
Enriquecimento funcional (Enrichr) sobre os genes diferencialmente expressos pelo DESeq2

7	dgidb_farmacos.py (DGldb)	
Mapeamento gene → fármaco (DGIdb) sobre os genes do DESeq2

Cada script indica, no próprio cabeçalho, os arquivos de entrada que espera encontrar e a pasta de saída. Scripts exploratórios ou de versões anteriores (ex.: análise sobre o limiar de 0,15 pré-DESeq2) foram mantidos no repositório para documentar o percurso metodológico, mas não fazem parte do fluxo principal acima.


Tecnologias Utilizadas
Python 3 (ambiente Anaconda / Spyder)
pandas, numpy — manipulação de dados
scikit-learn — PCA, K-means, padronização (z-score)
scipy, statsmodels — testes estatísticos (teste t, correção de Benjamini-Hochberg)
pyComBat — correção de efeito de lote
pydeseq2 — análise diferencial (DESeq2)
matplotlib, seaborn — visualização
requests — consulta às APIs do Enrichr e do DGIdb
📜 Licença

Trabalho acadêmico (TCC). Direitos reservados ao autor. Para reuso do código, entre em contato: jkazukaa@gmail.com
## 📧 Contato

