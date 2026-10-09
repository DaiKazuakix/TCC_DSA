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

- **E-mail:** [seu-email@exemplo.com]
- **LinkedIn:** [link-do-seu-linkedin]
