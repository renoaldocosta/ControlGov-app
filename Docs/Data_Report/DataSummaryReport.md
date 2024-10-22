# Data Summary Report

## General Summary of the Data

### Fontes de Dados

1. **Portais de Transparência Municipais:**
   - **Descrição:** Dados públicos sobre despesas e execução orçamentária das câmaras e prefeituras municipais do eixo Bahia|Sergipe.
   - **Formato:** Acessíveis via web scraping.
   - **Frequência de Atualização:** Diária.
   - **Tipo de Dados:** 
      - Registro de Despesas (Pagamentos).
   - **Objetivo de Uso:**
      - Monitorar diariamente os gastos públicos para detectar anomalias ou desvios.
      - Gerar relatórios de transparência financeira para uso interno.
      - Facilitar a análise de gastos por categoria, por unidade gestora, credor e período.
   - **Atualizações:**
      - **Novas Funcionalidades Planejadas:** Requisições aos portais de transparência para monitorar a disponibilidade e saúde dos sites, com visualização em gráficos.

2. **Dados Abertos do Governo:**
   - **Descrição:** Conjunto de dados financeiros e orçamentários disponibilizados por órgãos centrais de governo, como Tribunal de Contas dos Municípios da Bahia.
   - **Formato:** JSON/CSV.
   - **Frequência de Atualização:** Mensal.
   - **Tipo de Dados:** 
      - Registro de Despesas (Empenho, Liquidação e Pagamentos).
   - **Objetivo de Uso:**
      - Monitorar diariamente os gastos públicos para detectar anomalias ou desvios.
      - Gerar relatórios de transparência financeira para uso interno.
      - Facilitar a análise de gastos por categoria, por unidade gestora, credor e período.

3. **Sistemas Internos de Contabilidade:**
   - **Descrição:** Dados detalhados sobre a execução financeira e contábil, incluindo informações sobre credores, pagamentos, e balanços orçamentários.
   - **Formato:** CSV, Excel e PDF.
   - **Frequência de Atualização:** Diário, com atualizações em tempo real para algumas transações.
   - **Tipo de Dados:** 
      - Consolidações dos dados no sistema contábil.
      - Dados de empenho, liquidação e pagamento.
   - **Objetivo de Uso:**
      - Suporte à detecção de anomalias financeiras e auditorias internas.

### Volume de Dados

- Estimativa de 1 a 5 MB de dados financeiros por ano por arquivo, dependendo do tamanho do município, da quantidade de transações financeiras e tecnologia de armazenamento.
- **Atualização:** Não houve aumento significativo no volume de dados devido às novas ferramentas implementadas.

## Data Quality Summary

A qualidade dos dados coletados foi consistente entre as diferentes fontes. Não houve dados faltantes ou inconsistentes, o que garantiu um fluxo contínuo de informações sem a necessidade de etapas adicionais de preenchimento ou limpeza de duplicatas.

### Formatos e Transformação de Dados:
- **Coleta via Web Scraping:** Os dados foram extraídos de portais de transparência municipais utilizando Selenium e armazenados em um banco de dados NoSQL (MongoDB). Esses dados estavam em formato bruto e precisaram ser organizados para análise.
- **Coleta de Arquivos CSV/JSON:** Os dados foram baixados manualmente do site do Tribunal de Contas dos Municípios da Bahia (TCM/BA). Esses arquivos foram transformados e carregados em *dataframes* para serem manipulados de forma mais eficiente.
- **Automação das Transformações:** Utilização de Pydantic com FastAPI para definir o formato dos dados requisitados junto ao Streamlit, garantindo a integridade e consistência dos dados processados.

Essa abordagem dupla permitiu maior flexibilidade na coleta e transformação dos dados, garantindo que eles estivessem em um formato adequado para cada análise, em cada um dos módulos.

---

## Target Variable (Variável Alvo)

As principais variáveis analisadas nos dados financeiros coletados são as que refletem os valores monetários envolvidos nas transações. Cada módulo tem variáveis-alvo específicas:

- **Do módulo PMs & CMs/BA:**
  - **Variável Alvo:** Valor das transações financeiras, representando o total dos empenhos, liquidações e pagamentos realizados pelas prefeituras e câmaras municipais do eixo Bahia.
  - **Objetivo:** Identificar outliers que possam indicar desvios ou anomalias nas transações.
  
- **Do módulo CM Pinhão/SE:**
  - **Variáveis Alvo:** Empenhado, Liquidado e Pago. Essas variáveis representam as diferentes fases da execução orçamentária da Câmara Municipal de Pinhão/SE.
  - **Objetivo:** Detectar anomalias e outliers em cada fase do processo de execução orçamentária, com foco especial em pagamentos que possam indicar objetos de atenção quanto ao controle de despesas.

A análise dessas variáveis tem como objetivo principal detectar **outliers**, ou seja, transações que estejam fora do padrão esperado e que possam necessitar de investigação mais detalhada.

---

## Individual Variables (Variáveis Individuais)

As variáveis independentes (explicativas) utilizadas na análise variam conforme o módulo de dados:

- **Do módulo PMs & CMs/BA:**
  1. **Órgão:** Entidade responsável pela execução do orçamento.
  2. **Unidade Orçamentária:** Unidade administrativa dentro do órgão que realiza a despesa.
  3. **Credor:** Nome da pessoa ou entidade que recebeu o pagamento.
  4. **Código da Unidade Orçamentária:** Identificação da unidade orçamentária.
  5. **Código do Elemento Gestor:** Código que identifica a categoria da despesa.
  6. **Código da Fonte:** Código que indica a origem dos recursos financeiros.

- **Do módulo CM Pinhão/SE:**
  1. **Subelemento:** Detalhamento da despesa.
  2. **Credor:** Nome do credor que recebeu o pagamento.
  3. **Elemento de Despesa:** Categoria da despesa.

Essas variáveis são usadas para segmentar e analisar os dados financeiros em diferentes níveis, permitindo a criação de relatórios detalhados por categoria de despesa, unidade administrativa e credores.

---

## Variable Ranking (Ranking de Variáveis)

Ao analisar os dados de despesas públicas, foi possível identificar as variáveis que mais influenciam o comportamento geral dos gastos. O ranking de importância das variáveis foi baseado na frequência e no impacto dos gastos registrados.

- **Do módulo PMs & CMs/BA:**
  1. **Órgão:** Identifica as entidades que mais contribuem para o volume total de despesas.
  2. **Credor:** Fornecedores com maiores volumes de recebimentos são priorizados na análise de pagamentos.
  
- **Do módulo CM Pinhão/SE:**
  1. **Credor:** A relação entre os maiores credores e os valores recebidos é essencial para identificar fornecedores recorrentes.
  2. **Elemento de Despesa:** A categoria de despesa que mais influencia os valores empenhados, liquidados e pagos.

Atualmente, ainda não foi identificado um padrão específico nos gastos dessas variáveis, mas a análise contínua poderá revelar tendências significativas, como concentrações de gastos em certas categorias ou órgãos.

---

## Conclusão

As informações coletadas e analisadas até agora permitem uma visão ampla sobre a execução financeira das prefeituras e câmaras municipais do eixo Bahia|Sergipe. A identificação de outliers nas variáveis de valor de pagamento e empenhos ajudará a detectar anomalias financeiras, possibilitando a geração de relatórios mais detalhados e assertivos para auditorias e controle de despesas.

**Atualizações Recentes:**
- **Integração de FastAPI:** APIs desenvolvidas para facilitar a consulta e inserção de dados no MongoDB, melhorando a dinamização e simplicidade das aplicações.
- **Automação com Scraper:** Implementação de um scraper com BeautifulSoup para coleta diária de dados, armazenados inicialmente em uma base de stage antes da migração para o banco de dados final.
- **Planejamento de Integrações Futuras:** Consideração da integração com WhatsApp para notificações e uso de Scrapy para monitoramento de disponibilidade de sites, além da inclusão de LLMs para geração de relatórios e análises avançadas.

---

## Próximos Passos


1. **Implementar Medidas de Segurança:**
   - **Adicionar autenticação e autorização** às APIs desenvolvidas com FastAPI para proteger as operações de inserção, atualização e deleção de dados no MongoDB.

2. **Documentação e Treinamento:**
   - **Criar materiais de treinamento** para ensinar os usuários finais a interagir com as novas funcionalidades de LLMs.
   - **Gravar vídeos tutoriais** para auxiliar os usuários no download e upload de dados.

3. **Monitoramento e Feedback:**
   - **Estabelecer mecanismos de coleta de feedback** através de pesquisas ou formulários.
   - **Realizar reuniões mensais** para revisar feedbacks e iterar sobre as funcionalidades existentes.

### Próximos Passos:

1. **Implementar Medidas de Segurança:**
   - Adicionar autenticação às APIs com FastAPI para garantir a segurança dos dados.
   
2. **Desenvolver e Integrar LLMs:**
   - Iniciar a integração de LLMs utilizando Langchain para funcionalidades avançadas como geração de relatórios automáticos e sistemas de Q&A.
   
3. **Treinamento e Capacitação:**
   - Desenvolver e disponibilizar materiais de treinamento para a equipe técnica e usuários finais.

4. **Monitoramento Contínuo:**
   - Estabelecer rotinas de monitoramento para garantir a qualidade dos dados e a performance das novas funcionalidades.