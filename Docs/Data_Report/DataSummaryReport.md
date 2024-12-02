# Data Summary Report

## 1. Resumo Geral dos Dados

### 1.1 Fontes de Dados

1. **Portais de Transparência Municipais**
   - **Descrição:** Dados públicos sobre despesas e execução orçamentária das câmaras e prefeituras municipais do eixo Bahia|Sergipe.
   - **Formato:** Acessíveis via web scraping.
   - **Frequência de Atualização:** Diária.
   - **Tipo de Dados:** 
     - Registro de Despesas (Empenhos).
   - **Objetivo de Uso:**
     - Monitorar diariamente os gastos públicos para detectar anomalias ou desvios.
     - Gerar relatórios de transparência financeira para uso interno.
     - Facilitar a análise de gastos por categoria, por unidade gestora, credor e período.

2. **Dados Abertos do Governo**
   - **Descrição:** Conjunto de dados financeiros e orçamentários disponibilizados por órgãos centrais de governo, como Tribunal de Contas dos Municípios da Bahia.
   - **Formato:** JSON/CSV.
   - **Frequência de Atualização:** Mensal.
   - **Tipo de Dados:** 
     - Registro de Despesas (Empenho, Liquidação e Pagamentos).
   - **Objetivo de Uso:**
     - Monitorar mensalmente os gastos públicos para detectar anomalias ou desvios.
     - Gerar relatórios de transparência financeira para uso interno.
     - Facilitar a análise de gastos por categoria, por unidade gestora, credor e período.

3. **Sistemas Internos de Contabilidade**
   - **Descrição:** Dados detalhados sobre a execução financeira e contábil, incluindo informações sobre credores, pagamentos e balanços orçamentários.
   - **Formato:** CSV, Excel e PDF.
   - **Frequência de Atualização:** Diário, com atualizações em tempo real para algumas transações.
   - **Tipo de Dados:** 
     - Consolidações dos dados no sistema contábil.
     - Dados de empenho, liquidação e pagamento.
   - **Objetivo de Uso:**
     - Suporte à detecção de anomalias financeiras e auditorias internas.

### 1.2 Volume de Dados

- Estimativa de 1 a 5 MB de dados financeiros por ano por arquivo, dependendo do tamanho do município, da quantidade de transações financeiras e da tecnologia de armazenamento.

## 2. Resumo da Qualidade dos Dados

A qualidade dos dados coletados foi consistente entre as diferentes fontes. Não houve dados faltantes ou inconsistentes, o que garantiu um fluxo contínuo de informações sem a necessidade de etapas adicionais de preenchimento ou limpeza de duplicatas.

### 2.1 Formatos e Transformação de Dados

- **Coleta via Web Scraping:** Os dados foram extraídos de portais de transparência municipais utilizando Selenium e armazenados em um banco de dados NoSQL (MongoDB). Esses dados estavam em formato bruto e precisaram ser organizados para análise.
- **Coleta de Arquivos CSV/JSON:** Os dados foram baixados manualmente do site do Tribunal de Contas dos Municípios da Bahia (TCM/BA). Esses arquivos foram transformados e carregados em *dataframes* para serem manipulados de forma mais eficiente.
- **Automação das Transformações:** Utilização de Pydantic com FastAPI para definir o formato dos dados requisitados junto ao Streamlit, garantindo a integridade e consistência dos dados processados.
- **Integração de APIs e Chatbot com IA:**
  - **APIs Desenvolvidas com FastAPI:** Facilitam a consulta e inserção de dados no MongoDB, permitindo a integração contínua dos dados coletados via web scraping e fontes externas.
  - **Chatbot com IA:** Utiliza Langchain e OpenAI para responder consultas sobre a Câmara Municipal de Pinhão, acessando informações detalhadas através de um banco de dados vetorial de embeddings.

Essa abordagem dupla permitiu maior flexibilidade na coleta e transformação dos dados, garantindo que eles estivessem em um formato adequado para cada análise, em cada um dos módulos.

---

## 3. Integração e Utilização de Dados Coletados via APIs e Web Scraping

### 3.1 Integração de Dados

1. **APIs Desenvolvidas com FastAPI**
   - **Descrição:** APIs criadas com FastAPI para facilitar a consulta e inserção de dados no banco de dados MongoDB.
   - **Função:** Permitem a integração contínua e automática dos dados coletados via web scraping e fontes externas, garantindo que as informações estejam sempre atualizadas e disponíveis para análise.
   - **Utilização no Projeto:**
     - **Coleta Automática de Dados:** O scraper desenvolvido com BeautifulSoup coleta dados diariamente e envia para o banco de dados MongoDB através das APIs.
     - **Consulta de Dados pelo Chatbot:** A API de embeddings baseada em um banco de dados vetorial permite que o chatbot acesse informações detalhadas sobre a Câmara Municipal de Pinhão, incluindo elementos e subelementos de despesa, valores empenhados e consultas a CPF/CNPJ dos credores.

2. **Web Scraping com BeautifulSoup e Selenium**
   - **Descrição:** Técnicas de web scraping utilizadas para extrair dados dos Portais de Transparência Municipais.
   - **Função:** Coletar dados diariamente e armazená-los em uma base de dados intermediária antes de migrá-los para o banco de dados final no MongoDB.
   - **Utilização no Projeto:**
     - **Atualização Diária:** Garantir que os dados financeiros estejam sempre atualizados para análises em tempo real.

### 3.2 Uso de Dados no Projeto

1. **Visualização e Análise de Dados**
   - **Descrição:** Os dados coletados estão sendo utilizados para serem apresentados em registros/tabelas, análise exploratória e em gráficos dinâmicos.
   - **Funcionalidades:**
     - **Registros/Tabelas:** Exibição detalhada das transações financeiras em formatos tabulares para facilitar a consulta e revisão.
     - **Gráficos Dinâmicos:**
       - **Contagem e Valores Totais de Empenhos:** Gráfico dinâmico com ênfase em quantidade e valores relativos aos empenhos.
       - **Elementos e Subelementos de Despesa Empenhados:** Gráfico dinâmico com ênfase em quantidade e valores relativos aos elementos e subelementos de despesa empenhados.

2. **IA Geradora de Relatórios (Padrão e Personalizado)**
   - **Descrição:** Implementação de uma IA capaz de gerar análises ou relatórios com base nos dados refletidos nos gráficos.
   - **Funcionalidades:**
     - **Análise ou Relatório Padrão:** Geração automática de análises padrão dos dados.
     - **Análise Personalizada:** Permite que o usuário interaja e solicite análises específicas conforme a necessidade.
   - **Objetivo:** Facilitar a interpretação dos dados e fornecer insights acionáveis de forma automatizada.

3. **Chatbot com IA para Consultas de Dados**
   - **Descrição:** Implementação de um chatbot utilizando Langchain e OpenAI para responder consultas sobre a Câmara Municipal de Pinhão.
   - **Funcionalidades:**
     - **Consultas Diretas no Chat:** Permite aos usuários perguntar sobre elementos e subelementos de despesa, valores empenhados a pessoas físicas e jurídicas, e consultas a CPF/CNPJ dos credores.
     - **Conhecimento Avançado:** O chatbot possui conhecimento detalhado sobre CPFs e CNPJs de credores, valores empenhados para cada credor, e valores empenhados por elemento ou subelemento.
     - **Armazenamento de Memória:** A memória do chatbot é armazenada na sessão, garantindo que as interações sejam contextualmente relevantes.
     - **Notificações e Respostas:** As respostas do chatbot são exibidas em espaços reservados e notificações na página correspondente à Câmara de Pinhão.

4. **Banco de Dados Vetorial para Embeddings**
   - **Descrição:** Utilização de um banco de dados vetorial para armazenar embeddings gerados a partir dos dados financeiros.
   - **Função:** Facilitar a busca semântica e a recuperação eficiente de informações relevantes para o chatbot.
   - **Ferramentas Utilizadas:** Langchain e OpenAI para gerar e gerenciar embeddings.

### 3.3 Fluxo de Integração de Dados

1. **Coleta de Dados:**
   - Web scraping diário dos Portais de Transparência Municipais via Selenium e BeautifulSoup.
   - Importação mensal de dados financeiros em formato JSON/CSV do Tribunal de Contas dos Municípios da Bahia.
   - Extração diária de dados dos sistemas internos de contabilidade.

2. **Processamento e Armazenamento:**
   - Transformação e limpeza dos dados utilizando pandas.
   - Armazenamento dos dados limpos no banco de dados MongoDB através das APIs desenvolvidas com FastAPI.

3. **Análise e Visualização:**
   - Utilização de Streamlit para construir dashboards interativos.
   - Integração do chatbot para consultas avançadas, notificações e geração de relatórios.
   - Geração de gráficos dinâmicos para facilitar a visualização e análise dos dados.

---

## 4. Variável Alvo (Target Variable)

As principais variáveis analisadas nos dados financeiros coletados são as que refletem os valores monetários envolvidos nas transações. Cada módulo tem variáveis-alvo específicas:

- **Do módulo PMs & CMs/BA:**
  - **Variável Alvo:** Valor das transações financeiras, representando o total dos empenhos, liquidações e pagamentos realizados pelas prefeituras e câmaras municipais do eixo Bahia.
  - **Objetivo:** Identificar outliers que possam indicar desvios ou anomalias nas transações.

- **Do módulo CM Pinhão/SE:**
  - **Variáveis Alvo:** Empenhado, Liquidado e Pago. Essas variáveis representam as diferentes fases da execução orçamentária da Câmara Municipal de Pinhão/SE.
  - **Objetivo:** Detectar anomalias e outliers em cada fase do processo de execução orçamentária, com foco especial em pagamentos que possam indicar objetos de atenção quanto ao controle de despesas.

A análise dessas variáveis tem como objetivo principal detectar **outliers**, ou seja, transações que estejam fora do padrão esperado e que possam necessitar de investigação mais detalhada.

---

## 5. Variáveis Individuais (Individual Variables)

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

## 6. Ranking de Variáveis (Variable Ranking)

Ao analisar os dados de despesas públicas, foi possível identificar as variáveis que mais influenciam o comportamento geral dos gastos. O ranking de importância das variáveis foi baseado na frequência e no impacto dos gastos registrados.

- **Do módulo PMs & CMs/BA:**
  1. **Órgão:** Identifica as entidades que mais contribuem para o volume total de despesas.
  2. **Credor:** Fornecedores com maiores volumes de recebimentos são priorizados na análise de pagamentos.

- **Do módulo CM Pinhão/SE:**
  1. **Credor:** A relação entre os maiores credores e os valores recebidos é essencial para identificar fornecedores recorrentes.
  2. **Elemento de Despesa:** A categoria de despesa que mais influencia os valores empenhados, liquidados e pagos.

Atualmente, ainda não foi identificado um padrão específico nos gastos dessas variáveis, mas a análise contínua poderá revelar tendências significativas, como concentrações de gastos em certas categorias ou órgãos.

---

## 7. Conclusão

As informações coletadas e analisadas até agora permitem uma visão ampla sobre a execução financeira das prefeituras e câmaras municipais do eixo Bahia|Sergipe. A identificação de outliers nas variáveis de valor de pagamento e empenhos ajudará a detectar anomalias financeiras, possibilitando a geração de relatórios mais detalhados e assertivos para auditorias e controle de despesas.

**Atualizações Recentes:**
- **Integração de FastAPI:** APIs desenvolvidas para facilitar a consulta e inserção de dados no MongoDB, melhorando a dinamização e simplicidade das aplicações.
- **Automação com Scraper:** Implementação de um scraper com BeautifulSoup para coleta diária de dados, armazenados inicialmente em uma base de stage antes da migração para o banco de dados final.
- **Planejamento de Integrações Futuras:** Consideração da integração com WhatsApp para notificações e uso de Scrapy para monitoramento de disponibilidade de sites, além da inclusão de LLMs para geração de relatórios e análises avançadas.
- **Chatbot com IA:** Criação de um chatbot utilizando Langchain e OpenAI para responder consultas sobre a Câmara Municipal de Pinhão, facilitando o acesso a informações detalhadas via chat ou dashboard.
- **Gráficos Dinâmicos:** Implementação de gráficos dinâmicos que enfatizam a quantidade e os valores relativos aos empenhos, bem como aos elementos e subelementos de despesa empenhados.
- **IA Geradora de Relatórios:** Desenvolvimento de uma IA capaz de gerar análises e relatórios padrão ou personalizados com base nos dados refletidos nos gráficos, permitindo interações específicas conforme a necessidade do usuário.
- **Conhecimento Avançado do Chatbot:** Ampliação das capacidades do chatbot para incluir informações detalhadas sobre CPFs e CNPJs de credores, valores empenhados para cada credor, e valores empenhados por elemento ou subelemento.

Essas atualizações fortalecem a capacidade de monitoramento, análise e transparência dos dados financeiros, proporcionando ferramentas mais robustas para a detecção de anomalias e a geração de insights valiosos para a gestão pública.
