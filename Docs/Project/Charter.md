# Project Charter

### Business Background

* **Cliente:** Câmaras e Prefeituras Municipais do eixo Bahia|Sergipe.
* **Domínio de Negócio:** Gestão Pública, Controle Interno, Finanças Públicas.
* **Problemas de Negócio:** 
  * Dificuldade na gestão e monitoramento eficaz das finanças públicas devido à complexidade dos sistemas de contabilidade existentes.
  * Falta de interfaces intuitivas e painéis de controle que facilitem a visualização clara dos gastos públicos.
  * Incapacidade de identificar padrões suspeitos ou anomalias nas despesas financeiras, comprometendo a transparência e a segurança.
  * Necessidade de previsões orçamentárias precisas para evitar déficits ou superávits desnecessários.

### Scope (Escopo)

* **Soluções de Ciência de Dados a Serem Desenvolvidas:**
  * Dashboard interativo para visualização referente à despesa pública.
  * **Gráficos Dinâmicos:**
    * **Gráfico de Contagem e Valores Totais de Empenhos:** Foco na contagem e nos valores totais dos empenhos.
    * **Gráfico de Subelementos e Elementos de Despesa:** Foco na quantidade e valores relativos aos subelementos e elementos de despesa empenhados.
  * IA geradora de relatórios (Padrão e Personalizado): Capacidade de gerar análises ou relatórios baseados nos dados refletidos no Gráfico de Subelementos e Elementos de Despesa, permitindo tanto análises padrão quanto solicitações de análises específicas pelo usuário.
  * APIs desenvolvidas com FastAPI para consultar e inserir dados no banco MongoDB.
  * Integração do scraper com BeautifulSoup para coleta diária de dados.
  * Implementação de funcionalidades de LLMs utilizando Langchain.
  * Potencial integração com WhatsApp para notificações e uso de Scrapy para monitoramento de disponibilidade de sites.
  * **Chatbot com IA:** Implementação de um chatbot utilizando Langchain e OpenAI com conhecimento sobre CPFs e CNPJs de credores, valores empenhados para cada credor, e valores empenhados por elemento ou subelemento, facilitando consultas e interações com os dados financeiros.

* **O que faremos:**
  * Coletar dados financeiros dos Portais de Transparência, central de dados do governo e sistemas internos de contabilidade.
  * Realizar análise exploratória de dados (EDA) para entender a estrutura e qualidade dos dados.
  * Criar uma aplicação web utilizando Streamlit para disponibilizar os resultados para os Controles Internos.
  * Desenvolver e implementar APIs com FastAPI para facilitar a consulta e inserção de dados.
  * Configurar um scraper com BeautifulSoup para coleta automática diária de dados e migração para o banco de dados final.
  * Desenvolver e integrar Chatbot com IA: Utilizar Langchain e OpenAI para criar um chatbot que permita consultas avançadas e notificações, melhorando a interatividade e a acessibilidade às informações financeiras.
  * **Implementar gráficos dinâmicos para visualização detalhada das despesas:**
    * **Gráfico de Contagem e Valores Totais de Empenhos:** Visualização da quantidade e valores totais dos empenhos realizados.
    * **Gráfico de Subelementos e Elementos de Despesa:** Detalhamento das quantidades e valores relativos aos subelementos e elementos de despesa empenhados.
  * Desenvolver uma IA geradora de relatórios que permita a criação de análises padrão e personalizadas com base nos dados disponíveis.

* **Consumo pelo Cliente:**
  * A solução será acessível via interface web existente, permitindo que os Controles Internos das câmaras e prefeituras municipais utilizem a aplicação para monitorar e analisar as finanças públicas de seus municípios de forma eficiente e intuitiva.
  * As novas funcionalidades de relatórios e LLMs serão acessíveis através de interfaces adicionais alimentadas pelas APIs desenvolvidas.
  * **Interação via Chatbot:** Os usuários poderão interagir com o chatbot diretamente no dashboard ou através de notificações, facilitando o acesso rápido a informações específicas e detalhadas.
  * **Gráficos Dinâmicos e Relatórios:** Usuários poderão visualizar gráficos interativos e gerar relatórios detalhados diretamente pela interface web.

### Personnel

* **ControlGov:**
  - **Administrador de Dados:** Renoaldo Costa
* **Client:**
  - **Testador:** Gidelma dos Santos Bomfim (Pinhão/SE)
  - **Contato de Negócios de Prefeitura:** Raimunda Alvez (Pojuca/BA)
  - **Contato de Negócios de Câmara:** Sérgio Benedito (Nilo Peçanha/BA)
  - **Contato de Negócios de Câmara:** Dárcio Piatã (Piatã/BA)

### Metrics (Indicadores e Metas)

* **Objetivos Qualitativos:**
  * Melhorar a eficiência do monitoramento financeiro e a transparência nos gastos públicos.
  * Facilitar a identificação de fraudes, erros e má gestão dos recursos públicos.

* **Métricas Quantitativas:**
  * Reduzir o tempo médio necessário para análises financeiras em pelo menos 50%.
  * Atingir uma precisão de 95% na detecção de anomalias e padrões suspeitos.
  * Alcançar uma satisfação do usuário de 85% em termos de facilidade de uso da aplicação.
  * **Novos Indicadores de Desempenho:**
    * Acurácia das LLMs em 90%.
    * Consultas da API em 100%.
    * **Desempenho do Chatbot:**
      * Tempo de resposta do chatbot inferior a 2 segundos.
      * Taxa de satisfação das interações com o chatbot acima de 80%.
    * **Indicadores Adicionais:**
      * Utilização dos gráficos dinâmicos por 70% dos usuários.
      * Geração de relatórios personalizados por 60% dos usuários.
    * **Desempenho da IA Geradora de Relatórios:**
      * Precisão das análises geradas em 90%.
      * Tempo de geração de relatórios inferior a 5 segundos.

* **Baseline e Metodologia de Medição:**
  * **Baseline:** Tempo atual de análise financeira, precisão na detecção de anomalias e previsões orçamentárias serão registrados antes da implementação.
  * **Medidas:** Acompanhamento contínuo através de feedback dos usuários, análise de logs da aplicação e comparações periódicas com os valores de baseline.
  * **Avaliação do Chatbot:** Utilização de métricas como tempo de resposta, precisão das respostas e satisfação do usuário para medir a eficácia do chatbot.
  * **Avaliação dos Gráficos Dinâmicos e IA de Relatórios:** Monitoramento do uso e feedback dos usuários para aprimoramento contínuo.

### Plan (Plano de Projeto)

1. **Planejamento e Definição de Escopo:** Identificar stakeholders, definir objetivos detalhados, criar o Project Charter. *(15 dias)*
2. **Aquisição e Entendimento dos Dados:** Coleta de dados, EDA, validação de dados. *(15 dias)*
3. **Desenvolvimento do Dashboard:** Implementação da interface web, integração dos modelos de ML. *(1 mês)*
4. **Desenvolvimento e Integração do Chatbot com IA:**
   - **Desenvolvimento do Chatbot:** Utilizar Langchain e OpenAI para criar o chatbot. *(2 semanas)*
   - **Integração com Dashboard:** Integrar o chatbot ao dashboard existente, garantindo que as consultas possam ser feitas diretamente pela interface. *(2 semanas)*
5. **Desenvolvimento de Gráficos Dinâmicos:**
   - **Gráfico de Contagem e Valores Totais de Empenhos e Gráfico de Subelementos e Elementos de Despesa:** Implementação de gráficos dinâmicos para visualização detalhada das despesas. *(3 semanas)*
6. **Desenvolvimento da IA Geradora de Relatórios:**
   - **Relatórios Padrão e Personalizados:** Implementação da IA para geração de análises e relatórios com base nos dados dos Gráficos Dinâmicos. *(3 semanas)*
7. **Testes e Validação:** Testes com usuários finais, ajuste de funcionalidades, melhoria contínua. *(1 mês)*
8. **Implementação em Produção:** Lançamento da aplicação, treinamento de usuários, documentação. *(15 dias)*

* **Novas Etapas e Alterações no Cronograma:**
  * **Integração de FastAPI e Scraper:** Desenvolver e implementar APIs com FastAPI e configurar o scraper com BeautifulSoup para coleta diária de dados. *(Paralelamente às etapas existentes)*
  * **Medidas de Segurança da API:** Implementar autenticação e autorização para garantir a segurança dos dados na API. *(Paralelamente às etapas existentes)*
  * **Desenvolvimento e Integração do Chatbot com IA:** Adicionar duas semanas para o desenvolvimento e duas semanas para a integração do chatbot com a aplicação existente.
  * **Implementação de Gráficos Dinâmicos e IA de Relatórios:** Adicionar etapas específicas para o desenvolvimento e integração dessas funcionalidades.

### Architecture (Arquitetura)

* **Dados:**
  * **Fontes de Dados:** Dados financeiros dos Portais de Transparência, dados abertos do governo e sistemas internos de contabilidade.
  * **Movimentação de Dados:** Utilização de APIs com FastAPI, webscraping com BeautifulSoup, RPA e exportação de arquivos CSV e JSON para coleta de dados.
  * **Armazenamento e Processamento:** Utilização de ferramentas como Selenium para webscraping e RPA, pandas para processamento e transformação de dados, scikit-learn e TensorFlow para modelagem, Pydantic para modelagem de dados no FastAPI.
  * **Ferramentas de Análise:** Dashboards interativos construídos com Streamlit, hospedados em servidor seguro para acesso dos usuários.
  * **Banco de Dados Vetorial:** Implementação de um banco de dados vetorial para armazenar embeddings, facilitando a busca semântica e a interação com o chatbot.

* **Uso de Resultados:**
  * A solução permitirá que os Controles Internos das câmaras e prefeituras municipais visualizem os resultados em tempo real, facilitando a tomada de decisões informadas e baseadas em dados.
  * A automação de tarefas e a detecção de anomalias serão incorporadas ao fluxo de trabalho diário dos usuários.
  * **Novas Integrações:**
    * APIs com FastAPI alimentando dados para dashboards e interfaces de relatórios.
    * Scripts de scraping automatizados alimentando dados no banco de dados MongoDB.
    * **Chatbot com IA:** Integração do chatbot com Langchain e OpenAI para permitir consultas avançadas e notificações automatizadas.
    * **Banco de Dados Vetorial:** Utilização para armazenar e recuperar embeddings, melhorando a eficiência das respostas do chatbot.
    * **Gráficos Dinâmicos:** Implementação dos Gráficos de Contagem e Valores Totais de Empenhos e Subelementos e Elementos de Despesa para visualização detalhada das despesas.
    * **IA Geradora de Relatórios:** Integração da IA para geração de relatórios padrão e personalizados com base nos dados dos gráficos dinâmicos.

### Communication (Comunicação)

* **Reuniões Mensais:** Reuniões semanais de acompanhamento com a equipe de projeto para monitorar o progresso e ajustar o cronograma conforme necessário.
* **Relatórios de Progresso Mensais:** Relatórios mensais para stakeholders para manter todos informados sobre o progresso do projeto.
* **Contato de Comunicação:**
  - **Equipe Técnica:** WhatsApp
  - **Cliente:** WhatsApp

---

### Alinhamento dos Dados, Uso de IA e Engenharia de Prompts com a Resolução do Problema de Negócio

**Relevância no Contexto de Ciência de Dados:**

1. **Integração de Dados:**
   - A coleta de dados via APIs e web scraping garante a disponibilidade e atualização contínua das informações financeiras, fundamentais para análises precisas e em tempo real.
   - A utilização de um banco de dados vetorial para armazenar embeddings permite uma busca eficiente e recuperação de informações relevantes, otimizando o processo de análise e resposta a consultas.

2. **Uso de Inteligência Artificial:**
   - **Chatbot com IA:** Facilita o acesso às informações financeiras, permitindo que os usuários façam consultas rápidas e recebam respostas precisas, promovendo uma maior interação e compreensão dos dados.
   - **LLMs e Engenharia de Prompts:** A utilização de Langchain e OpenAI para criar o chatbot demonstra a aplicação avançada de modelos de linguagem para resolver problemas de acessibilidade e interatividade com grandes volumes de dados.
   - **IA Geradora de Relatórios:** Automatiza a criação de análises e relatórios, aumentando a eficiência e permitindo que os usuários obtenham insights detalhados rapidamente.

3. **Engenharia de Prompts:**
   - A definição de prompts eficazes para o chatbot garante que as consultas sejam interpretadas corretamente e que as respostas sejam relevantes e úteis, melhorando a experiência do usuário e a eficiência na obtenção de informações.

4. **Resolução do Problema de Negócio:**
   - **Transparência e Monitoramento:** A integração de dados e o uso de IA permitem um monitoramento contínuo e transparente das finanças públicas, facilitando a identificação de fraudes e desvios.
   - **Eficiência e Automação:** A automação das tarefas de coleta e análise de dados reduz significativamente o tempo necessário para realizar análises financeiras, aumentando a eficiência operacional.
   - **Tomada de Decisões Informadas:** As ferramentas desenvolvidas fornecem insights detalhados e precisos, apoiando a tomada de decisões baseadas em dados e contribuindo para uma gestão pública mais eficaz e responsável.
   - **Análises Detalhadas e Personalizadas:** A capacidade de gerar relatórios personalizados permite que os usuários obtenham informações específicas conforme suas necessidades, aumentando a relevância e a utilidade das análises fornecidas.

---

### Resumo das Inclusões:

1. **Gráficos Dinâmicos:**
   - **Gráfico de Contagem e Valores Totais de Empenhos**
   - **Gráfico de Subelementos e Elementos de Despesa**
   Adicionados na seção **Scope** e **Architecture** para detalhar a visualização das despesas públicas.

2. **IA Geradora de Relatórios (Padrão e Personalizado):**
   Integrada no **Scope**, **Architecture**, e **Metrics** para automatizar e personalizar análises financeiras.

3. **Chatbot Aprimorado com Conhecimento sobre CPFs e CNPJs:**
   Atualizado nas seções **Scope**, **Architecture**, e **Metrics** para incluir funcionalidades avançadas de consulta e interatividade.

