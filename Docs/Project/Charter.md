# Project Charter

### Business background

* **Cliente:** Câmaras e Prefeituras Municipais do eixo Bahia|Sergipe.
* **Domínio de Negócio:** Gestão Pública, Controle Interno, Finanças Públicas.
* **Problemas de Negócio:** 
  * Dificuldade na gestão e monitoramento eficaz das finanças públicas devido à complexidade dos sistemas de contabilidade existentes.
  * Falta de interfaces intuitivas e painéis de controle que facilitem a visualização clara dos gastos públicos.
  * Incapacidade de identificar padrões suspeitos ou anomalias nas despesas financeiras, comprometendo a transparência e a segurança.
  * Necessidade de previsões orçamentárias precisas para evitar déficits ou superávits desnecessários.



### Scope

* **Soluções de Ciência de Dados a Serem Desenvolvidas:**
  * Dashboard interativo para visualização referente à despesa pública.
  * Modelos de aprendizado de máquina para classificação automática de credores e detecção de anomalias em registros financeiros.
  * Ferramentas de análise para previsão de tendências sazonais e necessidades orçamentárias.

* **O que faremos:**
  * Coletar dados financeiros dos Portais de Transparência, central de dados do governo e sistemas internos de contabilidade.
  * Realizar análise exploratória de dados (EDA) para entender a estrutura e qualidade dos dados.
  * Desenvolver e implementar modelos de aprendizado de máquina para atender às necessidades identificadas.
  * Criar uma aplicação web utilizando Streamlit para disponibilizar os resultados para os Controles Internos.

* **Consumo pelo Cliente:**
  * A solução será acessível via interface web, permitindo que os Controles Internos das câmaras e prefeituras municipais utilizem a aplicação para monitorar e analisar as finanças públicas de seus municípios de forma eficiente e intuitiva.


### Personnel

* ControlGov:
  - **Administrador de Dados:** Renoaldo Costa
* Client:
  - **Testador:** Gidelma dos Santos Bomfim (Pinhão/SE)
  - **Contato de Negócios de Prefeitura:** Raimunda Alvez (Pojuca/BA)
  - **Contato de Negócios de Câmara:** Sérgio Benedito (Nilo Peçanha/BA)
  - **Contato de Negócios de Câmara:** Dárcio Piatã (Piatã/BA)
	

### Metrics
* **Objetivos Qualitativos:**
  * Melhorar a eficiência do monitoramento financeiro e a transparência nos gastos públicos.
  * Facilitar a identificação de fraudes, erro e má gestão dos recursos públicos.

* **Métricas Quantitativas:**
  * Reduzir o tempo médio necessário para análises financeiras em pelo menos 50%.
  * Atingir uma precisão de 95% na detecção de anomalias e padrões suspeitos.
  * Melhorar a acurácia das previsões orçamentárias em pelo menos 80% comparado aos métodos tradicionais.
  * Alcançar uma satisfação do usuário de 85% em termos de facilidade de uso da aplicação.

* **Baseline e Metodologia de Medição:**
  * **Baseline:** Tempo atual de análise financeira, precisão na detecção de anomalias e previsões orçamentárias serão registrados antes da implementação.
  * **Medidas:** Acompanhamento contínuo através de feedback dos usuários, análise de logs da aplicação e comparações periódicas com os valores de baseline.

### Plan
  1. **Planejamento e Definição de Escopo:** Identificar stakeholders, definir objetivos detalhados, criar o Project Charter. *(15 dias)*
  2. **Aquisição e Entendimento dos Dados:** Coleta de dados, EDA, validação de dados. *(15 dias)*
  3. **Desenvolvimento de Modelos:** Criação e teste de modelos de aprendizado de máquina para classificação de credores e detecção de anomalias. *(1 mês)*
  4. **Desenvolvimento do Dashboard:** Implementação da interface web usando Streamlit, integração dos modelos de ML. *(1 mês)*
  5. **Testes e Validação:** Testes com usuários finais, ajuste de funcionalidades, melhoria contínua. *(1 mês)*
  6. **Implementação em Produção:** Lançamento da aplicação, treinamento de usuários, documentação. *(15 dias)*


### Architecture
* **Dados:**
  * **Fontes de Dados:** Dados financeiros dos Portais de Transparência, dados abertos do governo e sistemas internos de contabilidade.
  * **Movimentação de Dados:** Utilização de APIs, webscraping, RPA e exportação de arquivos CSV e JSON para coleta de dados.
  * **Armazenamento e Processamento:** Utilização de ferramentas como selenium para webscraping e RPA, pandas para processamento e transformação de dados, scikit-learn e TensorFlow para modelagem.
  * **Ferramentas de Análise:** Dashboards interativos construídos com Streamlit, hospedados em servidor seguro para acesso dos usuários.

* **Uso de Resultados:**
  * A solução permitirá que os Controles Internos das câmaras e prefeituras municipais visualizem os resultados em tempo real, facilitando a tomada de decisões informadas e baseadas em dados.
  * A automação de tarefas e a detecção de anomalias serão incorporadas ao fluxo de trabalho diário dos usuários.

* **Diagramas de Fluxo de Dados:**
  * Criar diagramas mostrando o fluxo de dados desde a coleta até a visualização e uso dos resultados para facilitar o entendimento do processo.


### Communication
* **Reuniões Mensais:** Reuniões semanais de acompanhamento com a equipe de projeto para monitorar o progresso e ajustar o cronograma conforme necessário.
* **Relatórios de Progresso Mensais:** Relatórios mensais para stakeholders para manter todos informados sobre o progresso do projeto.
* **Contato de Comunicação:**
  - **Equipe Técnica:** Whatsapp
  - **Cliente:** Whatsapp
