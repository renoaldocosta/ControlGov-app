---
noteId: "6af3a9c08fde11efb4ca29123a3d3bae"
tags: []

---



## **1. Atualização do Project Charter**

### **a. Business Background**
1. **Novas Funcionalidades:** Quais são as novas funcionalidades que estão sendo adicionadas ao projeto, além das já mencionadas anteriormente?
   1. Adição de APIs utilizando fastapi, tanto para consultar trazer os dados do banco mongodb para o front-end com streamlit, quanto para iserir dados do scraper e mandar para o bando de dados. 
   As rotas de api para credores trazem a lista de credores que tiveram empenho na camara, alem de uma rota que traz os valores por mês e ano acumulado por cada um desses credores.
   A rota de elemento e a rota de subelemento trazem a soma total e a soma por mês e ano dos valores empenhados para cada um dos elemnetos e subelementos. 
   Foi realizado o deploy de três aplicações: A aplicação em si com streamlit; Um scraper com beautifulsoup que realiza o scraper todos os dias as 2AM para uma base de stage, e 'migra' os dados da stage para um bd final que serve a aplicação com front-end; Deploy de um Backend API com fastapi. Os três sistemas estão conversando.
2. **Novas Ferramentas:** Além do FastAPI e Selenium, há outras ferramentas ou tecnologias que estão sendo consideradas ou já implementadas?
   1. MongoDB foi inserido para armazenar os dados das aplicações. Pretende inserir funcionalidade com whatsapp, para receber notificações. Talvez uso de scrapy para olhar se os sites estão online e avisar caso não estejam. Langchain para inserir tecnologia de LLM.
3. **Objetivos Revisados:** Houve alguma mudança nos objetivos do projeto com a introdução das novas ferramentas? Se sim, quais são esses novos objetivos?
   1. Houve o adiamento das raspagens dos dados da prefeitura de Pojuca/BA, dado que já há a raspagem de dados da câmara de Pinhão/SE.
### **b. Problemas de Negócio**
1. **Novos Problemas Identificados:** Com a implementação de novas ferramentas, surgiram novos problemas de negócio que precisam ser abordados?
   1. Sim. A segurança da API. Por enquanto qualquer um pode inserir e deletar dados do MongoDB usando a API criada. Necessário implementar medidas de segurança.
2. **Solução das Novas Ferramentas:** Como as novas ferramentas (FastAPI, Selenium) ajudam a resolver os problemas de negócio identificados anteriormente?
   1. O selenium não foi utilizado por enquanto. Contudo, o FastAPI ajuda a tornar as aplicações mais dinâmicas e simples, permitindo retorno de dados com agregação mediante o mongodb pela chamada de uma api em um camminho especifico. 
   Além disso, permite a possibilidade de alimentar um LLM com API com vistas a dar ferramentas a essas LLMs.

### **c. Scope (Escopo)**
6. **Novas Soluções de Ciência de Dados:** Existem novas soluções de ciência de dados que serão desenvolvidas com a introdução das novas ferramentas?
   1. Sim, uso de LLMs.
7. **Integração com Ferramentas Existentes:** Como o FastAPI e o Selenium serão integrados às soluções existentes?
   1. O selenium já foi integrado tanto com a alimentação dos dados do frontend com streamlit quanto com a possibilidade de consulta por parte de LLMs.

### **d. Consumo pelo Cliente**
8. **Acesso às Novas Funcionalidades:** As novas funcionalidades serão acessíveis via a mesma interface web ou haverá novas interfaces?
   1. Os dois. O Fastapi alimenta os dados das tabelas e graficos da aplicação, mas também irá alimentar interfaces de relatórios e LLMs.
9.  **Impacto para os Usuários Finais:** Como as novas funcionalidades impactarão a experiência dos Controles Internos e gestores públicos?
    1.  Mais rapidez e customização quando for consultar dados. Rapidez frente a consulta de dados para tabelas e gráficos quanto customização com resposta por parte das aLLMs.

### **e. Personnel (Equipe)**
10. **Novos Papéis e Responsabilidades:** Com a introdução de novas ferramentas, há novos papéis ou responsabilidades dentro da equipe?
    1.  Não, continua como está.
11. **Treinamento Necessário:** Que tipos de treinamento serão necessários para a equipe técnica e para os usuários finais para utilizar as novas ferramentas?
    1.  Será necessário treinar o pessoal para saber como perguntar e o que perguntar ao modelo de LLM.

### **f. Metrics (Indicadores e Metas)**
12. **Novos Indicadores de Desempenho:** Há novos indicadores de desempenho que precisam ser monitorados devido às novas funcionalidades?
    A acurácia das LLMs em 90% e Consultas da API em 100%.
13. **Metas Atualizadas:** As metas atuais precisam ser ajustadas para refletir as melhorias trazidas pelas novas ferramentas?
    1.  

### **g. Plan (Plano de Projeto)**
14. **Cronograma Revisado:** Como a introdução de FastAPI e Selenium afeta o cronograma do projeto? Há novas etapas ou mudanças nas existentes?
    1.  Não. Não há novas etapas.
15. **Dependências e Riscos:** Existem novas dependências ou riscos associados à implementação das novas ferramentas que precisam ser gerenciados?
    1.  Sim, segurança dos dados da API. Necessário implementar medidas de autenticação.

### **h. Architecture (Arquitetura)**
16. **Mudanças na Arquitetura:** Como a arquitetura do sistema será modificada com a inclusão de FastAPI e Selenium?
17. **Fluxo de Dados Atualizado:** O fluxo de dados sofrerá alterações significativas devido às novas ferramentas? Se sim, como?
    1.  Sim. Os dados coletados via webscrap são envaminhados para um db stage e um script faz a migração desses dados para uma outra coleção com formatos de dados que permitam consultas diretamente no banco de dados, tornando mais eficiente o processo de carregamento dos dados.

### **i. Communication (Comunicação)**
18. **Novos Canais de Comunicação:** Há necessidade de estabelecer novos canais de comunicação para gerenciar as novas funcionalidades?
    1.  não.
19. **Atualização das Estratégias de Comunicação:** As estratégias de comunicação atuais precisam ser ajustadas para incluir informações sobre as novas ferramentas e funcionalidades?
    1.  Não.

## **2. Atualização do Data Summary Report**

### **a. Fontes de Dados**
20. **Novas Fontes de Dados:** Foram identificadas novas fontes de dados além das já mencionadas? Se sim, quais são elas?
    1.  As fontes primárias são as mesmas. Mas há planejamento para fazer requisições aos portais de transparência e guardar a disponibilidade destes em mostrar em um gráfico. resaltando a saúde do site ao longo do tempo.
21. **Scraping Dinâmico:** Como o scraping dinâmico com Selenium está sendo implementado? Quais portais ou fontes estão sendo acessados dinamicamente?
    1.  Não foi implementado ainda.
22. **Formatos Adicionais:** Há novos formatos de dados que estão sendo coletados com as novas ferramentas?
    Não
    2.  

### **b. Volume de Dados**
23. **Aumento no Volume de Dados:** A introdução de novas fontes ou scraping dinâmico afetou a estimativa de volume de dados? Houve um aumento significativo?
    1.  Não.
24. **Capacidade de Armazenamento:** A infraestrutura atual de armazenamento suporta o aumento no volume de dados? Há planos para expandir, se necessário?
    1.  Sim, os dados estão sendo armazenados no mongodb Atlas e tem suporte para crescimento.

### **c. Data Quality Summary**
25. **Qualidade dos Dados Coletados Dinamicamente:** Como a qualidade dos dados coletados via scraping dinâmico está sendo assegurada?
    1.  não foi inserido coleta dinamica
26. **Novas Técnicas de Limpeza e Transformação:** As novas ferramentas introduzem novos métodos de limpeza e transformação de dados? Se sim, quais?
    1.  não

### **d. Target Variable (Variável Alvo)**
27. **Novas Variáveis Alvo:** Com as novas ferramentas e fontes de dados, há novas variáveis-alvo que precisam ser consideradas?
    
28. **Objetivos das Novas Variáveis:** Quais são os novos objetivos associados a essas variáveis-alvo?

### **e. Individual Variables (Variáveis Individuais)**
29. **Novas Variáveis Explicativas:** Há novas variáveis independentes que estão sendo coletadas ou criadas com as novas funcionalidades?
30. **Segmentação Atualizada:** A segmentação dos dados para análise mudou com as novas ferramentas? Se sim, como?

### **f. Variable Ranking (Ranking de Variáveis)**
31. **Impacto das Novas Variáveis:** Como as novas variáveis influenciam o ranking de importância das variáveis existentes?
32. **Novas Análises de Importância:** Foram realizadas novas análises para determinar a importância das variáveis introduzidas pelas novas ferramentas?

### **g. Formatos e Transformação de Dados**
33. **Novos Métodos de Transformação:** As novas ferramentas introduzem novos métodos ou formatos de transformação de dados? Se sim, quais?
34. **Automação das Transformações:** Há novos processos automatizados para transformação de dados utilizando FastAPI ou Selenium?
    1.  Sim, foi implementada as modelagem com pydantic para definir formato dos dados que são requisitados mediante junto do streamlit. 

## **3. Considerações Gerais**

### **a. Segurança e Privacidade**
35. **Impacto das Novas Ferramentas na Segurança:** Como a introdução de FastAPI e Selenium afeta as medidas de segurança e privacidade dos dados?
    1.  É necessário implementar medidas de autenticação para utilização da API para inserção, update e delete dos dados.
36. **Compliance com Novas Ferramentas:** As novas ferramentas estão em conformidade com as diretrizes de privacidade e proteção de dados aplicáveis?
    1.  Sim, os dados utilizados pela aplicação são todos provenientes do portal da transparência.

### **b. Integração das Ferramentas**
37. **Integração com Sistemas Existentes:** Como FastAPI e Selenium serão integrados aos sistemas e processos já existentes no projeto?
    1.  O fastapi alimenta o streamlit e será a a interface para script o webscrap inserir dados no banco mongodb.
38. **Compatibilidade das Ferramentas:** As novas ferramentas são compatíveis com a infraestrutura atual? Há necessidade de ajustes ou atualizações?
    1.  Há compatibilide. Foi necessário apenas ajustar os scripts do codigo para fazer deploy na plataforma. 

### **c. Treinamento e Capacitação**
39. **Necessidade de Treinamento Específico:** Que tipos de treinamento específico serão necessários para a equipe técnica e usuários finais com as novas ferramentas?
    1.  
40. **Recursos de Aprendizado:** Quais recursos (documentação, tutoriais, workshops) serão fornecidos para facilitar a adoção das novas ferramentas?
    1.  será gravado um video para auxiliar os usuários a fazer o Download dos dados da bahia e upload dos dados na aplicação. 

### **d. Feedback e Iteração**
41. **Mecanismos de Feedback:** Como será coletado o feedback dos usuários sobre as novas funcionalidades e ferramentas?
    1.  será questionado diretamente ou seria realizada pesquisa mediante formulário.
42. **Processo de Iteração:** Qual será o processo para iterar e melhorar as novas funcionalidades com base no feedback recebido?
    1.  Mensalmente serão reunidos os feedbacks e implementados.

