---
noteId: "0947b78063ff11efae5d6b430dce5d9d"
tags: []

---

# Data Summary Report


## General Summary of the Data

### Fontes de Dados

1. **Portais de Transparência Municipais:**
   - **Descrição:** Dados públicos sobre despesas e execução orçamentária das câmaras e prefeituras municipais do eixo Bahia|Sergipe.
   - **Formato:** acessíveis via web scraping.
   - **Frequência de Atualização:** diária.
   - **Tipo de Dados:** 
      - Registro de Despesas (Pagamentos).
    - **Objetivo de Uso:**
      - Monitorar diariamente os gastos públicos para detectar anomalias ou desvios.
      - Gerar relatórios de transparência financeira para uso interno.
      - Facilitar a análise de gastos por categoria, por unidade gestora, credor e período.


2. **Dados Abertos do Governo:**
   - **Descrição:** Conjunto de dados financeiros e orçamentários disponibilizados por órgãos centrais de governo, como Tribunal de Contas dos Municípios da Bahia.
   - **Formato:** JSON.
   - **Frequência de Atualização:** Mensal.
   - **Tipo de Dados:** 
      - Registro de Despesas (Pagamentos).
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
     -  Suporte à detecção de anomalias financeiras e auditorias internas.


### Volume de Dados

- Estimativa de 4 MB de dados financeiros por ano, dependendo do tamanho do município e da quantidade de transações financeiras.

## Data Quality Summary

### Integridade dos Dados



### Validação de Dados



## Target Variable

### Variável-Alvo Principal



### Definição



### Uso



## Individual Variables
### Variáveis Principais

1. **NumeroDocumeto:** Identificador único de cada transação financeira.
   - **Tipo de Dados:** Categórico (nominal).
  
2. **Empenho:** Número do empenho associado à transação.
   - **Tipo de Dados:** Numérico (discreto).
  
3. **Data:** Data em que a transação foi registrada.
   - **Tipo de Dados:** Temporal (data).
  
4. **Município:** Nome do município onde a transação foi realizada.
   - **Tipo de Dados:** Categórico (nominal).
  
5. **Credor:** Nome do credor que recebeu o pagamento.
   - **Tipo de Dados:** Categórico (nominal).
  
6. **Valor:** Montante financeiro da transação.
   - **Tipo de Dados:** Numérico (contínuo).
  
7. **Elemento Gestor:** Código e descrição da categoria de despesa.
   - **Tipo de Dados:** Categórico (nominal).
  
8. **Fonte:** Código e descrição da fonte de recursos.
   - **Tipo de Dados:** Categórico (nominal).


### Descrição Estatística

## Variable Ranking

### Rankeamento de Variáveis

## Relationship Between Explanatory Variables and Target Variable

### Análise de Correlação

### Visualização


## Conclusão

