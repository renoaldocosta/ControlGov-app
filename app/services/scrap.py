import requests
from bs4 import BeautifulSoup
import pandas as pd


# Função para consultar a atualização de atos legais e infralegais
def consultar_atualizacao_atos_legais_infralegais():
    """
    Extrai dados da tabela da página especificada, processa-os e gera um relatório.
    
    Retorna:
        resposta (pd.DataFrame ou str): DataFrame contendo o relatório gerado ou uma mensagem de erro.
    """
    # URL da página
    url = 'https://cmpinhao.org/20-2-leis-e-atos-infralegais/'
    
    try:
        # Enviar uma requisição GET para a página
        response = requests.get(url, timeout=20)
        
        # Verificar se a requisição foi bem-sucedida
        if response.status_code != 200:
            return f"Falha ao acessar a página. Status code: {response.status_code}"
        
        # Parsear o conteúdo HTML com BeautifulSoup usando o parser 'lxml'
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Encontrar a tabela com id "table_1"
        table = soup.find('table', id='table_1')
        
        if not table:
            return "Tabela com id 'table_1' não encontrada."
        
        # Extrair os cabeçalhos da tabela
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                for th in header_row.find_all('th'):
                    header = th.get_text(strip=True)
                    headers.append(header)
            else:
                return "Linha de cabeçalho (<tr>) não encontrada."
        else:
            return "Cabeçalho da tabela (<thead>) não encontrado."
        
        # Extrair todas as linhas da tabela
        tbody = table.find('tbody')
        if not tbody:
            return "Corpo da tabela (<tbody>) não encontrado."
        
        rows = tbody.find_all('tr')
        
        # Lista para armazenar os dados
        data = []
        
        for row in rows:
            cells = row.find_all('td')
            row_data = []
            for cell in cells:
                # Extrair texto limpo da célula
                text = cell.get_text(strip=True)
                
                # Se a célula contém links, extrair o href
                if cell.find('a'):
                    link = cell.find('a').get('href')
                    text = f"{text} ({link})"
                
                row_data.append(text)
            
            # Adicionar a linha de dados à lista principal
            data.append(row_data)
        
        # Criar o DataFrame se headers e data estiverem disponíveis
        if not headers or not data:
            return "Não foi possível extrair cabeçalhos ou dados da tabela."
        
        # Ajustar os dados caso o número de células seja menor que os cabeçalhos
        num_headers = len(headers)
        adjusted_data = []
        for row in data:
            if len(row) < num_headers:
                # Preencher células faltantes com None
                row += [None] * (num_headers - len(row))
            elif len(row) > num_headers:
                # Truncar células excedentes
                row = row[:num_headers]
            adjusted_data.append(row)
        
        df = pd.DataFrame(adjusted_data, columns=headers)
        
        # Função para tornar os nomes das colunas únicos
        def make_unique_columns(columns):
            seen = {}
            unique_columns = []
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    unique_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    unique_columns.append(col)
            return unique_columns
        
        df.columns = make_unique_columns(df.columns)
        
        # Converter a coluna 'Data de Publicação' para datetime
        if 'Data de Publicação' not in df.columns:
            return "A coluna 'Data de Publicação' não foi encontrada no DataFrame."
        
        df_order = df.copy()
        df_order['Data de Publicação'] = pd.to_datetime(df_order['Data de Publicação'], format='%d/%m/%Y', errors='coerce')
        
        # Verificar e tratar valores ausentes na coluna 'Data de Publicação'
        if df_order['Data de Publicação'].isnull().any():
            print("Aviso: Existem valores ausentes na coluna 'Data de Publicação'. Eles serão ignorados na ordenação.")
            df_order = df_order.dropna(subset=['Data de Publicação'])
        
        # Ordenar o DataFrame pela data de publicação em ordem decrescente
        df_order = df_order.sort_values(by='Data de Publicação', ascending=False)
        
        # Verificar se a coluna 'Tipo Aprovado' existe
        if 'Tipo Aprovado' not in df_order.columns:
            return "A coluna 'Tipo Aprovado' não foi encontrada no DataFrame."
        
        # Agrupar por 'Tipo Aprovado' e obter o índice da linha com a data mais recente
        try:
            idx = df_order.groupby('Tipo Aprovado')['Data de Publicação'].idxmax()
        except Exception as e:
            return f"Ocorreu um erro ao agrupar e selecionar os índices: {e}"
        
        if not idx.empty:
            # Selecionar essas linhas do DataFrame original
            df_order = df_order.loc[idx, ['Tipo Aprovado', 'Tipo de Matéria - Numeração', 'Data de Publicação']]
            
            # Resetar o índice para uma melhor visualização             
            report = df_order.sort_values(by='Data de Publicação', ascending=False).reset_index(drop=True)
            
            # Resetar o índice para uma melhor visualização
            report = report.reset_index(drop=True)
            
            # Ordenar o relatório por data de publicação em ordem decrescente
            report.sort_values(by='Data de Publicação', ascending=False)
            
            # Tentar salvar o relatório em um arquivo CSV
            try:
                report.to_csv('relatorio_tipo_aprovado.csv', sep=';', index=False, encoding='utf-8-sig') 
                resposta = pd.read_csv('relatorio_tipo_aprovado.csv', sep=';')
                print("Relatório salvo com sucesso em 'relatorio_tipo_aprovado.csv'.")
            except Exception as e:
                resposta = f"Ocorreu um erro ao salvar o relatório: {e}"
        else:
            resposta = "Nenhum dado disponível para gerar o relatório."
        
        return resposta
    
    except Exception as ex:
        return f"Ocorreu um erro inesperado: {ex}"


# Função para consultar a atualização de projetos de leis e atos infralegais
def consultar_atualizacao_projetos_atos_legais_infralegais():
    """
    Extrai dados da tabela da página especificada, processa-os e gera um relatório.
    
    Retorna:
        resposta (pd.DataFrame ou str): DataFrame contendo o relatório gerado ou uma mensagem de erro.
    """
    # URL da página
    url = 'https://cmpinhao.org/20-3-projetos-de-leis-e-de-atos-infralegais/'
    
    try:
        # Enviar uma requisição GET para a página
        response = requests.get(url, timeout=20)
        
        # Verificar se a requisição foi bem-sucedida
        if response.status_code != 200:
            return f"Falha ao acessar a página. Status code: {response.status_code}"
        
        # Parsear o conteúdo HTML com BeautifulSoup usando o parser 'lxml'
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Encontrar a tabela com id "table_1"
        table = soup.find('table', id='table_1')
        
        if not table:
            return "Tabela com id 'table_1' não encontrada."
        
        # Extrair os cabeçalhos da tabela
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                for th in header_row.find_all('th'):
                    header = th.get_text(strip=True)
                    headers.append(header)
            else:
                return "Linha de cabeçalho (<tr>) não encontrada."
        else:
            return "Cabeçalho da tabela (<thead>) não encontrado."
        
        # Extrair todas as linhas da tabela
        tbody = table.find('tbody')
        if not tbody:
            return "Corpo da tabela (<tbody>) não encontrado."
        
        rows = tbody.find_all('tr')
        
        # Lista para armazenar os dados
        data = []
        
        for row in rows:
            cells = row.find_all('td')
            row_data = []
            for cell in cells:
                # Extrair texto limpo da célula
                text = cell.get_text(strip=True)
                
                # Se a célula contém links, extrair o href
                if cell.find('a'):
                    link = cell.find('a').get('href')
                    text = f"{text} ({link})"
                
                row_data.append(text)
            
            # Adicionar a linha de dados à lista principal
            data.append(row_data)
        
        # Criar o DataFrame se headers e data estiverem disponíveis
        if not headers or not data:
            return "Não foi possível extrair cabeçalhos ou dados da tabela."
        
        # Ajustar os dados caso o número de células seja menor que os cabeçalhos
        num_headers = len(headers)
        adjusted_data = []
        for row in data:
            if len(row) < num_headers:
                # Preencher células faltantes com None
                row += [None] * (num_headers - len(row))
            elif len(row) > num_headers:
                # Truncar células excedentes
                row = row[:num_headers]
            adjusted_data.append(row)
        
        df = pd.DataFrame(adjusted_data, columns=headers)
        
        # Função para tornar os nomes das colunas únicos
        def make_unique_columns(columns):
            seen = {}
            unique_columns = []
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    unique_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    unique_columns.append(col)
            return unique_columns
        
        df.columns = make_unique_columns(df.columns)
        
        # Converter a coluna 'Data de Apresentação' para datetime
        if 'Data de Apresentação' not in df.columns:
            return "A coluna 'Data de Apresentação' não foi encontrada no DataFrame."
        
        df_order = df.copy()
        df_order['Data de Apresentação'] = pd.to_datetime(df_order['Data de Apresentação'], format='%d/%m/%Y', errors='coerce')
        
        # Verificar e tratar valores ausentes na coluna 'Data de Apresentação'
        if df_order['Data de Apresentação'].isnull().any():
            print("Aviso: Existem valores ausentes na coluna 'Data de Apresentação'. Eles serão ignorados na ordenação.")
            df_order = df_order.dropna(subset=['Data de Apresentação'])
        
        # Ordenar o DataFrame pela Data de Apresentação em ordem decrescente
        df_order = df_order.sort_values(by='Data de Apresentação', ascending=False)
        
        # Verificar se a coluna 'Tipo de Projeto' existe
        if 'Tipo de Projeto' not in df_order.columns:
            return "A coluna 'Tipo de Projeto' não foi encontrada no DataFrame."
        
        # Agrupar por 'Tipo de Projeto' e obter o índice da linha com a data mais recente
        try:
            idx = df_order.groupby('Tipo de Projeto')['Data de Apresentação'].idxmax()
        except Exception as e:
            return f"Ocorreu um erro ao agrupar e selecionar os índices: {e}"
        
        if not idx.empty:
            # Selecionar essas linhas do DataFrame original
            df_order = df_order.loc[idx, ['Tipo de Projeto', 'Projeto', 'Data de Apresentação']]
            
            # Resetar o índice para uma melhor visualização             
            report = df_order.sort_values(by='Data de Apresentação', ascending=False).reset_index(drop=True)
            
            # Resetar o índice para uma melhor visualização
            report = report.reset_index(drop=True)
            
            # Ordenar o relatório por Data de Apresentação em ordem decrescente
            report.sort_values(by='Data de Apresentação', ascending=False)
            
            # Tentar salvar o relatório em um arquivo CSV
            try:
                report.to_csv('relatorio_tipo_aprovado.csv', sep=';', index=False, encoding='utf-8-sig') 
                resposta = pd.read_csv('relatorio_tipo_aprovado.csv', sep=';')
                print("Relatório salvo com sucesso em 'relatorio_tipo_aprovado.csv'.")
            except Exception as e:
                resposta = f"Ocorreu um erro ao salvar o relatório: {e}"
        else:
            resposta = "Nenhum dado disponível para gerar o relatório."
        
        return resposta
    
    except Exception as ex:
        return f"Ocorreu um erro inesperado: {ex}"




