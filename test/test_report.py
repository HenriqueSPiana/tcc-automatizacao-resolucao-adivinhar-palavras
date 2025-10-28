import sys
import os
import time
import nltk
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any
import csv
from threading import Lock 




# --- Configuração de Path ---
# Encontra o diretório do script atual (test/)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Sobe um nível para o diretório raiz (seu_projeto/)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
# Encontra o diretório src/
SRC_DIR = os.path.join(ROOT_DIR, 'src')





# Adiciona 'src/' ao sys.path para permitir importações
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)




# --- Imports dos Módulos do Usuário ---

from solver import ContextoSolver
from contexto_api import ContextoAPI
from contexto_offline import ContextoOffline
from contexto_online import ContextoOnline
from prepare_embedding import load_huggingface_to_gensim
from gensim.models import KeyedVectors

def run_analysis_case(
    test_name: str,
    solver_factory: Callable[[], ContextoSolver],
    max_attempts: int,
    logs_dir: str  # <--- NOVO PARÂMETRO
) -> dict:
    """
    Executa um único cenário do solver e retorna estatísticas detalhadas.
    Gera um arquivo de log de tentativas na pasta 'logs_dir'.
    """
    
    print(f"[EXECUTANDO]... {test_name}")
    start_time = time.time()
    original_cwd = os.getcwd()
    report = {"nome": test_name}
    
    # Prepara o nome do arquivo de log (ex: "api_dia_1.log")
    safe_filename = test_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    log_file_path = os.path.join(logs_dir, f"{safe_filename}.log")
    
    # --- MUDANÇA (1/2): Inicializa solver como None ---
    solver = None 
    
    try:
        # Muda para o diretório RAIZ do projeto
        os.chdir(ROOT_DIR)
        
        # Instancia o solver 
        solver = solver_factory()
        
        # Executa com verbose=False para não poluir o console principal
        history = solver.solve(max_attempts=max_attempts, verbose=False)
        
        # Coleta os resultados
        report['sucesso'] = solver.best_rank == 1
        report['tentativas_totais'] = len(history)
        report['melhor_rank'] = solver.best_rank
        if solver.best_words:
            report['melhor_palavra'] = solver.best_words[0][0]
        else:
            report['melhor_palavra'] = "N/A"

        try:
            

            limit = solver.optimization_rank_limit 
            best_rank_so_far = float('inf')
            optimization_mode = False
            
            with open(log_file_path, 'w', encoding='utf-8') as log_f:
                log_f.write(f"--- Log de Execução para: {test_name} ---\n")
                log_f.write(f"Limite de Otimização: {limit}\n")
                log_f.write(f"Total de Tentativas: {len(history)}\n")
                log_f.write(f"Resultado: {'SUCESSO' if report['sucesso'] else 'FALHA'}\n")
                log_f.write(f"Melhor Palavra: {report['melhor_palavra']} (Rank: {report['melhor_rank']})\n")
                log_f.write("-" * 40 + "\n\n")
                
                # Recria o log detalhado a partir do histórico
                for attempt_num, (word, rank) in enumerate(history, start=1):
                    if rank < best_rank_so_far:
                        best_rank_so_far = rank
                        # Simula a ativação do modo de otimização
                        if not optimization_mode and best_rank_so_far < limit:
                            optimization_mode = True
                    
                    mode = "OPTIMIZE" if optimization_mode else "EXPLORE"
                    # Simula a linha de log do verbose=True
                    log_f.write(f"Tentativa {attempt_num:03d} [{mode}]: {word:<15} → rank {rank:<5} (Melhor: {best_rank_so_far})\n")
                
                if report['sucesso']:
                    log_f.write(f"Descoberta em {len(history)} tentativas: {report['melhor_palavra']}\n")
                else:
                    log_f.write(f"Não encontrou a palavra. Melhor tentativa: '{report['melhor_palavra']}' (rank {report['melhor_rank']})\n")

        except Exception as log_e:
            print(f"[ERRO DE LOG] Falha ao escrever log para {test_name}: {log_e}")
        # --- FIM DO BLOCO DE LOG ---

    except Exception as e:
        report['sucesso'] = False
        report['erro'] = str(e)
        report['melhor_rank'] = float('inf')
        report['tentativas_totais'] = 0
        report['melhor_palavra'] = "ERRO"
    
    finally:
        # Garante que o CWD seja restaurado
        os.chdir(original_cwd)
        
        # --- MUDANÇA (2/2): Bloco 'finally' corrigido ---
        # Garante que o browser (se houver) seja fechado
        # Verifica se o solver foi instanciado e se a query_function é um método
        # de uma instância ContextoOnline.
        try:
            if solver and hasattr(solver.query_function, '__self__'):
                instance = solver.query_function.__self__
                if isinstance(instance, ContextoOnline):
                    # print(f"[CLEANUP] Fechando browser do ContextoOnline for {test_name}...")
                    instance.close()
        except Exception as e:
            print(f"[ERRO NO CLEANUP] Falha ao fechar o browser: {e}")
        # --- FIM DA MUDANÇA ---

    return report

# --- FUNÇÃO MODIFICADA ---
def generate_report(report_list: List[Dict[str, Any]], output_dir: str, filename: str = "analysis_report.txt") -> str:
    """
    Gera um arquivo TXT de relatório resumido a partir de uma lista de dicionários.

    Args:
        report_list: Uma lista de dicionários, onde cada dicionário é um 'report'
                     vindo da função 'run_analysis_case'.
        output_dir: O diretório onde o arquivo TXT será salvo.
        filename: O nome do arquivo TXT a ser gerado.

    Returns:
        O caminho completo para o arquivo TXT gerado.
    """
    
    if not report_list:
        print("[AVISO] A lista de relatórios está vazia. Nenhum TXT será gerado.")
        return ""

    # Garante que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Define o caminho completo do arquivo
    filepath = os.path.join(output_dir, filename)
    
    # Define o formato (largura das colunas)
    header_fmt = "{:<25} | {:<8} | {:<12} | {:<12} | {:<18} | {}\n"
    line_fmt   = "{:<25} | {:<8} | {:<12} | {:<12} | {:<18} | {}\n"
    
    print(f"[RELATÓRIO] Gerando TXT em: {filepath}")

    try:
        with open(filepath, 'w', encoding='utf-8') as txtfile:
            # Escreve o cabeçalho
            txtfile.write(header_fmt.format(
                "NOME DO TESTE", 
                "SUCESSO", 
                "TENTATIVAS", 
                "MELHOR_RANK", 
                "MELHOR_PALAVRA", 
                "ERRO"
            ))
            txtfile.write("-" * 110 + "\n") # Linha separadora
            
            # Escreve os dados
            for report_data in report_list:
                txtfile.write(line_fmt.format(
                    str(report_data.get("nome", "N/A")),
                    str(report_data.get("sucesso", "N/A")),
                    str(report_data.get("tentativas_totais", "N/A")),
                    str(report_data.get("melhor_rank", "N/A")),
                    str(report_data.get("melhor_palavra", "N/A")),
                    str(report_data.get("erro", "N/A"))
                ))
                
        print(f"[RELATÓRIO] Geração do TXT concluída com {len(report_list)} registros.")
        return filepath
        
    except IOError as e:
        print(f"[ERRO TXT] Falha ao escrever arquivo TXT em {filepath}: {e}")
        return ""
    except Exception as e:
        print(f"[ERRO INESPERADO] Ocorreu um erro ao gerar o TXT: {e}")
        return ""
    

def main():
    
    # 1. Dependências
    print("Baixando pacotes NLTK (stopwords, rslp)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)

    try:
        print("Pré-carregando NLTK stopwords e stemmer...")
        _ = nltk.corpus.stopwords.words("portuguuese")
        _ = nltk.stem.RSLPStemmer()
        print("Recursos NLTK pré-carregados com sucesso.")
    except Exception as e:
        print(f"Erro fatal ao pré-carregar recursos NLTK: {e}")
        sys.exit(1)

    # Instala o playwright (necessário para ContextoOnline)
    # print("Instalando dependências do Playwright...")
    # os.system(f"{sys.executable} -m playwright install") 

    # 2. Carregar Modelo
    print("Carregando modelo word embedding (Gensim/HF)...")
    start_model_load = time.time()
    try:
        # Certifique-se que 'load_huggingface_to_gensim' está importado
        model = load_huggingface_to_gensim("nilc-nlp/glove-300d")
        print(f"Modelo carregado em {time.time() - start_model_load:.2f}s.")
    except Exception as e:
        print(f"Erro fatal ao carregar o modelo: {e}")
        sys.exit(1)

    # 3. Definir Cenários de Análise
    output_directory = "reports"

    DIA_INICIAL = 1
    DIA_FINAL = 6  
    MAX_TENTATIVAS_POR_DIA = 100 
    
    test_cases = []
    for dia in range(DIA_INICIAL, DIA_FINAL + 1):
        test_cases.append({
            "nome": f"API (Dia {dia})",
            "factory": lambda d=dia: ContextoSolver(model, ContextoAPI(dia=d).query),
            "max_attempts": MAX_TENTATIVAS_POR_DIA
        })
        
    # Exemplo de como adicionar um teste ContextoOnline (requer Playwright)
    # def online_factory():
    #     contexto = ContextoOnline(headless=True).start()
    #     return ContextoSolver(model, contexto.query)
    # test_cases.append({
    #     "nome": "Online (Hoje)",
    #     "factory": online_factory,
    #     "max_attempts": MAX_TENTATIVAS_POR_DIA
    # })
        

    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(LOGS_DIR, exist_ok=True)

    

    print(f"\nIniciando execução de {len(test_cases)} casos de teste...")
    start_all_time = time.time()
    
    results_list = [] # <-- Lista para guardar resultados REAIS
    max_workers = 4 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_analysis_case, 
                case["nome"], 
                case["factory"], 
                case["max_attempts"],
                LOGS_DIR           
            ): case 
            for case in test_cases
        }

        print(f"Testes submetidos (usando até {max_workers} workers). Aguardando conclusão...")
        
        # Itera sobre as futures à medida que elas terminam
        for future in as_completed(futures):
            case = futures[future]
            try:
                report = future.result() # Pega o dicionário de resultado
                results_list.append(report)
                
                # Fornece feedback visual
                status = "SUCESSO" if report.get('sucesso') else "FALHA"
                if 'erro' in report:
                     status = f"ERRO ({report['erro']})"
                print(f"[CONCLUÍDO] {case['nome']} (Status: {status})")
                
            except Exception as e:
                # Captura erros inesperados na própria thread
                print(f"[ERRO FATAL NO WORKER] {case['nome']} falhou: {e}")
                results_list.append({
                    "nome": case['nome'],
                    "sucesso": False,
                    "erro": str(e),
                    "melhor_rank": float('inf'),
                    "tentativas_totais": 0,
                    "melhor_palavra": "ERRO FATAL"
                })
    
    print(f"\nTodos os testes concluídos em {time.time() - start_all_time:.2f}s.")

    
    if not results_list:
        print("Nenhum resultado foi coletado. O relatório TXT não será gerado.")
        return

    # --- CHAMADA MODIFICADA ---
    generated_file_path = generate_report(
        report_list=results_list, 
        output_dir=output_directory,
        filename="meu_relatorio_de_analise.txt" # <-- Mudado de .csv para .txt
    )
    
    if generated_file_path:
        print(f"\nArquivo de relatório real gerado: {generated_file_path}")
    else:
        print("\nFalha ao gerar o arquivo de relatório TXT.") # <-- Mensagem atualizada




if __name__ == "__main__":
    main()