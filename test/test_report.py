import sys
import os
import time
import nltk
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any






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

        # --- NOVO BLOCO: GERA O ARQUIVO DE LOG ---
        try:
            # Obtém o limite de otimização do solver
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
                            log_f.write(f"--- ATIVANDO MODO DE OTIMIZAÇÃO (MELHOR RANK: {best_rank_so_far}) ---\n\n")
                    
                    mode = "OPTIMIZE" if optimization_mode else "EXPLORE"
                    # Simula a linha de log do verbose=True
                    log_f.write(f"Tentativa {attempt_num:03d} [{mode}]: {word:<15} → rank {rank:<5} (Melhor: {best_rank_so_far})\n")
                
                log_f.write("\n" + "-" * 40 + "\n")
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
        # Tenta salvar o erro no log também
        with open(log_file_path, 'w', encoding='utf-8') as log_f:
            log_f.write(f"--- ERRO NA EXECUÇÃO DE: {test_name} ---\n")
            log_f.write(str(e))
    
    finally:
        # Garante que o CWD seja restaurado
        os.chdir(original_cwd)
        # Garante que o browser (se houver) seja fechado
        if isinstance(solver_factory.__self__, ContextoOnline):
             solver_factory.__self__.close()

    report['tempo_s'] = time.time() - start_time
    status = "SUCESSO" if report['sucesso'] else "FALHA"
    print(f"[CONCLUÍDO] {test_name:<20} | {status:<7} | Rank: {report['melhor_rank']:<5} | Tentativas: {report['tentativas_totais']}")
    return report


def generate_report(results: List[Dict[str, Any]], output_dir: str):
    """
    Processa a lista de resultados e gera o relatório estatístico
    e o gráfico em pizza.
    """
    print("\n\n" + "="*50)
    print("      GERANDO RELATÓRIO DE DESEMPENHO DO SOLVER")
    print("="*50)

    total_runs = len(results)
    if total_runs == 0:
        print("Nenhuma execução foi concluída. Saindo.")
        return

    # 1. Coleta de Dados
    successes = [r for r in results if r.get('sucesso') == True]
    failures = [r for r in results if r.get('sucesso') == False]
    
    num_success = len(successes)
    num_failure = len(failures)

    # 2. Cálculos Estatísticos
    perc_success = (num_success / total_runs) * 100
    perc_failure = (num_failure / total_runs) * 100
    
    # Média de tentativas para acertar
    attempts_success_list = [r['tentativas_totais'] for r in successes]
    avg_attempts_success = np.mean(attempts_success_list) if attempts_success_list else 0
    
    # Média do melhor ranking ao errar
    # Usar .get() para ser seguro contra KeyErrors. Se a chave não existir, retorna 'inf'
    ranks_list = [r.get('melhor_rank', float('inf')) for r in failures]
    best_rank_failure_list = [rank for rank in ranks_list if rank != float('inf')]
    avg_rank_failure = np.mean(best_rank_failure_list) if best_rank_failure_list else 0

    # 3. Relatório no Console
    print("\n--- ESTATÍSTICAS GERAIS ---")
    print(f"Execuções Totais: {total_runs}")
    print(f"Acertos: {num_success} ({perc_success:.1f}%)")
    print(f"Erros:   {num_failure} ({perc_failure:.1f}%)")
    
    print("\n--- MÉDIAS DE DESEMPENHO ---")
    print(f"Média de Tentativas (em Acertos): {avg_attempts_success:.2f}")
    print(f"Média do Melhor Rank (em Erros):  {avg_rank_failure:.2f}")

    if failures:
        print("\n--- DETALHAMENTO DE FALHAS ---")
        for f in failures:
            if 'erro' in f:
                print(f"  - {f['nome']:<25} | ERRO: {f['erro']}")
            else:
                print(f"  - {f['nome']:<25} | Melhor Rank: {f['melhor_rank']:<5} (Palavra: {f['melhor_palavra']})")

    # 4. Geração do Gráfico em Pizza
    try:
        labels = ['Acertos', 'Erros']
        sizes = [num_success, num_failure]
        colors = ['#4CAF50', '#F44336'] # Verde, Vermelho
        explode = (0.1 if num_success > 0 else 0, 0) # Destaca a fatia de acertos

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * total_runs / 100),
            shadow=True,
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax.axis('equal')  # Assegura que a pizza seja um círculo
        
        plt.title(f'Relatório de Acertos vs Erros ({total_runs} Execuções)', fontsize=16)
        
        report_filename = os.path.join(output_dir, 'relatorio_solver_pizza.png')
        plt.savefig(report_filename)
        
        print("\n" + "-"*50)
        print(f"✅ Gráfico em pizza salvo em: {report_filename}")
        print("="*50)

    except Exception as e:
        print(f"\n[ERRO] Não foi possível gerar o gráfico em pizza: {e}")
        print("Verifique se a biblioteca 'matplotlib' está instalada: pip install matplotlib")

def main():
    print("--- INICIANDO CAMPANHA DE ANÁLISE DO SOLVER ---")
    
    # 1. Dependências
    print("Baixando pacotes NLTK (stopwords, rslp)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)




# ---  Pré-carregamento NLTK ---
    # Força o NLTK a carregar os recursos ANTES de iniciar as threads
    # Isso evita uma "race condition" no LazyCorpusLoader do NLTK
    try:
        print("Pré-carregando NLTK stopwords e stemmer...")
        _ = nltk.corpus.stopwords.words("portuguese")
        _ = nltk.stem.RSLPStemmer()
        print("Recursos NLTK pré-carregados com sucesso.")
    except Exception as e:
        print(f"Erro fatal ao pré-carregar recursos NLTK: {e}")
        print("Verifique sua instalação do NLTK e os dados baixados.")
        sys.exit(1)

    os.system(f"{sys.executable} -m playwright install") # Para ContextoOnline


    # 2. Carregar Modelo
    print("Carregando modelo word embedding (Gensim/HF)...")
    start_model_load = time.time()
    try:
        model = load_huggingface_to_gensim("nilc-nlp/glove-300d")
        print(f"Modelo carregado em {time.time() - start_model_load:.2f}s.")
    except Exception as e:
        print(f"Erro fatal ao carregar o modelo: {e}")
        sys.exit(1)

    # 3. Definir Cenários de Análise
    print("Gerando cenários de teste para os dias 1 a 1000...")
    
    # --- Configuração da Campanha ---
    DIA_INICIAL = 1
    DIA_FINAL = 100  # AVISO: Isso pode levar horas/dias e pode ser bloqueado
    MAX_TENTATIVAS_POR_DIA = 100 
    # ---------------------------------

    test_cases = []
    for dia in range(DIA_INICIAL, DIA_FINAL + 1):
        test_cases.append({
            "nome": f"API (Dia {dia})",
            "factory": lambda d=dia: ContextoSolver(model, ContextoAPI(dia=d).query),
            "max_attempts": MAX_TENTATIVAS_POR_DIA
        })
        
    print(f"{len(test_cases)} cenários de teste (API Dias {DIA_INICIAL}-{DIA_FINAL}) foram criados.")

    # --- NOVO: Criar diretório de Logs ---
    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Logs de cada dia serão salvos em: {LOGS_DIR}")
    # ------------------------------------

    all_results = [] 

    # 4. Executar Concorrentemente



    max_workers = 8  # Ajuste conforme a capacidade do seu sistema 
    print(f"\nIniciando {len(test_cases)} cenários concorrentes (max_workers={max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_analysis_case, 
                case["nome"], 
                case["factory"], 
                case["max_attempts"],
                LOGS_DIR  # <-- NOVO: Passa o caminho dos logs
            ): case 
            for case in test_cases
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                case_name = futures[future]["nome"]
                all_results.append({"nome": case_name, "sucesso": False, "erro": f"Erro no executor: {e}"})

    # 5. Gerar Relatório
    # O relatório será salvo na pasta 'test/'
    generate_report(all_results, TEST_DIR)


if __name__ == "__main__":
    main()