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
    max_attempts: int
) -> dict:
    """
    Executa um único cenário do solver e retorna estatísticas detalhadas.
    """

    
    print(f"[EXECUTANDO]... {test_name}")
    start_time = time.time()
    original_cwd = os.getcwd()
    report = {"nome": test_name}
    
    try:
        # Muda para o diretório RAIZ do projeto
        os.chdir(ROOT_DIR)
        
        # Instancia o solver 
        solver = solver_factory()
        
        history = solver.solve(max_attempts=max_attempts, verbose=False)
        
        # Coleta os resultados
        report['sucesso'] = solver.best_rank == 1
        report['tentativas_totais'] = len(history)
        report['melhor_rank'] = solver.best_rank
        if solver.best_words:
            report['melhor_palavra'] = solver.best_words[0][0]
        else:
            report['melhor_palavra'] = "N/A"

    except Exception as e:
        report['sucesso'] = False
        report['erro'] = str(e)
        report['melhor_rank'] = float('inf')
        report['tentativas_totais'] = 0
        report['melhor_palavra'] = "ERRO"
    
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
    DIA_FINAL = 20
    # Alterado conforme sua solicitação
    MAX_TENTATIVAS_POR_DIA = 100 
    # ---------------------------------

    test_cases = []
    for dia in range(DIA_INICIAL, DIA_FINAL + 1):
        test_cases.append({
            "nome": f"API (Dia {dia})",
            
            # O 'd=dia' é crucial. Ele "captura" o valor de 'dia'
            # no momento da criação do lambda. Sem isso, todos os testes
            # rodariam para o último dia (DIA_FINAL) por causa do escopo.
            "factory": lambda d=dia: ContextoSolver(model, ContextoAPI(dia=d).query),
            
            "max_attempts": MAX_TENTATIVAS_POR_DIA
        })
        
    print(f"{len(test_cases)} cenários de teste (API Dias {DIA_INICIAL}-{DIA_FINAL}) foram criados.")

    # Se você também quiser rodar testes Offline JUNTAMENTE com os 1000 dias:
    # test_cases.append(
    #    {"nome": "Offline (animal)", "factory": lambda: ContextoSolver(model, ContextoOffline(model, "animal").query_rank), "max_attempts": 100}
    # )
    # test_cases.append(
    #    {"nome": "Offline (computador)", "factory": lambda: ContextoSolver(model, ContextoOffline(model, "computador").query_rank), "max_attempts": 100}
    # )

    # --- CORREÇÃO: Inicializa a lista de resultados ---
    all_results = [] 

    # 4. Executar Concorrentemente
    # Alterado de 4 para 3 para reduzir a carga na rede
    max_workers = 8
    print(f"\nIniciando {len(test_cases)} cenários concorrentes (max_workers={max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_analysis_case, 
                case["nome"], 
                case["factory"], 
                case["max_attempts"]
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