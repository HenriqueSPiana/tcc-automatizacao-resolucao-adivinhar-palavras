# tcc-automatizacao-resolucao-adivinhar-palavras
## A.1. Objetivo Geral

O objetivo geral do projeto é aplicar conceitos de engenharia e arquitetura de software no desenvolvimento de uma aplicação computacional que automatiza a resolução de jogos de adivinhação de palavras. O foco está na criação de um sistema modular, flexível e com bom desempenho, que aborda requisitos não funcionais essenciais enquanto explora o potencial da Inteligência Artificial e do Processamento de Linguagem Natural (PLN) para simular o raciocínio linguístico humano.

---

## A.2. Etapas do Projeto

### Etapa 1 – Definição do Projeto Computacional

* **Nome do projeto:** Automatização da Resolucao de Jogos de Adivinhar Palavras.

* **Objetivo do projeto:** Desenvolver uma abordagem computacional baseada em Inteligência Artificial para resolver automaticamente jogos de adivinhação de palavras em português, como o *Contexto.me*. O sistema utiliza modelos de representação vetorial de palavras (*word embeddings*) para analisar a similaridade semântica, gerar tentativas e ajustar sua estratégia com base no *feedback* recebido do jogo.

* **Público-alvo:**
    * **Acadêmico e de Pesquisa:** Estudantes e pesquisadores nas áreas de IA e PLN que desejam explorar e testar algoritmos de resolução de desafios linguísticos.
    * **Entretenimento e Educação:** Entusiastas de jogos de palavras e desenvolvedores interessados em criar aplicações educacionais baseadas em linguagem.

* **Funcionalidades principais:**
    * **Interação Multimodal com o Jogo:** O sistema pode se conectar ao jogo de três formas distintas:
        1.  **Online via Browser (`contexto_online.py`):** Utiliza o Playwright para automatizar a interação com a interface web do jogo, simulando um usuário humano.
        2.  **Online via API (`contexto_api.py`):** Comunica-se diretamente com a API oficial do jogo para obter respostas de forma mais rápida e eficiente.
        3.  **Offline (`contexto_offline.py`):** Simula o jogo localmente, usando um modelo de *embedding* e uma palavra secreta predefinida, ideal para testes e depuração.
    * **Lógica de Resolução Inteligente (`solver.py`):**
        * Inicia com palavras-semente para explorar o espaço semântico.
        * Alterna para um "modo de otimização" quando uma palavra com boa pontuação é encontrada, utilizando regressão linear (Ridge) para prever as melhores próximas tentativas.
        * Utiliza estratégias baseadas em similaridade de cossenos e analogias vetoriais (`most_similar`) para gerar novos candidatos a partir das melhores palavras já encontradas.
    * **Gerenciamento de Vocabulário (`solver.py`):**
        * Carrega modelos de *embedding* pré-treinados (GloVe, Word2Vec, etc.) a partir de repositórios como o Hugging Face (`prepare_embedding.py`).
        * Filtra o vocabulário do modelo para conter apenas palavras válidas do português, removendo *stopwords* e termos irrelevantes para otimizar o desempenho.
    * **Prevenção de Repetições (`solver.py`):** Utiliza lematização para evitar tentar palavras com a mesma raiz lexical (ex: "casa", "casas"), tornando a busca mais eficiente.

### Etapa 2 – Identificação dos Requisitos Não Funcionais (RNFs)

1.  **Manutenibilidade:** A capacidade de corrigir, adaptar e melhorar o software facilmente. A separação clara de responsabilidades entre a lógica do resolvedor (`ContextoSolver`) e os métodos de interação com o jogo é crucial.
2.  **Flexibilidade/Adaptabilidade:** A facilidade de adaptar o sistema para diferentes contextos. O resolvedor deve ser capaz de operar com diferentes modelos de *embedding* e diferentes interfaces de jogo sem alterações em sua lógica central.
3.  **Desempenho:** O sistema deve ser capaz de encontrar a palavra secreta em um tempo e número de tentativas razoáveis, otimizando tanto o algoritmo de escolha quanto a forma de comunicação com o jogo.
4.  **Confiabilidade:** O sistema deve lidar com falhas de comunicação (ex: *timeouts* de rede) sem interromper sua execução.

### Etapa 3 – Escolha e Justificativa da Arquitetura

* **Arquitetura Escolhida:** **Arquitetura em Camadas com Injeção de Dependência.**
    O sistema é dividido em três camadas lógicas principais:
    1.  **Camada de Interface com o Jogo:** Abstrai a comunicação com o jogo (`ContextoAPI`, `ContextoOnline`, `ContextoOffline`).
    2.  **Camada de Lógica de Negócio (Resolvedor):** Contém o "cérebro" do sistema (`ContextoSolver`).
    3.  **Camada de Modelo de Dados (Embedding):** Consiste no modelo `KeyedVectors` que fornece as representações vetoriais.

* **Justificativa e Mapeamento com os RNFs:**
    O padrão de **Injeção de Dependência** é fundamental, pois a classe `ContextoSolver` recebe uma `query_function` em seu construtor, desacoplando totalmente a lógica de resolução da forma de interação com o jogo.
    * **Manutenibilidade:** A arquitetura é altamente manutenível. Para dar suporte a um novo jogo, basta criar uma nova classe de interface sem modificar o `ContextoSolver`.
    * **Flexibilidade:** O sistema atende diretamente a este requisito. O usuário pode "injetar" qualquer função de consulta e qualquer modelo de *embedding* compatível.
    * **Desempenho:** A separação de camadas permite escolher a implementação mais performática para cada tarefa (ex: `ContextoAPI` para velocidade, `ContextoOnline` para depuração).
    * **Confiabilidade:** A lógica de tratamento de erros fica contida na Camada de Interface, impedindo que falhas de comunicação afetem a lógica principal.

### Etapa 4 – Estruturação e Desenvolvimento

O produto desta etapa é a aplicação funcional, cujo código-fonte está organizado de forma coerente com a arquitetura escolhida nos seguintes arquivos:
* `solver.py`
* `contexto_api.py`
* `contexto_online.py`
* `contexto_offline.py`
* `prepare_embedding.py`