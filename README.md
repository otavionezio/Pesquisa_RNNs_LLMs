# Trabalho 3ª Etapa – Inteligência Artificial

**Aluno:** Otávio Henrique Nézio Melo
**Disciplina:** Inteligência Artificial – Ciência da Computação
**Instituição:** Centro Universitário Presidente Antônio Carlos (UNIPAC Barbacena)
**Professor:** Felipe Roncalli de Paula Carneiro

Pesquisa teórica sobre a transição das RNNs para os Transformers e LLMs (Parte 1) e execução local de um LLM com o Ollama, incluindo experimento de temperatura e script Python com streaming de tokens (Parte 2).

## Conteúdo do repositório

| Arquivo | Descrição |
| --- | --- |
| `Relatorio_Trabalho_Etapa_3.pdf` | Relatório completo: Parte 1 (fundamentação teórica) e Parte 2 (registro comentado dos testes) |
| `ollama_stream.py` | Script de inferência via API do Ollama com streaming, controle de hiperparâmetros e salvamento em .txt |
| `respostas/` | Saídas reais dos três cenários de temperatura (A: T=0.0, B: T=0.9, C: T=1.5) |

## Ambiente

- Windows (x64)
- [Ollama](https://ollama.com) com o modelo `qwen2.5-coder:7b` (7,6B parâmetros, GGUF quantizado em Q4_K_M)
- Python 3 com a biblioteca `requests`

## Como executar

1. Instale o Ollama e baixe o modelo:
   ```
   ollama pull qwen2.5-coder:7b
   ```
2. Instale a dependência do script:
   ```
   pip install requests
   ```
3. Com o Ollama em execução, rode o script:
   ```
   # Modo interativo: pergunta prompt, temperatura, top_p e num_predict
   python ollama_stream.py

   # Hiperparâmetros pela linha de comando
   python ollama_stream.py --temperature 0.8 --top_p 0.5 --num_predict 500

   # Experimento: roda os cenários A, B e C com o mesmo prompt
   python ollama_stream.py --experimento
   ```

Cada execução salva a resposta em `respostas/`, com um cabeçalho que registra modelo, hiperparâmetros, tokens gerados, duração e motivo da parada.

## Parâmetros do script

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--model` | `qwen2.5-coder:7b` | Modelo do Ollama |
| `--prompt` | Bubble Sort em C | Texto enviado ao modelo |
| `--temperature` | `0.2` | Temperatura (0.0 a 2.0) |
| `--top_p` | `0.9` | Nucleus sampling (0.0 a 1.0) |
| `--num_predict` | `300` | Máximo de tokens gerados |
| `--saida` | `respostas` | Pasta dos arquivos .txt |
| `--experimento` | – | Roda os três cenários de temperatura |

## Resultado principal

Os três cenários geraram código C válido. A temperatura alterou o estilo e a estrutura das respostas, mas mesmo T = 1.5 não causou colapso sintático, porque os filtros top_k (padrão 40 no Ollama) e top_p = 0.9 removem os tokens improváveis antes do sorteio. A análise completa está no relatório.
