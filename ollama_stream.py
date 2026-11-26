"""
Trabalho 3ª Etapa - Inteligência Artificial (UNIPAC Barbacena)
Parte 2, item 4: Desafio Prático de Modificação de Código

Expansão do script base de streaming com o Ollama:
  - Controle dinâmico dos hiperparâmetros (temperature, top_p, num_predict)
    pela linha de comando ou, se nada for informado, por perguntas no terminal.
  - Exibição dos tokens em tempo real (stream=True).
  - Salvamento automático da resposta em um arquivo .txt, com cabeçalho
    registrando modelo, prompt e hiperparâmetros usados.
  - Modo --experimento: roda os três cenários de temperatura do item 2
    (A: T=0.0, B: T=0.9, C: T=1.5) com o mesmo prompt e salva cada um.

Exemplos de uso:
  python ollama_stream.py
  python ollama_stream.py --top_p 0.5 --num_predict 500
  python ollama_stream.py --temperature 0.8 --prompt "Explique recursão em C"
  python ollama_stream.py --experimento
"""

import argparse
import json
import os
from datetime import datetime

import requests

# URL do endpoint local do Ollama
URL = "http://localhost:11434/api/generate"

MODELO_PADRAO = "qwen2.5-coder:7b"  # Substitua pelo modelo baixado no seu Ollama
PROMPT_PADRAO = "Escreva um algoritmo de ordenação Bubble Sort em C bem documentado."

# Cenários do experimento de temperatura (Parte 2, item 2)
CENARIOS = [
    ("A", 0.0, "Determinístico"),
    ("B", 0.9, "Criatividade controlada"),
    ("C", 1.5, "O Caos"),
]


def ler_argumentos():
    """Lê os hiperparâmetros da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Inferência local no Ollama com streaming e salvamento em .txt"
    )
    parser.add_argument("--model", default=MODELO_PADRAO, help="Nome do modelo no Ollama")
    parser.add_argument("--prompt", default=None, help="Texto enviado ao modelo")
    parser.add_argument("--temperature", type=float, default=None, help="Temperatura (0.0 a 2.0)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-P / nucleus sampling (0.0 a 1.0)")
    parser.add_argument("--num_predict", type=int, default=None, help="Máximo de tokens gerados")
    parser.add_argument("--saida", default="respostas", help="Pasta onde os .txt serão salvos")
    parser.add_argument(
        "--experimento",
        action="store_true",
        help="Roda os cenários A (T=0.0), B (T=0.9) e C (T=1.5) com o mesmo prompt",
    )
    return parser.parse_args()


def perguntar(texto, padrao, tipo):
    """Pergunta um valor no terminal; Enter vazio mantém o padrão."""
    while True:
        resposta = input(f"{texto} [{padrao}]: ").strip()
        if resposta == "":
            return padrao
        try:
            return tipo(resposta.replace(",", "."))  # aceita 0,9 ou 0.9
        except ValueError:
            print("  Valor inválido, tente novamente.")


def validar(temperature, top_p, num_predict):
    """Confere se os hiperparâmetros estão em faixas aceitáveis."""
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature deve estar entre 0.0 e 2.0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p deve estar entre 0.0 (exclusivo) e 1.0")
    if num_predict <= 0:
        raise ValueError("num_predict deve ser um inteiro positivo")


def gerar(model, prompt, temperature, top_p, num_predict):
    """Envia o prompt ao Ollama, imprime os tokens em tempo real e devolve
    o texto completo e as estatísticas finais."""
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
        "stream": True,  # recebe token por token em tempo real
    }

    texto_completo = []  # acumula os tokens para salvar no arquivo depois
    estatisticas = {}

    # stream=True no requests mantém a conexão aberta e lê o corpo aos poucos
    with requests.post(URL, json=payload, stream=True, timeout=(5, 600)) as response:
        response.raise_for_status()

        # Cada linha é um objeto JSON com um pedaço (chunk) da resposta
        for chunk in response.iter_lines():
            if not chunk:
                continue

            data = json.loads(chunk.decode("utf-8"))

            if "error" in data:  # ex.: modelo não encontrado
                raise RuntimeError(data["error"])

            token_text = data.get("response", "")
            print(token_text, end="", flush=True)
            texto_completo.append(token_text)

            if data.get("done", False):
                estatisticas = {
                    "eval_count": data.get("eval_count", 0),  # tokens gerados
                    "prompt_eval_count": data.get("prompt_eval_count", 0),  # tokens do prompt
                    "total_duration_s": data.get("total_duration", 0) / 1e9,  # ns -> s
                    "done_reason": data.get("done_reason", ""),  # "stop" ou "length"
                }
                break

    return "".join(texto_completo), estatisticas


def salvar(pasta, model, prompt, temperature, top_p, num_predict, texto, estat, rotulo=""):
    """Salva a resposta em um .txt com cabeçalho dos parâmetros usados."""
    os.makedirs(pasta, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    sufixo = f"_cenario{rotulo}" if rotulo else ""
    nome = f"resposta_{carimbo}{sufixo}_T{temperature}_P{top_p}_N{num_predict}.txt"
    caminho = os.path.join(pasta, nome)

    cabecalho = (
        "=== Registro de Inferência - Ollama ===\n"
        f"Data/hora:          {datetime.now():%d/%m/%Y %H:%M:%S}\n"
        f"Modelo:             {model}\n"
        f"Temperature:        {temperature}\n"
        f"Top-P:              {top_p}\n"
        f"Num_predict:        {num_predict}\n"
        f"Tokens do prompt:   {estat.get('prompt_eval_count', 0)}\n"
        f"Tokens gerados:     {estat.get('eval_count', 0)}\n"
        f"Duração total:      {estat.get('total_duration_s', 0):.2f} s\n"
        f"Motivo da parada:   {estat.get('done_reason', '')}\n"
        f"Prompt:             {prompt}\n"
        "=======================================\n\n"
    )

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(cabecalho + texto + "\n")

    return caminho


def executar(args, prompt, temperature, top_p, num_predict, rotulo=""):
    """Roda uma geração completa: valida, gera, mostra estatísticas e salva."""
    validar(temperature, top_p, num_predict)

    titulo = f"Cenário {rotulo} | " if rotulo else ""
    print(f"\n--- {titulo}T={temperature} | top_p={top_p} | num_predict={num_predict} ---\n")

    texto, estat = gerar(args.model, prompt, temperature, top_p, num_predict)

    print("\n\n--- Geração Concluída ---")
    print(f"Total de tokens gerados: {estat.get('eval_count', 0)}")
    if estat.get("done_reason") == "length":
        print("Aviso: a resposta foi cortada pelo limite de num_predict.")

    caminho = salvar(args.saida, args.model, prompt, temperature, top_p, num_predict, texto, estat, rotulo)
    print(f"Resposta salva em: {caminho}")


def main():
    args = ler_argumentos()

    # Se nenhum hiperparâmetro veio pela linha de comando, pergunta no terminal
    modo_interativo = (
        args.prompt is None
        and args.temperature is None
        and args.top_p is None
        and args.num_predict is None
        and not args.experimento
    )

    if modo_interativo:
        print("Configure a geração (Enter mantém o valor padrão):")
        prompt = perguntar("Prompt", PROMPT_PADRAO, str)
        temperature = perguntar("Temperature", 0.2, float)
        top_p = perguntar("Top-P", 0.9, float)
        num_predict = perguntar("Num_predict (máx. de tokens)", 300, int)
    else:
        prompt = args.prompt or PROMPT_PADRAO
        temperature = args.temperature if args.temperature is not None else 0.2
        top_p = args.top_p if args.top_p is not None else 0.9
        num_predict = args.num_predict if args.num_predict is not None else 300

    try:
        if args.experimento:
            # Mesmo prompt, top_p e num_predict; só a temperatura muda
            for rotulo, temp, descricao in CENARIOS:
                print(f"\n##### Cenário {rotulo}: {descricao} #####")
                executar(args, prompt, temp, top_p, num_predict, rotulo)
        else:
            executar(args, prompt, temperature, top_p, num_predict)

    except requests.exceptions.ConnectionError:
        print(
            "Erro: Não foi possível conectar ao Ollama. "
            "Verifique se o serviço está rodando localmente (ollama serve)."
        )
    except ValueError as e:
        print(f"Parâmetro inválido: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


if __name__ == "__main__":
    main()
