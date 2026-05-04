"""Agente ReAct LangChain + Groq (Llama 3.3) — orquestrador do Copiloto IBOV.

Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting in
Language Models. https://arxiv.org/abs/2210.03629

O agente recebe perguntas em linguagem natural sobre o IBOV e decide qual
tool acionar (forecast, RAG, calculator, market_context). Toda a sequência
Thought → Action → Observation é registrada via Langfuse para auditoria.
"""

from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Apenas Erros (nível 2), ocultando Infos (0) e Warnings (1)

import logging
from dotenv import load_dotenv

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)
load_dotenv()

SYSTEM_PROMPT = """Você é o **Copiloto Analítico do IBOV** — um assistente \
que apoia gestores e analistas a entender o mercado brasileiro cruzando \
sinais quantitativos (LSTM) com contexto macroeconômico (relatórios Focus, \
Copom, análises de corretoras).

REGRAS FUNDAMENTAIS:
1. Sempre que o usuário perguntar sobre direção/preço futuro do IBOV, \
   USE a tool ibov_forecast para obter a previsão do modelo.
2. Sempre que o usuário perguntar "por quê?", contexto macro, expectativas \
   de Selic/IPCA/PIB, USE macro_rag para citar fontes.
3. Para qualquer cálculo numérico (variações %, somas), USE calculator — \
   NUNCA faça matemática no raciocínio.
4. Para cotações ao vivo de USD/BRL, S&P 500, Brent, USE market_context.
5. Seja honesto sobre as limitações do modelo: a acurácia direcional do \
   LSTM univariado é ~50%. O modelo é um **filtro de viés**, não uma \
   bola de cristal.
6. CITAÇÃO OBRIGATÓRIA — sempre que usar macro_rag, sua Final Answer DEVE \
   terminar com um bloco "Fontes:" listando cada PDF e página citados.
   FORMATO EXATO:

   Fontes:
   - mornicall_xp_30_04_2026.pdf, pg. 2
   - Copom276-not20260128276.pdf, pg. 1

   Respostas sem esse bloco são consideradas INCOMPLETAS e violam o
   contrato do sistema. Use APENAS as fontes que aparecem nos
   "[Fonte: ... — pg. ...]" das observações de macro_rag — nunca invente.
7. "Se você usar a ferramenta macro_rag e a informação necessária \
    não estiver no texto retornado, NÃO tente usar a ferramenta novamente.\
    Pare imediatamente e responda: 'Desculpe, não encontrei essa informação \
    nos documentos disponíveis'."

IMPORTANTE: ao emitir "Action:", escreva o nome da ferramenta EXATAMENTE \
como listado em [{tool_names}], sem crases, aspas ou outros caracteres.
"""


REACT_TEMPLATE = """Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato:

Question: a pergunta original
Thought: pensar sobre o que fazer
Action: uma das ferramentas: [{tool_names}]
Action Input: o input para a ferramenta
Observation: resultado da ferramenta
... (Thought/Action/Observation podem repetir N vezes)
Thought: agora sei a resposta final
Final Answer: resposta consolidada para o usuário (em português). \
Se você usou macro_rag em qualquer passo acima, INCLUA OBRIGATORIAMENTE \
o bloco "Fontes:" conforme regra 6.

INSTRUÇÕES DO SISTEMA:
""" + SYSTEM_PROMPT + """

Question: {input}
{agent_scratchpad}
"""


def _build_llm():
    """Cria o cliente Groq (lazy import para não pesar test collect)."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY não definido. Copie .env.example para .env e "
            "obtenha a chave em https://console.groq.com/keys."
        )

    return ChatGroq(
        model=agent_cfg.llm.model_name,
        temperature=agent_cfg.llm.temperature,
        max_tokens=agent_cfg.llm.max_tokens,
        api_key=api_key,
    )


def create_copiloto_agent(tools: list | None = None):
    """Cria e retorna um AgentExecutor pronto para receber perguntas.

    Args:
        tools: Lista de tools. Se None, usa o conjunto padrão (4 tools).

    Returns:
        AgentExecutor LangChain.

    Raises:
        ValueError: Se houver < 3 tools (regra do datathon).
        RuntimeError: Se GROQ_API_KEY não estiver definida.
    """
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_classic.prompts import PromptTemplate

    from src.agent_pipeline.tools import get_default_tools

    if tools is None:
        tools = get_default_tools()

    if len(tools) < 3:
        raise ValueError(
            f"O guia do Datathon exige ≥ 3 tools. Recebido: {len(tools)}."
        )

    llm = _build_llm()
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=agent_cfg.llm.max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

def query(question: str) -> dict:
    """Atalho conveniente: cria agente, valida com guardrails, faz a pergunta, 
    registra telemetria e retorna resposta + traços.

    Returns:
        dict com chaves: answer, intermediate_steps, model, blocked, block_reason
    """
    from src.security.guardrails import InputGuardrail, OutputGuardrail
    from src.monitoring.telemetry import get_langfuse_callback

    # 1) Input Guardrail
    input_guard = InputGuardrail()
    ok, reason = input_guard.validate(question)
    if not ok:
        logger.warning(f"Query bloqueada pelo Guardrail: {reason}")
        return {
            "answer": f"⚠️ Consulta bloqueada: {reason}",
            "intermediate_steps": [],
            "model": "blocked",
            "blocked": True,
            "block_reason": reason,
        }

    # 2) Configuração de Telemetria (Langfuse)
    callbacks = []
    lf_handler = get_langfuse_callback()
    if lf_handler:
        callbacks.append(lf_handler)

    # 3) Criação e Invocação do Agente
    executor = create_copiloto_agent()
    try:
        result = executor.invoke(
            {"input": question},
            config={"callbacks": callbacks} if callbacks else None
        )
        raw_answer = result.get("output", "")
        steps = result.get("intermediate_steps", [])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro interno na execução do agente.")
        return {
            "answer": f"❌ Erro ao consultar o agente: {exc}",
            "intermediate_steps": [],
            "model": agent_cfg.llm.model_name,
            "blocked": False,
            "block_reason": "Error",
        }

    # 4) Output Guardrail (Sanitização de PII)
    output_guard = OutputGuardrail()
    try:
        clean_answer = output_guard.sanitize(raw_answer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Output guardrail falhou (%s) — retornando bruto.", exc)
        clean_answer = raw_answer

    return {
        "answer": clean_answer,
        "intermediate_steps": steps,
        "model": agent_cfg.llm.model_name,
        "blocked": False,
        "block_reason": "OK",
    }

if __name__ == "__main__":
    import sys

    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("react_agent")
    pergunta = " ".join(sys.argv[1:]) or "Qual a previsão do IBOV para amanhã e qual o contexto macro?"
    print(f"Pergunta: {pergunta}\n")
    out = query(pergunta)
    print(f"\n=== Resposta ===\n{out['answer']}")