"""Interface Gradio do Copiloto IBOV — montada em /chat sobre o FastAPI.

Reusa os mesmos guardrails, agente e callbacks Langfuse do endpoint
/agent/query. A UI mostra duas colunas:
- Esquerda: campo de pergunta + resposta final do agente.
- Direita: linha de raciocínio (intermediate_steps) com cada Action/Observation.

Lazy imports para evitar custo no startup do FastAPI quando o usuário
nunca acessa /chat.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _format_intermediate_steps(steps: list) -> str:
    """Renderiza intermediate_steps do agente em markdown legível e dinâmico."""
    if not steps:
        return "_(nenhum passo intermediário registrado)_"

    blocks = []
    for i, step in enumerate(steps, 1):
        try:
            # Desempacota com segurança (se a observação ainda não existir, será None)
            action = step[0]
            observation = step[1] if len(step) > 1 else None
            
            tool = getattr(action, "tool", "?")
            tool_input = getattr(action, "tool_input", "")
            log_text = (getattr(action, "log", "") or "").strip()

            # Formatação melhorada usando Emojis e negrito para destacar a ação
            block = [f"### 🛠️ Passo {i}: Executando `{tool}`"]
            if tool_input:
                block.append(f"**Input da Ferramenta:** `{tool_input}`")
            
            # Se a observação for None, significa que a ferramenta ainda está rodando!
            if observation is None:
                block.append("\n> ⏳ *Aguardando resultado da ferramenta...*")
            else:
                obs_str = str(observation).strip()
                if len(obs_str) > 800:
                    obs_str = obs_str[:800] + "… [truncado]"
                block.append(f"**Resultado Obtido:**\n```text\n{obs_str}\n```")
                
            blocks.append("\n\n".join(block))
        except Exception as exc:  # noqa: BLE001
            blocks.append(f"### ⚠️ Passo {i}\n```\n{step}\n```\n_(erro ao parsear: {exc})_")

    return "\n\n---\n\n".join(blocks)


def _on_submit(question: str): # Note que removemos a tipagem de retorno -> tuple, pois agora é um Generator
    """Handler do botão de submit. Usa yield para atualizar a UI em tempo real."""
    if not question or not question.strip():
        yield "_Digite uma pergunta antes de enviar._", ""
        return

    from src.serving import app as serving_app

    input_guard = serving_app._get_input_guard()
    output_guard = serving_app._get_output_guard()
    agent_executor = serving_app._get_agent_executor()
    langfuse_callbacks = serving_app._get_langfuse_callbacks()

    # 1) Input guardrail
    ok, reason = input_guard.validate(question)
    if not ok:
        yield f"⚠️ **Bloqueado pelo guardrail de entrada:** {reason}", ""
        return

    # Avisa a UI que começou a pensar
    yield "⏳ **Analisando a pergunta e decidindo ferramentas...**", "🔄 *Iniciando linha de raciocínio...*"

    steps_history = []
    final_answer = ""

    # 2) Agente (Modo Streaming)
    try:
        # Usamos .stream() em vez de .invoke()
        for chunk in agent_executor.stream(
            {"input": question},
            config={"callbacks": langfuse_callbacks} if langfuse_callbacks else None,
        ):
            # O LangChain avisa quando decide USAR uma ferramenta
            if "actions" in chunk:
                for action in chunk["actions"]:
                    # Adicionamos a ação, mas a observação ainda é None
                    steps_history.append((action, None))
                    yield "⏳ **Executando ferramentas...**", _format_intermediate_steps(steps_history)

            # O LangChain avisa quando a ferramenta TERMINOU e devolveu resultado
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    # Atualizamos o último passo com a observação real
                    if steps_history:
                        steps_history[-1] = (step.action, step.observation)
                    yield "⏳ **Processando resultados e elaborando resposta...**", _format_intermediate_steps(steps_history)

            # O LangChain avisa quando chegou na RESPOSTA FINAL
            elif "output" in chunk:
                final_answer = chunk["output"]

    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro no agente (Gradio)")
        yield f"❌ **Erro ao consultar o agente:** `{exc}`", _format_intermediate_steps(steps_history)
        return

    # 3) Output guardrail (PII) na resposta final
    try:
        clean_answer = output_guard.sanitize(final_answer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Output guardrail falhou (%s) — retornando bruto.", exc)
        clean_answer = final_answer

    # Retorno final definitivo
    yield clean_answer or "_(resposta vazia)_", _format_intermediate_steps(steps_history)

def build_gradio_blocks():
    """Constrói o gr.Blocks. Mantém o import dentro da função para que
    o módulo possa ser importado sem gradio instalado (testes)."""
    import gradio as gr

    with gr.Blocks(
        title="Copiloto IBOV — Chat",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🤖 Copiloto Analítico do IBOV - Datathon FIAP fase 5.

        Pergunte sobre **previsão do índice**, **contexto macroeconômico**
        (Focus, Copom) ou peça **cálculos**. O agente decide quais
        ferramentas acionar (LSTM, RAG, calculadora, cotações live) e
        cita as fontes consultadas.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                question_in = gr.Textbox(
                    label="Sua pergunta",
                    placeholder=(
                        "Ex: Qual a previsão do IBOV para amanhã? "
                        "Qual a expectativa do Focus para a Selic em 2026?"
                    ),
                    lines=3,
                    autofocus=True,
                )
                with gr.Row():
                    submit_btn = gr.Button("Perguntar", variant="primary", scale=2)
                    clear_btn = gr.Button("Limpar", scale=1)

                gr.Markdown("### Resposta")
                answer_out = gr.Markdown()

            with gr.Column(scale=1):
                gr.Markdown("### Linha de raciocínio")
                steps_out = gr.Markdown(
                    value="_(faça uma pergunta para ver os passos do agente)_",
                )

        gr.Examples(
            examples=[
                "Qual a previsão do IBOV para amanhã?",
                "Qual a expectativa do Focus para a Selic ao final de 2026?",
                "Como o dólar está hoje? E o S&P 500?",
                "Calcule a variação percentual entre 128.500 e 130.300 pontos.",
            ],
            inputs=question_in,
        )

        gr.Markdown("""
        ---
        **Endpoints REST equivalentes:** `POST /agent/query` (JSON), `POST /predict` (LSTM direto).
        Documentação interativa: [/docs](/docs). Telemetria: traces no MLflow UI.
        """)

        # Wire-up
        submit_btn.click(
            fn=_on_submit,
            inputs=[question_in],
            outputs=[answer_out, steps_out],
        )
        question_in.submit(
            fn=_on_submit,
            inputs=[question_in],
            outputs=[answer_out, steps_out],
        )
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[question_in, answer_out, steps_out],
        )

    return demo
