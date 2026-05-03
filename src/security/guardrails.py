"""Guardrails de input e output do agente LLM.

- **InputGuardrail** (puro Python/regex): detecta prompt injection,
  bloqueia inputs longos demais, normaliza whitespace.
- **OutputGuardrail** (Presidio): remove PII (CPF, e-mail, telefone, nome)
  do output do LLM antes de devolver ao cliente.

Mapeamento OWASP Top 10 LLM (2025):
- LLM01 Prompt Injection      → InputGuardrail.validate
- LLM02 Insecure Output       → OutputGuardrail.sanitize
- LLM06 Sensitive Info Leak   → OutputGuardrail.sanitize (Presidio PII)
"""
from __future__ import annotations

import logging
import re

from src.agent_pipeline.config import agent_cfg

logger = logging.getLogger(__name__)

# Lista de termos financeiros que o NER PT confunde com nomes
FINANCIAL_ALLOW_LIST = [
    "Brent", "S&P", "S&P 500", "Wall Street", "Nasdaq",
    "Bovespa", "Ibovespa", "Selic", "Copom", "Fed", "FOMC",
    "USD", "BRL", "EUR", "WTI",
    # Termos do framework LangChain que não são PII
    "Agent", "AgentExecutor", "Action", "Thought", "Observation",
]

# =============================================================================
# Input Guardrail — regex puro, sem deps pesadas
# =============================================================================

class InputGuardrail:
    """Valida e sanitiza input do usuário ANTES de enviar ao LLM."""

    INJECTION_PATTERNS: list[str] = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?previous\s+messages",
        r"forget\s+(everything|all|your\s+instructions)",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(?:a|an|the)\s+",
        r"system\s*:\s*",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"###\s*system",
    ]

    def __init__(self, max_length: int | None = None):
        self.max_length = max_length or agent_cfg.guardrails.max_input_length
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida o input.

        Returns:
            Tupla (is_valid, reason). reason='OK' se válido.
        """
        if not user_input or not user_input.strip():
            return False, "Input bloqueado: vazio."

        if len(user_input) > self.max_length:
            logger.warning(
                "Input bloqueado: %d chars excede max %d.",
                len(user_input), self.max_length,
            )
            return False, (
                f"Input bloqueado: excede {self.max_length} caracteres."
            )

        for pattern in self._compiled:
            if pattern.search(user_input):
                logger.warning(
                    "Prompt injection detectado: %s", user_input[:200],
                )
                return False, (
                    "Input bloqueado: padrão de prompt injection detectado."
                )

        return True, "OK"


# =============================================================================
# Output Guardrail — Presidio (lazy load, opcional)
# =============================================================================

class OutputGuardrail:
    """Remove PII do output do LLM via Presidio.

    Inicialização do Presidio é lazy — só carrega quando .sanitize() é chamado
    pela primeira vez. Isso evita custo de import em testes que não usam PII.
    """

    def __init__(
        self,
        language: str | None = None,
        entities: list[str] | None = None,
    ):
        self.language = language or agent_cfg.guardrails.language
        self.entities = entities or agent_cfg.guardrails.pii_entities
        self._analyzer = None
        self._anonymizer = None

    def _ensure_loaded(self) -> None:
        if self._analyzer is None:
            import logging
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            # 0. Silencia os logs  do Presidio (mostra apenas Erros graves)
            logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

            # 1. Configura o Presidio para usar o dicionário spaCy em Português
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "pt", "model_name": "pt_core_news_sm"}]
            }

            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()

            # 2. Injeta o motor em português no AnalyzerEngine
            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["pt"]
            )
            
            # 3. Ensina o Presidio a achar CPFs no formato brasileiro
            cpf_pattern = Pattern(
                name="cpf_regex", 
                regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", 
                score=0.8
            )
            cpf_recognizer = PatternRecognizer(
                supported_entity="BR_CPF", 
                patterns=[cpf_pattern], 
                supported_language="pt"
            )
            self._analyzer.registry.add_recognizer(cpf_recognizer)

            self._anonymizer = AnonymizerEngine()

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do texto. Retorna o texto original se não houver PII."""
        if not llm_output:
            return llm_output

        self._ensure_loaded()
        results = self._analyzer.analyze(
            text=llm_output,
            language=self.language,
            entities=self.entities,
            allow_list=FINANCIAL_ALLOW_LIST, 
            score_threshold=0.85
        )

        if not results:
            return llm_output

        logger.warning(
            "PII detectado no output: %d entidades (%s)",
            len(results),
            ", ".join(sorted({r.entity_type for r in results})),
        )
        anonymized = self._anonymizer.anonymize(
            text=llm_output,
            analyzer_results=results,
        )
        return anonymized.text
