# Plano de Conformidade LGPD — Copiloto IBOV

> Lei nº 13.709/2018 (LGPD).

## 1. Escopo de dados pessoais

| Categoria | Coletado pelo sistema? | Justificativa |
|---|---|---|
| Dados de mercado público (preços, índices) | Sim | Não são dados pessoais (Art. 5º, X) |
| Dados pessoais sensíveis (saúde, religião, biometria) | Não | N/A |
| Dados pessoais não sensíveis (e-mail, telefone) | **Eventual** (em PDFs de RAG) | Removidos antes do output via Presidio |
| Logs de uso da API | Sim, sem identificador pessoal | IPs e queries são logados em `logs/`, sem cookies/login |

## 2. Bases legais aplicáveis (Art. 7º)

- **Inciso II — cumprimento de obrigação legal regulatória**: documentação
  exigida pelo Banco Central para sistemas de apoio à decisão financeira.
- **Inciso V — execução de contrato**: o sistema apoia analistas no
  cumprimento de tarefas contratadas pela empresa.
- **Inciso IX — interesse legítimo**: análise macroeconômica é interesse
  legítimo do controlador (gestor do projeto).

## 3. Princípios da LGPD aplicados

| Princípio (Art. 6º) | Implementação |
|---|---|
| Finalidade | Sistema serve apenas para análise IBOV — documentado no System Card |
| Adequação | RAG só ingere PDFs do domínio macro-financeiro, validados em revisão prévia |
| Necessidade | Apenas séries de fechamento + contexto macro são processados |
| Livre acesso | Endpoint `/health` é público; logs auditáveis em `logs/` |
| Qualidade dos dados | Schema validation pandera no `make_dataset.py` |
| Transparência | Model Card + System Card + OWASP Mapping disponíveis |
| Segurança | Guardrails input/output, MLflow tags de governança, Dockerfile não-root |
| Prevenção | Drift detection em `monitoring/drift_detection.py` dispara retraining |
| Não discriminação | LSTM atua sobre série univariada — não usa atributos protegidos |
| Responsabilização | `owner` taggeado em cada run MLflow; logs persistentes |

## 4. Direitos dos titulares (Art. 18)

O sistema **não armazena dados de titulares identificáveis**. Caso futuras
features adicionem login/cadastro:

- **Acesso, correção, exclusão**: implementar endpoint `/user/data` com auth.
- **Portabilidade**: export JSON dos logs do usuário.
- **Anonimização**: já implementada via Presidio no output.
- **Revogação de consentimento**: pseudonimização imediata + soft-delete em 30d.

## 5. Tratamento de PII em PDFs de RAG

**Problema**: relatórios financeiros podem conter nomes de analistas, CPFs
de signatários, e-mails de contato.

**Mitigações**:
1. **Revisão prévia** dos PDFs antes da ingestão (responsabilidade humana).
2. **OutputGuardrail** Presidio remove `PERSON`, `EMAIL_ADDRESS`,
   `PHONE_NUMBER`, `BR_CPF` no output do agente.
3. **Configuração** em `agent_config.yaml::guardrails.pii_entities` —
   ampliar lista conforme novos vetores forem identificados.

## 6. Retenção e descarte

- **Logs operacionais**: retenção de 90 dias em `logs/`. Após esse período,
  rotação automática.
- **MLflow runs**: retenção indefinida (auditoria), sem dados pessoais.
- **Vector store ChromaDB**: rebuilt sob demanda; não persiste queries dos
  usuários, apenas embeddings dos PDFs revisados.

## 7. Governança e responsável

| Papel | Responsabilidade |
|---|---|
| Controlador | grupo-XX (Datathon Fase 05) |
| Encarregado (DPO) | _preencher: pessoa designada_ |
| Operador | Groq (LLM API), Langfuse (telemetria) — contratos com cláusulas LGPD |

## 8. Plano de incidentes

Em caso de incidente de segurança envolvendo dados pessoais:

1. **Detecção**: alertas Prometheus + revisão Langfuse.
2. **Contenção**: desativar o endpoint via feature flag.
3. **Notificação**: ANPD em até 2 dias úteis (Art. 48).
4. **Comunicação aos titulares**: e-mail aos afetados em até 7 dias.
5. **Postmortem**: documentar em `docs/incidents/<data>.md`.
