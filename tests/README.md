# 🧪 Tests — Suite de Testes Automatizados

## Visão Geral

Esta pasta contém a suite de testes automatizados do projeto. Os testes validam que cada componente funciona isoladamente (testes unitários) e que integram bem (testes de integração).

**Filosofia:**
- ✅ Dados sintéticos com seed fixo (sem rede, sem APIs reais)
- ✅ Rápidos: unitários rodam em ~5s
- ✅ Seguros: integração marcada com `@pytest.mark.integration` (pulam sem `GROQ_API_KEY`)
- ✅ Observáveis: gate ≥ 60% cobertura — CI falha se cair abaixo

---

## 📁 Estrutura dos Arquivos

### **`conftest.py`** — Setup Compartilhado
Fixtures reutilizáveis em todos os testes:
- `synthetic_close_series` — 500 pontos IBOV com trend + ruído (seed=42)
- `synthetic_ohlc_df` — DataFrame OHLC com DatetimeIndex
- `fitted_scaler` — MinMaxScaler ajustado em dados sintéticos
- `tiny_lstm_data` — Tensores mínimos (X: 20x60x1, y: 20) para smoke tests

**Filosofia:** Nenhum dado real, nenhuma rede. Tudo reproduzível.

---

### **`test_agent.py`** — Agent ReAct (8 testes)
**O que testa:**
- Config carrega e tem valores válidos
- Paths são absolutos e existem
- 4 tools obrigatórias presentes (≥ 3 do guia)
- Calculator bloqueia comandos perigosos (`__import__`, `os.system`)
- Agente requer `GROQ_API_KEY` para iniciar

**Tipo:** 2 unitários + 6 integração

---

### **`test_train_lstm.py`** — Modelo LSTM (15 testes)
**O que testa:**
- `compute_metrics()` retorna exatamente 4 métricas (MAE, RMSE, MAPE, directional_accuracy)
- Previsão perfeita → erros próximos de zero
- `build_lstm()` cria arquitetura 3D correta
- `evaluate_on_holdout()` funciona
- Grid de hiperparâmetros é válido
- Shapes de input/output são corretos

**Tipo:** 15 unitários (usam `tiny_lstm_data` para speed)

---

### **`test_predict.py`** — Predição LSTM (6 testes)
**O que testa:**
- Modelo LSTM carrega sem erros
- Predição retorna shape correto
- Valores preditos em range [0, 1] (normalizado)
- Erro sem arquivo de modelo é claro
- Scaler carrega corretamente

**Tipo:** 6 unitários

---

### **`test_guardrails.py`** — Segurança (8 testes)
**O que testa:**
- Detecção de prompt injection (patterns suspeitos)
- Bloqueio de tokens perigosos
- Mensagens de erro são claras
- Guardrails rodam ANTES de chamar LLM
- Requisições malformadas são rejeitadas

**Tipo:** 8 unitários (segurança → determinístico)

---

### **`test_api.py`** — FastAPI (10 testes)
**O que testa:**
- `/health` retorna `status: "ok"` com domains listados
- `/predict` (LSTM) funciona com input válido
- `/agent/query` (agente) bloqueia injeção ANTES de chamar agente
- Status codes corretos (200, 400, 422)
- Responses têm formato esperado (JSON válido)

**Tipo:** 10 unitários (usa TestClient mock, sem rede)

---

### **`test_feature_engineering.py`** — Feature Eng (10 testes)
**O que testa:**
- Lags são calculados corretamente
- Volatilidade é normalizada
- NaN handling funciona
- Shapes preservados após transformação

**Tipo:** 10 unitários

---

### **`test_make_dataset.py`** — Coleta de Dados (5 testes)
**O que testa:**
- DataFrame passa no schema Pandera
- Não há gaps nas datas
- Close prices são positivos
- Duplicatas são removidas

**Tipo:** 5 unitários

---

### **`test_train_baseline.py`** — Baseline (3 testes)
**O que testa:**
- Baseline treina sem erro
- Predição é razoável (não NaN/inf)
- Métrica do baseline é consistente

**Tipo:** 3 unitários

---

### **`test_config.py`** — Configuração (5 testes)
**O que testa:**
- Config carrega de `config.yaml`
- Paths existem (absolute paths)
- Valores numéricos estão em ranges válidos
- Nenhuma chave obrigatória falta

**Tipo:** 5 unitários

---

### **`test_logging_config.py`** — Logging (2 testes)
**O que testa:**
- Logger inicializa sem erro
- Arquivo de log é criado em `logs/`

**Tipo:** 2 unitários

---

## 🚀 Como Rodar

### **Todos os Testes (Unitários + Integração)**
```bash
cd /Users/alan/Data\ Science\ Projects/datathon-techchallenge-fase-5
pytest
```

**Tempo:** ~30s (unitários rápidos + integração se `GROQ_API_KEY` estiver set)

---

### **Apenas Unitários** (Rápido, sem rede)
```bash
pytest -m "not integration"
```

**Tempo:** ~5s | **Usado em:** CI/CD, pré-commit

---

### **Apenas Integração** (Requer `GROQ_API_KEY`)
```bash
pytest -m "integration"
```

**Tempo:** ~20s | **Usado em:** Dev local, antes de push

---

### **Arquivo Específico**
```bash
pytest tests/test_agent.py -v
```

---

### **Função Específica**
```bash
pytest tests/test_agent.py::test_calculator_basic_arithmetic -v
```

---

### **Com Cobertura (HTML)**
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Resultado:** Relatório visual de linhas cobertas

---

### **Modo Verbose** (Vê cada teste)
```bash
pytest -v
```

---

### **Com Output Completo** (Print statements aparecem)
```bash
pytest -v -s
```

---

## 📊 Cobertura Esperada

| Módulo | Arquivo | Cobertura | Testes |
|--------|---------|-----------|--------|
| `agent_pipeline` | test_agent.py | ~85% | 8 |
| `ibov_pipeline` | test_train_lstm.py | ~75% | 15 |
| `ibov_pipeline` | test_predict.py | ~80% | 6 |
| `ibov_pipeline` | test_train_baseline.py | ~70% | 3 |
| `ibov_pipeline` | test_feature_engineering.py | ~80% | 10 |
| `ibov_pipeline` | test_make_dataset.py | ~75% | 5 |
| `serving` (API) | test_api.py | ~90% | 10 |
| `guardrails` | test_guardrails.py | ~100% | 8 |
| **TOTAL** | — | **~80%** | **~65** |

**Gate:** Mínimo 60% (definido em `pytest.ini`)

---

## 🔧 Convenções Locais

### **Marcadores Pytest**

```python
@pytest.mark.integration
def test_something_with_groq():
    """Testes que precisam GROQ_API_KEY ou rede pulam em CI."""
    pass
```

- **Unitários:** Sem marcador = rodarm em CI
- **Integração:** `@pytest.mark.integration` = rodam só localmente (se `GROQ_API_KEY` setado)

---

### **Fixtures do conftest.py**

Sempre usar fixtures ao invés de dados hardcoded:

```python
def test_something(synthetic_close_series):  # ✅ Usa fixture
    assert len(synthetic_close_series) == 500

def test_something_wrong():  # ❌ Hardcoded
    data = [1, 2, 3]  # Não reproduzível
```

---

### **Nomenclatura**

- **Arquivo:** `test_<module>.py` (ex: `test_agent.py`)
- **Função:** `test_<what>()` (ex: `test_calculator_basic_arithmetic()`)
- **Descrição:** Docstring explica o que valida

---

## 🔄 Integração com CI/CD

### **GitHub Actions** (`.github/workflows/ci.yml`)

```yaml
- name: Run tests with coverage gate
  run: pytest -m "not integration" --cov=src --cov-report=xml --cov-fail-under=60
```

**Comportamento:**
1. Roda apenas unitários (CI é rápido, sem rede)
2. Coleta cobertura
3. **Falha se cobertura < 60%** → PR não pode mergear

---

## 📝 Escrevendo Novos Testes

### **Checklist**

- [ ] Usa fixtures do `conftest.py` (dados sintéticos)
- [ ] Sem chamadas de rede (sem yfinance, Groq, ChromaDB remoto)
- [ ] Marcado com `@pytest.mark.integration` se precisar de rede
- [ ] Docstring clara explicando o que valida
- [ ] Nome descritivo: `test_<what_and_why>()`
- [ ] Sem dados hardcoded

### **Exemplo de Bom Teste**

```python
def test_compute_metrics_perfect_prediction(fitted_scaler):
    """Previsão = real → erros (mae, rmse, mape) próximos de zero."""
    y_true = np.linspace(0.1, 0.9, 50)
    m = compute_metrics(y_true, y_true.copy(), fitted_scaler)
    assert m["mae"] < 1e-6
    assert m["rmse"] < 1e-6
```

---

## 🐛 Troubleshooting

### **Erro: `GROQ_API_KEY not set`**
- **Unitários:** Pulam automaticamente (marcado com `@pytest.mark.integration`)
- **Solução:** `export GROQ_API_KEY="sk-..."` para rodar integração

### **Erro: `ModuleNotFoundError: No module named 'langchain'`**
- **Solução:** `pip install -r requirements.txt`

### **Cobertura abaixo de 60%**
- **Solução:** `pytest --cov=src --cov-report=html` → abrir `htmlcov/index.html` e ver o que falta

### **Um teste falha de repente**
- **Causa provável:** Dados sintéticos têm seed=42, deve ser determinístico
- **Solução:** Verificar se a função testada foi modificada

---

## 📚 Recursos Úteis

- **Pytest docs:** https://docs.pytest.org/
- **Fixtures:** https://docs.pytest.org/en/stable/how-to/fixtures.html
- **Markers:** https://docs.pytest.org/en/stable/how-to/mark.html

---

## 🎯 Resumo

| Quando | Comando | Tempo |
|--------|---------|-------|
| Pré-commit | `pytest -m "not integration"` | ~5s |
| Antes de push | `pytest -m "integration"` | ~20s |
| CI/CD (automático) | `pytest -m "not integration" --cov` | ~10s |
| Investigar cobertura | `pytest --cov=src --cov-report=html` | ~15s |
| Debugar teste | `pytest tests/test_x.py::test_y -v -s` | varies |

**Mantra:** Testes rápidos = feedback rápido = desenvolvimento mais rápido! 🚀
