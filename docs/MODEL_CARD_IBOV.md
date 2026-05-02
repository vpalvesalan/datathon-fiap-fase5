# Model Card — LSTM Preditor IBOV

> Padrão: Mitchell et al. (2019) — Model Cards for Model Reporting (FAccT)

## Detalhes do Modelo

| Campo | Valor |
|---|---|
| **Nome** | `ibov-lstm-predictor` |
| **Versão** | 0.1.0 |
| **Tipo** | Regressão (séries temporais univariada) |
| **Framework** | TensorFlow / Keras |
| **Owner** | grupo-XX (Datathon Fase 05) |
| **Risk Level** | medium |
| **Registry** | MLflow Model Registry, stage `Staging` → `Production` após approval |

## Uso pretendido

**Caso de uso primário**: filtro de viés direcional para um agente analítico
(`Copiloto IBOV`). O modelo é exposto pelo `IBOVForecastTool` e consumido em
contexto, junto com RAG macro e cotações live.

**Não usar para**: trading autônomo, geração de sinais de entrada/saída sem
supervisão humana, decisão de hedging em tempo real.

## Dados de Treino

- **Fonte**: yfinance, ticker `^BVSP`
- **Período**: configurável em `model_config.yaml`, default `2019-01-01 → hoje`
- **Volume**: ~1700 pregões (varia conforme `start_date`)
- **Validação de schema**: pandera (`Close > 0`, sem nulls, índice DatetimeTZ UTC)
- **Splits cronológicos**: 80% treino, 20% teste, 10% holdout imutável
- **Hash do dataset**: registrado como tag MLflow (`dataset_hash`) por run

## Métricas de Avaliação

Médias do champion atual (preencher após `train_lstm.run()`):

| Métrica | Test set | Holdout imutável |
|---|---|---|
| MAE (pontos) | _preencher_ | _preencher_ |
| RMSE (pontos) | _preencher_ | _preencher_ |
| MAPE | _preencher_ | _preencher_ |
| Acurácia direcional | _preencher_ | _preencher_ |

> Os números acima são gerados automaticamente — ver
> `notebooks/02_lstm_training_report.ipynb`.

## Limitações conhecidas

1. **Acurácia direcional ≈ 50%**. O modelo univariado, ao otimizar MSE,
   converge para "amanhã ≈ hoje" (estratégia ótima em mercado eficiente).
   Bom para magnitude (RMSE baixo), ruim para direção.
2. **Sem features exógenas** no v0 (sem Dólar, S&P 500, Selic).
3. **Sem regime switching** — não distingue mercado lateral vs. tendencial.
4. **Sensível a quebras estruturais** (COVID, eleições) — drift detectado
   via PSI em `src/monitoring/drift_detection.py`, threshold 0.20 dispara
   retraining.

## Considerações Éticas

- Não armazena dados pessoais — opera só sobre série pública (yfinance).
- Decisões financeiras tomadas com base na previsão do modelo são de
  responsabilidade do usuário; documentado explicitamente no System Card.

## Manutenção

- **Re-treino**: `python -m src.ibov_pipeline.retrain` (champion-challenger,
  promove só se `delta_mape ≥ 0.5%` no holdout).
- **Monitoramento**: PSI em features e predições, dispara alerta em `> 0.20`.
- **Versionamento**: tags MLflow (`git_sha`, `dataset_hash`, `model_version`).
