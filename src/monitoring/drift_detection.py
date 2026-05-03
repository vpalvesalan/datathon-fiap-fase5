"""Detecção de drift via PSI (Population Stability Index).

Implementa o GAP 06 do guia. PSI é a métrica mais robusta para séries
financeiras — detecta mudança de regime na distribuição da feature.

Convenções (do `cfg.retrain`):
- PSI > 0.10  → warning (drift moderado)
- PSI > 0.20  → trigger de retreinamento

Uso:
    from src.monitoring.drift_detection import compute_psi, detect_drift

    # Comparação simples
    psi = compute_psi(reference=train_close, current=last_30d_close)

    # Pipeline completo (logando no MLflow):
    detect_drift(reference=X_train.flatten(), current=X_recent.flatten())
"""
from __future__ import annotations

import logging

import numpy as np

from src.ibov_pipeline.config import cfg

logger = logging.getLogger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcula o PSI entre duas distribuições.

    PSI = Σ (p_i - q_i) * ln(p_i / q_i)

    Args:
        reference: Distribuição de referência (treino).
        current: Distribuição corrente (produção).
        n_bins: Quantis usados para binning.

    Returns:
        PSI ≥ 0. Convenção: <0.1 estável | 0.1–0.2 atenção | >0.2 drift.
    """
    ref = np.asarray(reference).flatten()
    cur = np.asarray(current).flatten()

    # Binning baseado em quantis da referência
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # cobre extremos

    eps = 1e-6  # evita log(0)
    p = np.histogram(ref, bins=edges)[0] / max(len(ref), 1) + eps
    q = np.histogram(cur, bins=edges)[0] / max(len(cur), 1) + eps

    psi = float(np.sum((p - q) * np.log(p / q)))
    return psi


def classify_drift(psi: float) -> str:
    """Categoriza o nível de drift segundo as conventions do projeto."""
    if psi < 0.10:
        return "stable"
    if psi < cfg.retrain.retrain_trigger_psi:
        return "warning"
    return "retrain_required"


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    log_to_mlflow: bool = True,
    output_path: str | None = None,
) -> dict[str, float | str]:
    """Pipeline completo: calcula PSI, classifica, loga no MLflow e/ou JSON.

    Args:
        reference: Distribuição de referência (treino).
        current: Distribuição atual (teste/produção).
        log_to_mlflow: Loga métrica e tags em um nested run.
        output_path: Se fornecido, persiste JSON do relatório nesse caminho.
            Necessário para integração com DVC (stage `drift_check`).

    Returns:
        {"psi": float, "level": str, "trigger_threshold": float, "timestamp": str}
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    psi = compute_psi(reference, current)
    level = classify_drift(psi)
    threshold = cfg.retrain.retrain_trigger_psi

    result = {
        "psi": psi,
        "level": level,
        "trigger_threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Drift detection: PSI=%.4f | level=%s | threshold=%.2f",
        psi, level, threshold,
    )

    if log_to_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)
            with mlflow.start_run(run_name="drift-check", nested=True):
                mlflow.set_tag("pipeline_stage", "monitoring")
                mlflow.log_metric("psi", psi)
                mlflow.set_tag("drift_level", level)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao logar drift no MLflow: %s", exc)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Relatório de drift gravado em %s", out)

    return result


if __name__ == "__main__":
    from pathlib import Path

    from src.ibov_pipeline.feature_engineering import load_artifacts
    from src.ibov_pipeline.logging_config import setup_logging

    setup_logging("drift_check")

    X_train, X_test, _, _, _ = load_artifacts()
    output_path = Path(cfg.data.processed_x_path).parent / "drift_report.json"

    result = detect_drift(
        reference=X_train.flatten(),
        current=X_test.flatten(),
        output_path=str(output_path),
    )
    print(f"\nPSI       : {result['psi']:.4f}")
    print(f"Nível     : {result['level']}")
    print(f"Threshold : {result['trigger_threshold']}")
    print(f"Relatório : {output_path}")
