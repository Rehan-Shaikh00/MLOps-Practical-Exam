"""End-to-end integration tests for the Wine MLOps pipeline.

Runs the full pipeline once (via ``sys.executable -m src.pipeline`` so the
active virtual environment is always used) and verifies:
1. Exit code 0 and clean stdout/stderr (no tracebacks).
2. Parquet schema, non-empty row count, and ``registry.json`` integrity.
3. SQLite MLflow tracking DB contains the experiment and logged runs.
4. The saved ``.pkl`` model loads with joblib and predicts valid labels.

Run from the repository root with::
    pytest -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REGISTRY_PATH: Path = PROJECT_ROOT / "feature_store" / "registry.json"
MLFLOW_DB_PATH: Path = PROJECT_ROOT / "mlflow.db"
BEST_MODEL_PATH: Path = PROJECT_ROOT / "models" / "best_wine_model.pkl"
TRACKING_URI: str = "sqlite:///mlflow.db"
EXPERIMENT_NAME: str = "Wine_Classification_Team_G2"
EXPECTED_N_CLASSES: int = 3


@pytest.fixture(scope="module")
def pipeline_result() -> subprocess.CompletedProcess[str]:
    """Run the full pipeline once and return the subprocess result."""
    return subprocess.run(
        [sys.executable, "-m", "src.pipeline"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def registry(pipeline_result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Load the feature-store registry produced by the pipeline run."""
    assert pipeline_result.returncode == 0, f"Pipeline failed: {pipeline_result.stderr}"
    assert REGISTRY_PATH.exists(), "Feature-store registry missing!"
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_pipeline_exit_code_and_clean_output(
    pipeline_result: subprocess.CompletedProcess[str],
) -> None:
    """Pipeline exits 0 with no traceback on stderr."""
    assert pipeline_result.returncode == 0, f"Pipeline failed: {pipeline_result.stderr}"
    assert "Traceback" not in pipeline_result.stderr
    assert "Traceback" not in pipeline_result.stdout


def test_registry_integrity(registry: dict[str, Any]) -> None:
    """Registry has all required keys and points at an existing Parquet file."""
    required_keys = {
        "feature_group",
        "version",
        "entity_key",
        "timestamp_key",
        "target",
        "features",
        "storage_path",
    }
    assert required_keys.issubset(registry.keys()), f"Registry missing keys: {required_keys - set(registry)}"
    assert registry["feature_group"] == "wine_classification_features"
    assert isinstance(registry["features"], list) and len(registry["features"]) > 0
    assert (PROJECT_ROOT / registry["storage_path"]).exists(), "Parquet feature file missing!"


def test_parquet_schema_and_row_count(registry: dict[str, Any]) -> None:
    """Parquet store is non-empty, null-free, and matches the registry schema."""
    df = pd.read_parquet(PROJECT_ROOT / registry["storage_path"])
    assert len(df) > 0, "Feature store has 0 rows!"
    assert int(df.isnull().sum().sum()) == 0, "Feature store contains nulls!"
    expected_cols = set(registry["features"] + [registry["entity_key"], registry["target"]])
    assert expected_cols.issubset(set(df.columns)), f"Schema mismatch: {expected_cols - set(df.columns)}"
    assert int(df[registry["target"]].nunique()) == EXPECTED_N_CLASSES


def test_mlflow_sqlite_tracking() -> None:
    """SQLite MLflow DB contains the expected experiment and >=3 logged runs."""
    assert MLFLOW_DB_PATH.exists(), "SQLite MLflow database not generated!"
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None, f"Experiment '{EXPERIMENT_NAME}' not found in mlflow.db!"
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3, f"Expected >=3 logged runs, found {len(runs)}"
    for run in runs:
        assert run.info.status == "FINISHED"
        assert "accuracy" in run.data.metrics
        assert run.data.tags.get("team") == "Group_G2"


def test_saved_model_predictions(registry: dict[str, Any]) -> None:
    """Saved best-model .pkl loads with joblib and predicts valid class labels."""
    assert BEST_MODEL_PATH.exists(), "Best model artifact missing!"
    model = joblib.load(BEST_MODEL_PATH)
    df = pd.read_parquet(PROJECT_ROOT / registry["storage_path"])
    sample = df[registry["features"]].iloc[:5]
    preds = model.predict(sample)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1, 2}), f"Invalid predictions: {set(preds)}"