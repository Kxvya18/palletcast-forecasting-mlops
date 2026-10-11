"""Copy the current production model out of MLflow into ./serving_model, so it can be
baked into the Docker image (the container then needs no access to the registry).

    uv run python -m palletcast.serving.export_model
"""

import json
import shutil

import mlflow
from mlflow import MlflowClient

from palletcast.models.common import ROOT

NAME = "outbound_forecaster"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    version = MlflowClient().get_model_version_by_alias(NAME, "production")
    out = ROOT / "serving_model"
    if out.exists():
        shutil.rmtree(out)
    mlflow.artifacts.download_artifacts(f"models:/{NAME}@production", dst_path=str(out))
    (out / "version.json").write_text(
        json.dumps({"name": NAME, "version": version.version, "tags": version.tags}, indent=1)
    )
    print(f"exported {NAME} v{version.version} to {out}")


if __name__ == "__main__":
    main()
