#!/usr/bin/env bash
# LightGBM and XGBoost wheels on macOS need the OpenMP runtime (libomp), normally installed
# with `brew install libomp`. Without Homebrew, point them at the copy scikit-learn already
# ships inside the virtual environment. Only touches .venv; run once after `uv sync`.
set -euo pipefail
SP=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
for lib in "$SP/lightgbm/lib/lib_lightgbm.dylib" "$SP/xgboost/lib/libxgboost.dylib"; do
  rel="@loader_path/$(python3 -c "import os,sys; print(os.path.relpath('$SP/sklearn/.dylibs', os.path.dirname('$lib')))")"
  otool -l "$lib" | grep -q "$rel" || install_name_tool -add_rpath "$rel" "$lib"
  codesign --force --sign - "$lib" 2>/dev/null   # re-sign after editing (required on Apple silicon)
done
echo "libomp linked for lightgbm and xgboost"
