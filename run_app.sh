#!/usr/bin/env bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
"$DIR/env/bin/streamlit" run "$DIR/app.py" --server.headless false --server.port 8501
