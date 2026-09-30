#!/usr/bin/env bash
set -euo pipefail

# Reliability-Oriented Adversarial Testing Framework command runner.
# Usage:
#   bash run_project.sh setup     # create venv and install dependencies
#   bash run_project.sh phase     # start the interactive phase selector
#   bash run_project.sh phase1    # run the quick Phase 1 helper
#   bash run_project.sh all       # setup, then start the phase selector

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR=".venv"

create_venv() {
  "$PYTHON_BIN" -m venv "$VENV_DIR"
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

install_requirements() {
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
}

create_env_file() {
  if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp ".env.example" ".env"
  fi
}

run_phase_selector() {
  python main.py
}

run_phase1_helper() {
  python auto_run.py
}

case "${1:-phase}" in
  setup)
    create_venv
    activate_venv
    install_requirements
    create_env_file
    ;;
  phase)
    activate_venv
    run_phase_selector
    ;;
  phase1)
    activate_venv
    run_phase1_helper
    ;;
  all)
    create_venv
    activate_venv
    install_requirements
    create_env_file
    run_phase_selector
    ;;
  *)
    echo "Unknown command: $1"
    echo "Available commands: setup, phase, phase1, all"
    exit 1
    ;;
esac
