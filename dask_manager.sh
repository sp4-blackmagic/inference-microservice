#!/usr/bin/env bash

# Script to check for dask, manage a Python virtual environment,
# install dependencies, and run dask scheduler or worker.
# Usage: ./dask_manager.sh [scheduler|worker]

# --- Configuration ---
SCHEDULER_ADDRESS="192.168.2.2:8786"
VENV_DIR=".venv" # Name of the virtual environment directory
REQUIREMENTS_FILE="requirements.txt" # Requirements file name

# Attempt to use python3 first, then python. Adjust if your system needs a specific command.
PYTHON_CMD_FOR_VENV="python3"
if ! command -v "$PYTHON_CMD_FOR_VENV" &> /dev/null; then
    PYTHON_CMD_FOR_VENV="python"
    if ! command -v "$PYTHON_CMD_FOR_VENV" &> /dev/null; then
        echo "Error: Neither 'python3' nor 'python' command found. Please install Python." >&2
        exit 1
    fi
fi
echo "Using '$PYTHON_CMD_FOR_VENV' to manage virtual environment."

# Paths within the virtual environment
PYTHON_IN_VENV="$VENV_DIR/bin/python3"
PIP_IN_VENV="$VENV_DIR/bin/pip3"
DASK_IN_VENV="$VENV_DIR/bin/dask"

# --- Helper Functions ---

# Function to set up the virtual environment and install dependencies
ensure_venv_and_dependencies() {
    echo "Ensuring Python virtual environment '$VENV_DIR' is set up..."

    # 1. Create Virtual Environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment '$VENV_DIR' not found. Creating it..."
        if "$PYTHON_CMD_FOR_VENV" -m venv "$VENV_DIR"; then
            echo "Virtual environment created successfully in '$VENV_DIR'."
        else
            echo "Error: Failed to create virtual environment in '$VENV_DIR'." >&2
            echo "Please check your Python installation and venv module." >&2
            return 1 # Failure
        fi
    else
        echo "Virtual environment '$VENV_DIR' already exists."
    fi

    # Ensure pip is available in venv
    if [ ! -x "$PIP_IN_VENV" ]; then
        echo "Error: pip not found in virtual environment at '$PIP_IN_VENV'." >&2
        echo "The virtual environment might be corrupted. Try removing it ('rm -rf $VENV_DIR') and running the script again." >&2
        return 1
    fi

    # 2. Install dependencies
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Found '$REQUIREMENTS_FILE'. Installing dependencies..."
        if "$PIP_IN_VENV" install -r "$REQUIREMENTS_FILE"; then
            echo "Dependencies installed successfully from '$REQUIREMENTS_FILE'."
        else
            echo "Error: Failed to install dependencies from '$REQUIREMENTS_FILE'." >&2
            return 1 # Failure
        fi
    else
        echo "Warning: '$REQUIREMENTS_FILE' not found." >&2
        echo "Attempting to install default packages: dask and distributed..."
        if "$PIP_IN_VENV" install dask distributed --upgrade; then
            echo "Default packages (dask, distributed) installed successfully."
        else
            echo "Error: Failed to install default packages (dask, distributed)." >&2
            return 1 # Failure
        fi
    fi

    # 3. Verify Dask executable is available in the venv
    if [ ! -x "$DASK_IN_VENV" ]; then
        echo "Error: Dask executable not found at '$DASK_IN_VENV' after installation attempts." >&2
        echo "Please check your '$REQUIREMENTS_FILE' or the installation process." >&2
        return 1 # Failure
    fi

    echo "Virtual environment and dependencies are ready."
    return 0 # Success
}

# Function to run the Dask scheduler
run_scheduler() {
    echo "Attempting to start Dask scheduler..."
    if ! ensure_venv_and_dependencies; then
        echo "Exiting due to environment setup issues." >&2
        exit 1
    fi

    echo "Starting Dask scheduler from venv at $SCHEDULER_ADDRESS..."
    # Using exec will replace the script process with the dask scheduler process.
    exec "$DASK_IN_VENV" scheduler --host "$SCHEDULER_ADDRESS"
}

# Function to run the Dask worker
run_worker() {
    echo "Attempting to start Dask worker..."
    if ! ensure_venv_and_dependencies; then
        echo "Exiting due to environment setup issues." >&2
        exit 1
    fi

    echo "Starting Dask worker from venv, connecting to scheduler at $SCHEDULER_ADDRESS..."
    # Using exec will replace the script process with the dask worker process.
    exec "$DASK_IN_VENV" worker "$SCHEDULER_ADDRESS"
}

# --- Main Script Logic ---

# Check if an argument (mode) is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [scheduler|worker]" >&2
    echo "Example: $0 scheduler" >&2
    echo "         $0 worker" >&2
    exit 1
fi

MODE="$1"

# Determine whether to run as scheduler or worker
case "$MODE" in
    scheduler)
        run_scheduler
        ;;
    worker)
        run_worker
        ;;
    *)
        echo "Error: Invalid mode '$MODE'." >&2
        echo "Please use 'scheduler' or 'worker'." >&2
        exit 1
        ;;
esac
