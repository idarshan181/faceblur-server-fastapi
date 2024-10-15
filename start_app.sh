#!/bin/bash

# Adjust these paths to match your project structure
APP_MODULE="app.main:app"
VENV_PATH=".venv/bin/activate"

# Activate your virtual environment
source $VENV_PATH

# # Start Gunicorn with 4 Uvicorn workers
# gunicorn $APP_MODULE \
#     --workers 4 \
#     --worker-class uvicorn.workers.UvicornWorker \
#     --bind 127.0.0.1:8000 \
#     --bind 127.0.0.1:8001 \
#     --bind 127.0.0.1:8002 \
#     --bind 127.0.0.1:8003 \


# Start Gunicorn with 10 Uvicorn workers
gunicorn $APP_MODULE \
    --workers 10 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --reload