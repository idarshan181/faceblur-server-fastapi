import os

if __name__ == "__main__":
    os.system("poetry run uvicorn --host=127.0.0.1 --port=8000 app.main:app --reload")
# --workers 3
    # os.system("poetry run gunicorn app.main:app --workers 10 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 --reload")
