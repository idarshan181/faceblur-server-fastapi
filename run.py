import os

if __name__ == "__main__":
    os.system("poetry run uvicorn --host=0.0.0.0 app.main:app --reload")
    # os.system("poetry run gunicorn app.main:app --workers 10 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --reload")
