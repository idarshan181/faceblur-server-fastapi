#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print colored output
print_color() {
    COLOR=$1
    MESSAGE=$2
    echo -e "\033[${COLOR}m${MESSAGE}\033[0m"
}

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    print_color "31" "Error: pip is not installed. Please install pip and try again."
    exit 1
fi

# Check if Poetry is installed, if not, install it
if ! command -v poetry &> /dev/null; then
    print_color "33" "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry installation
if ! command -v poetry &> /dev/null; then
    print_color "31" "Error: Poetry installation failed. Please install Poetry manually and try again."
    exit 1
fi

# Create a new Poetry environment if it doesn't exist
if [ ! -f "pyproject.toml" ]; then
    print_color "34" "Initializing a new Poetry project..."
    poetry init -n
fi

# Create and activate the Poetry virtual environment
print_color "34" "Creating and activating Poetry virtual environment..."
poetry env use python
poetry shell

# Install dependencies using Poetry
print_color "34" "Installing dependencies using Poetry..."
poetry install

# Create and populate .env file
print_color "34" "Creating and populating .env file..."
touch .env
echo "APP_ENV=dev" >> .env
echo "DATABASE_URL=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" >> .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env

print_color "33" "Please ensure that the following environment variables are set:"
print_color "33" "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB"

# Run database migrations
print_color "34" "Running database migrations..."
alembic upgrade head

print_color "32" "Setup complete! Your FastAPI project is now ready."
print_color "33" "You are now in the Poetry shell. To exit, type 'exit'."
print_color "33" "To activate the Poetry environment in the future, run: poetry shell"
print_color "33" "To start the application, run: uvicorn app.main:app --reload"