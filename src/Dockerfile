
FROM python:3.12-slim

# Install uv and system packages
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install system dependencies (including Rust/Cargo for packages like base2048)
RUN apt-get update && apt-get install -y \
    build-essential unixodbc-dev gcc g++ curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies from pyproject.toml (cache layer)
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the rest of the code
COPY . .

# Copy .env file into the image (for python-dotenv)
COPY .env .env

# Expose the port (default FastAPI/uvicorn port)
EXPOSE 8000

# Set environment variables (override in production)
ENV PORT=8000

# Start the FastAPI app
CMD ["uvicorn", "chat_app:app", "--host", "0.0.0.0", "--port", "8000"]
