FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install required system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user (UID 1000 is required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Change ownership of the app directory to the new user
RUN chown user:user /app

# Switch to the non-root user
USER user

# Set PATH for the user so uv can be found if installed locally
ENV PATH="/home/user/.local/bin:$PATH"

# Copy only the dependency files first (for better layer caching)
COPY --chown=user:user pyproject.toml uv.lock ./

# Install dependencies using uv (creates a .venv inside /app)
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY --chown=user:user . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the application using uv
CMD ["uv", "run", "python", "app.py"]
