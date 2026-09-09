# Docker image banane ka recipe; build mein environment taiyar hota hai, CMD container start par website chalata hai.
# Lightweight Python 3.12 Linux image ko base bana rahe hain.
FROM python:3.12-slim

# Set environment variables
# Logs turant output hon aur Python .pyc cache files na banaye.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install required system dependencies (if any)
# Native Python dependencies compile karne ke liye build-essential install hota hai; package-list cache clean hoti hai.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
# uv ke published image se uv/uvx binaries copy karte hain.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user (UID 1000 is required for Hugging Face Spaces)
# Root ke bajay UID 1000 wala user banao; HF Space ke runtime layout ke saath compatible hai.
RUN useradd -m -u 1000 user

# Set working directory
# Aage ke relative COPY/RUN/CMD paths /app directory se resolve honge.
WORKDIR /app

# Change ownership of the app directory to the new user
# App directory non-root user ko writable banane ke liye ownership set karo.
RUN chown user:user /app

# Switch to the non-root user
# Aage ke build commands aur container process is user ke roop mein chalenge.
USER user

# Set PATH for the user so uv can be found if installed locally
# User-local executable directory ko PATH mein rakho, taaki wahan installed commands mil sakein.
ENV PATH="/home/user/.local/bin:$PATH"

# Copy only the dependency files first (for better layer caching)
# Dependency files pehle copy karne se sirf source code badalne par dependency layer reuse ho sakti hai.
COPY --chown=user:user pyproject.toml uv.lock ./

# Install dependencies using uv (creates a .venv inside /app)
# Locked dependencies image ki .venv mein install karo; --no-dev dev dependency group skip karta hai.
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
# Bacha hua app source image mein copy karo; .dockerignore wale files include nahi hote.
COPY --chown=user:user . .

# Expose the port the Flask app runs on
# Container ka intended web port 5000 document hota hai; actual port mapping host/Compose karta hai.
EXPOSE 5000

# Command to run the application using uv
# Container start par uv environment ke Python se app.py chalti hai.
CMD ["uv", "run", "python", "app.py"]
