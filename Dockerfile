# Build recipe: prepare the container environment during build and start the website with CMD at runtime.
# Use a lightweight Python 3.12 Linux base image.
FROM python:3.12-slim

# Set environment variables.
# Flush logs immediately and disable Python .pyc cache generation.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install required system dependencies.
# Install build-essential for native dependencies and remove the package-list cache afterward.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management.
# Copy uv and uvx binaries from the published uv image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user with UID 1000 for Hugging Face Spaces.
# Use a UID 1000 account compatible with the HF Space runtime instead of running as root.
RUN useradd -m -u 1000 user

# Set the working directory.
# Resolve subsequent relative COPY, RUN, and CMD paths from /app.
WORKDIR /app

# Give the application directory to the new user.
# Set ownership so the non-root user can write to the application directory.
RUN chown user:user /app

# Switch to the non-root user.
# Run subsequent build commands and the container process as this user.
USER user

# Make user-local executables discoverable through PATH.
# Include the user-local binary directory so commands installed there can be found.
ENV PATH="/home/user/.local/bin:$PATH"

# Copy dependency files first for better layer caching.
# Reuse the dependency layer when only application source code has changed.
COPY --chown=user:user pyproject.toml uv.lock ./

# Install dependencies into /app/.venv using uv.
# Install locked dependencies; --no-dev skips the development dependency group.
RUN uv sync --frozen --no-dev

# Copy the remaining application source.
# Exclude files listed in .dockerignore from the copied build context.
COPY --chown=user:user . .

# Document the application port.
# EXPOSE documents port 5000; the host or Compose configures actual port forwarding.
EXPOSE 5000

# Run the application using uv.
# Start app.py with the Python interpreter from the uv environment when the container launches.
CMD ["uv", "run", "python", "app.py"]
