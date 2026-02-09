# Multi-stage Dockerfile for Universal Model Benchmark AI

# Stage 1: Base image with Python
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Rust build
FROM base as rust-build

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy Rust code
COPY rust /app/rust
WORKDIR /app/rust

# Build Rust library
RUN cargo build --release

# Stage 3: Python dependencies
FROM base as python-deps

# Copy requirements
COPY python/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Final image
FROM base as final

# Copy Python dependencies
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy Rust library
COPY --from=rust-build /app/rust/target/release/libbenchmark_core.so /app/libbenchmark_core.so

# Copy application code
COPY python /app/python
COPY rust/target/release/libbenchmark_core.so /app/python/core/libbenchmark_core.so

# Set working directory
WORKDIR /app/python

# Environment variables
ENV PYTHONPATH=/app/python
ENV RUST_BACKTRACE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "api.rest_api"]
