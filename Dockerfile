# =============================================================================
# Stage 1: Build static Rust binary (cached separately from Python code)
# =============================================================================
FROM rust:1-slim as rust-builder

WORKDIR /build

# Copy the DES thermal simulation code for Rust build
COPY DES_thermal_simulation DES_thermal_simulation

# Fix case-sensitivity issue for Linux builds
WORKDIR /build/DES_thermal_simulation/src
RUN ln -sf Interpolator.rs interpolator.rs

# Build Rust code (this layer is cached unless DES code changes)
WORKDIR /build/DES_thermal_simulation
RUN cargo build --release

# Create version file (build timestamp since we don't have git history)
RUN date -u +"%Y%m%d-%H%M%S" > /build/DES_VERSION.txt

# =============================================================================
# Stage 2: Python application with pre-built Rust binary
# =============================================================================
FROM python:3.13-slim

ENV HOME=/root
WORKDIR /app

# Install graphics libraries, ffmpeg, and uv (no Rust/build-essential needed!)
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libgl1 \
    libgl1-mesa-dev \
    libglu1-mesa \
    libosmesa6 \
    libosmesa6-dev \
    mesa-utils \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libgles2 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python dependencies (cached unless requirements.txt changes)
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy pre-built Rust binary from builder stage
COPY --from=rust-builder /build/DES_thermal_simulation /app/DES_thermal_simulation

# Copy DES version file from builder stage
COPY --from=rust-builder /build/DES_VERSION.txt /app/DES_VERSION.txt

# Copy Python application code (changes here don't rebuild Rust!)
COPY . .

# Create projects directory
RUN mkdir -p /app/projects

# Environment variables
ENV QT_QPA_PLATFORM=offscreen
ENV ETS_TOOLKIT=null
ENV MPLBACKEND=Agg
ENV RUST_BACKTRACE=1

EXPOSE 8080

RUN chmod +x /app/start.sh

CMD ["./start.sh"]
