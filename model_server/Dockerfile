# model_server/Dockerfile
# Use python 3.10 slim as base 
FROM python:3.10-slim

# Set workind dir
WORKDIR /app

# System dependencies for RDKit and PyTorch Geometric
RUN apt-get update && apt-get install -y \
    gcc g++ libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy XSmiles source (entire project, since model_server import src/)
COPY . /app/

# Install Python dependencies
# Note: heavy package first for better Docker layer caching
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \ 
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \   
    torch-scatter torch-geometric \ 
    -f https://data.org/whl/torch-2.4.0+cpu.html

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic>=2.0

# Export port (Cloud Run uses 8080)
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "model_server.main:app", "--host", "0.0.0.0", "--port", "8080"]


