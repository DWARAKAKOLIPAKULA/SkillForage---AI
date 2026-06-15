# Use official Python 3.11 slim image — smaller than 3.13, more stable
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If requirements don't change, this layer is reused on rebuild
COPY requirements-core.txt .
COPY requirements-agents.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-core.txt
RUN pip install --no-cache-dir --no-deps -r requirements-agents.txt
# Copy rest of application code
COPY . .

# Expose port FastAPI runs on
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]