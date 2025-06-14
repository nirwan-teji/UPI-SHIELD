# Use slim Python with essential dependencies
FROM python:3.10-slim

# System libs for OpenCV and core functionality
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and models
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make startup script executable
RUN chmod +x startup.sh

# Expose FastAPI default port
EXPOSE 8000

# Default command
CMD ["./startup.sh"]
