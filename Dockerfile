FROM python:3.11-slim

# Create app user early
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --shell /bin/bash app

# Set environment variables for HuggingFace
ENV HF_HOME=/tmp/huggingface_cache
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache/transformers
ENV HF_HUB_CACHE=/tmp/huggingface_cache/hub
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface_cache/sentence_transformers

# Create cache directories with proper permissions
RUN mkdir -p /tmp/huggingface_cache/hub \
    /tmp/huggingface_cache/transformers \
    /tmp/huggingface_cache/sentence_transformers \
    && chmod -R 777 /tmp/huggingface_cache

# Set working directory
WORKDIR /home/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "localhost", "--port", "8000"]