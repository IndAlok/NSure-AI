# ---- Build Stage ----

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /usr/src/app

# Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install build-time dependencies
RUN pip install --upgrade pip

# Copy requirements and install project dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# Use a minimal base image for the final container to reduce size and attack surface.
FROM python:3.11-slim

# Create a non-root user for security best practices
RUN addgroup --system app && adduser --system --group app

# Set the working directory
WORKDIR /home/app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /usr/src/app/wheels /wheels

# Copy the application code
COPY . .

# Install the dependencies from the wheels without hitting the network again
RUN pip install --no-cache /wheels/*

# Change ownership of the app directory to the non-root user
RUN chown -R app:app /home/app

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
