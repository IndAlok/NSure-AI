
# We use 'slim' to keep the image size smaller and more efficient.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# Copy the file that lists our dependencies
COPY requirements.txt .

# Install dependencies
# --no-cache-dir makes the image smaller. --upgrade pip ensures we have the latest pip.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt


COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# We use --host 0.0.0.0 to make the app accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]