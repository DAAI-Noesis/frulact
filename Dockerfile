FROM python:3.10-slim
 
# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including poppler-utils for PDF processing and tesseract for OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app
 
# Copy the application code
COPY . /app
 
# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# Expose port 80 for the application
EXPOSE 80
 
# Ensure compatibility with amd64 architecture (optional)
ARG TARGETPLATFORM=linux/amd64
 
# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:80"]