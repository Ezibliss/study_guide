# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy local files to the container
COPY . /app

# Copy dependencies first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port your app runs on
ENV PORT 8080
EXPOSE 8080

# Run the app with Gunicorn
#CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]

# Start the Flask app
CMD ["python", "app.py"]
