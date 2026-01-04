# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app


# Copy the rest of the app
COPY . /app



# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . /app

# Expose port
EXPOSE 5000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
