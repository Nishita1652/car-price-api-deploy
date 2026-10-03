# 1. Use an official Python runtime as a parent image
# Using python:3.10-slim for a good balance of features and size
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
# --no-cache-dir: Reduces image size by not storing the download cache
# --upgrade pip: Ensures we have the latest pip
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# 5. Copy the model file and the application code into the container
# !!! IMPORTANT !!!
# Make sure your saved model file is in the same directory as this
# Dockerfile and is named "best_car_price_model.pkl" (as defined in api_server.py)
COPY best_car_price_model.pkl .
COPY api_server.py .

# 6. Expose the port Google Cloud Run expects
# Cloud Run listens on port 8080 by default
EXPOSE 10000

# 7. Define the command to run the application
# Use uvicorn to run the FastAPI app
# --host 0.0.0.0 makes the app accessible from outside the container
# --port 8080 matches the EXPOSE command and Cloud Run's default
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT
