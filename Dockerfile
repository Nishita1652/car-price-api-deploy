# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# 5. Copy the model file and the application code into the container
COPY best_car_price_model.pkl .
COPY api_server.py .

# 6. Expose the port (this is just metadata, 10000 is Render's default)
EXPOSE 10000

# 7. Define the command to run the application
# Uses the $PORT environment variable, which Render provides.
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT