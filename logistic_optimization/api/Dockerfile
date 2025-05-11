FROM python:3.13

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY logistic_optimization/api/requirements.txt .

COPY logistic_optimization/api .

COPY logistic_optimization/api/rf_model_pipeline_final.pkl .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]