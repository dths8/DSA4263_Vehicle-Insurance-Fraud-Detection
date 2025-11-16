# Use a minimal Python 3.10 image for speed and security
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt from the root to the working directory
COPY requirements.txt .

# Run pip install and clean up caches
RUN pip install --no-cache-dir -r requirements.txt

# Copy the models directory for model.pkl (assumed to still be in /models)
COPY models /app/models

# Copy the entire streamlit directory (which contains app.py and preprocessor.pkl)
COPY streamlit /app/streamlit

# Expose the port
EXPOSE 8501

# Define environment variable to run streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit/app.py"]