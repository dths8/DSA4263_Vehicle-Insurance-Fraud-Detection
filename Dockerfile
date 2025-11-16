# Use a minimal Python 3.10 image for speed and security
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependencies file and install packages
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8501

# Define environment variable to run streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py"]