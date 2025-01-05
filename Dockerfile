# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Turn off buffering for easier container logging
ENV PYTHONUNBUFFERED True

# Create a working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual Streamlit app
COPY app.py .

# Expose the default Streamlit port
EXPOSE 8501

# By default, run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
