# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The command to run the dashboard. This can be overridden by docker-compose.
CMD ["streamlit", "run", "dashboard/dashboard.py"]
