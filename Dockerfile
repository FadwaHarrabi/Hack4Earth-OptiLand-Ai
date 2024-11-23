# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install the required Python packages
RUN python -m pip install -r requirements.txt  

# Expose the port that Streamlit will run on
EXPOSE 8501  

# Run the Streamlit application when the container starts
CMD ["streamlit", "run", "app.py"]
