# Use a base image with Jupyter Notebook pre-installed
FROM jupyter/base-notebook

# Switch to root user to perform system updates and installations
USER root

# Set the working directory inside the container
WORKDIR /app

# Copy your entire project directory into the container
COPY . /app

# Install build tools and development headers
RUN apt-get update && apt-get install -y gcc python3-dev

# Switch back to the non-root user used by the base image
USER $NB_UID

# Install any additional dependencies if needed
RUN pip install -r requirements.txt  # Replace with your requirements file if you have one

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

# Run Jupyter Notebook without any additional command-line arguments
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.notebook_dir=/app", "--NotebookApp.iopub_data_rate_limit=100000000", "--NotebookApp.token=", "--NotebookApp.password="]
