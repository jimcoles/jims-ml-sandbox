# Use TensorFlow as a base image since it's already part of requirements
FROM tensorflow/tensorflow:2.18.0

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Jupyter/Notebook default port (if applicable)
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]