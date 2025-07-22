# Base image
FROM docker_image

# Test apt
RUN ls -al /etc/apt
RUN apt-get moo

# Set the working directory inside the container
RUN mkdir -p /workspace
WORKDIR /workspace

# Copy the entire project structure to the container
COPY ./workspace .

# Install the Python dependencies
RUN pip3 install -r folder/requirements.txt

# Essential environment
ENV PYTHONPATH="/workspace"

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "folder/main.py"]
