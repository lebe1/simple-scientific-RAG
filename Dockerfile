FROM python:3.12

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements.txt file
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code to the container
COPY ./app /code

# Set the default command to run the FastAPI app (will be overridden by docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
