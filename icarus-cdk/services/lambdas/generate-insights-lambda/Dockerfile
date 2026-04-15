# Start with AWS's official Python image for Lambda
# This image already has everything Lambda needs
FROM public.ecr.aws/lambda/python:3.13

# Set Working Directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy your requirements file into the image
# ${LAMBDA_TASK_ROOT} is a special folder inside the container
COPY requirements.txt .

# Install all Python packages listed in requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Copy your Python code into the image
COPY . .

# Tell Lambda which function to run
# Format: filename.function_name
CMD ["generate_insights_lambda_v2.lambda_handler"]
