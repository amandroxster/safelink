FROM public.ecr.aws/lambda/python:3.11

# Copy code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Lambda entrypoint
CMD ["app.main.handler"]