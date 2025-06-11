# Use official Python base image
FROM gitlab.startorus.org:5050/scientific-computing/tokamak-simulation-framework:latest

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DATA_DIR=/data

# Create and set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY . .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# Create data directory and expose port
RUN mkdir -p ${DATA_DIR}
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
