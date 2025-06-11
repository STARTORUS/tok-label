from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Label Studio configuration
LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL', 'http://example.com:8080')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

# File server configuration
FILE_SERVER_URL = os.getenv('FILE_SERVER_URL', 'http://example.com:8000')

# Extractor server configuration
EXTRACTOR_SERVER_URL = os.getenv('EXTRACTOR_SERVER_URL', 'http://example.com:8001')

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'example.com')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'example_password')
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

# PostgreSQL configuration
POSTGRE_HOST = os.getenv('POSTGRE_HOST', 'example.com')
POSTGRE_PORT = os.getenv('POSTGRE_PORT', '5432')
POSTGRE_USER = os.getenv('POSTGRE_USER', 'example_user')
POSTGRE_PASSWORD = os.getenv('POSTGRE_PASSWORD', 'example_password')
POSTGRE_DATABASE = os.getenv('POSTGRE_DATABASE', 'example_db')
