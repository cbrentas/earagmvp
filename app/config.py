import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

DATABASE_URL = os.environ['DATABASE_URL']
API_KEY = os.getenv('NVIDIA_API_KEY', '')
BASE_URL = os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1').rstrip('/')
CHAT_MODEL = os.getenv('NVIDIA_CHAT_MODEL', 'openai/gpt-oss-20b')
EMBED_MODEL = os.getenv('NVIDIA_EMBED_MODEL', 'nvidia/nemotron-3-embed-1b')
EMBED_DIMS = 2048
UPLOAD_DIR = ROOT / 'data' / 'uploads'
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 80
MAX_CHUNKS = 300
