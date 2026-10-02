import os
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "True").lower() == "true"
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")

if not MOCK_MODE and not NEBIUS_API_KEY:
    raise RuntimeError("NEBIUS_API_KEY is required when MOCK_MODE=False")
