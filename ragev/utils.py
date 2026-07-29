"""
RAGEv API Utilities.

Provides rate-limited interaction with the RAG platform API.
"""
import requests
import threading
import time

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ragev.data_models import VastConfig
from ragev.config import settings


def RateLimited(max_per_minute: int):
    """Decorator that limits function calls to a maximum per minute.

    Args:
        max_per_minute: Maximum number of calls allowed per minute.
    """
    lock = threading.Lock()
    min_interval = 60.0 / float(max_per_minute)

    def decorate(func):
        last_time_called = [0.0]

        def rate_limited_function(*args, **kwargs):
            with lock:
                elapsed = time.time() - last_time_called[0]
                left_to_wait = min_interval - elapsed
                if left_to_wait > 0:
                    time.sleep(left_to_wait)
                ret = func(*args, **kwargs)
                last_time_called[0] = time.time()
                return ret

        return rate_limited_function

    return decorate


@RateLimited(110)
def get_vast_answer(
    query: str,
    api_key: str,
    user: str,
    config: VastConfig,
    timeout: int = 1000,
) -> dict:
    """Query the RAG platform API with a given configuration.

    Args:
        query: The question to submit.
        api_key: API authentication key.
        user: User identifier.
        config: Complete RAG pipeline configuration.
        timeout: Request timeout in seconds.

    Returns:
        JSON response dictionary containing the answer and metadata.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": user,
        "dgl_model_config": config.model_dump(),
    }

    # Configure retry strategy for robustness
    session = requests.Session()
    retries = Retry(
        total=20,
        backoff_factor=1,
        status_forcelist=[400, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    response = session.post(
        settings.URL,
        json=payload,
        headers=headers,
        stream=False,
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json()
