import requests
import time

REWARD_API_URL = "http://localhost:38000/compute_score"


def compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info=None,
        sandbox_fusion_url=None,
        concurrent_semaphore=None,
        memory_limit_mb=None,
        max_retries=10
):

    for attempt in range(max_retries + 1):
        try:
            payload = {"solution_str": solution_str}

            response = requests.post(REWARD_API_URL, json=payload)
            response.raise_for_status()

            result = response.json()
            return result["reward_score"]

        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                print(f"Error after {max_retries + 1} attempts: {e}")
                return 0.0

            print(f"Attempt {attempt + 1} failed, retrying in 100ms... ({e})")
            time.sleep(0.1)