import os
import requests
from dotenv import load_dotenv
load_dotenv()

import requests

model = "openchat/openchat_3.5"
model = "HuggingFaceH4/zephyr-7b-beta"
# "inputs": "Can you please let us know more details about your ",
model = "distilbert-base-uncased-distilled-squad"


API_URL = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}

import time

def query(payload):
    retries = 5
    for i in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
            return response.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
            if i < retries - 1:
                print("Retrying...")
                time.sleep(1)
                continue
            else:
                print("Error querying API. Max retries exceeded.")
                raise
	
output = query({
     "inputs": {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France."
    }
})

print(output)