import os
import json
from huggingface_hub import InferenceClient

bot_name = "Sam"
HF_Token = "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"
repo_id = "microsoft/TinyLlama-1.1B-Chat-v1.0"
llm_client = InferenceClient(model=repo_id, timeout=120)


def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


response = call_llm(llm_client, "tell me a joke")
print(response)
