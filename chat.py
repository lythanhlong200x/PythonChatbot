import random
import json
import openai
import torch
import requests

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = "data1.pth"
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
import os
from huggingface_hub import InferenceClient

bot_name = "Sam"
import cohere
import time

co = cohere.Client("OjKKymufS81FhzxfDeuIZwRHhogduxO3Rs4zSPrs")
cache = {}
cache_expiry = {}


def format_response(response_text):
    # Chia nhỏ câu trả lời thành các đoạn và thêm ngắt dòng
    paragraphs = response_text.split(". ")
    formatted_response = "<br>".join(paragraphs)
    return formatted_response


def get_response_from_cohere(msg):
    start_time = time.time()  # Bắt đầu ghi thời gian
    try:
        response = co.generate(
            model="command-xlarge-nightly",
            prompt=f"Hãy trả lời ngắn gọn và có cấu trúc rõ ràng cho câu hỏi sau: {msg}",
            max_tokens=500,
            temperature=0.5,
        )
        response_text = response.generations[0].text.strip()
        cache[msg] = format_response(response_text)
        cache_expiry[msg] = time.time() + 600  # Hết hạn sau 10 phút (600 giây)
        duration = time.time() - start_time  # Tính toán thời gian
        print(f"Thời gian gọi API: {duration:.2f} giây")
        return format_response(response_text)
    except Exception as e:
        print(f"Có lỗi khi gọi Cohere: {e}")
        return "Xin lỗi, tôi không thể trả lời câu hỏi này ngay bây giờ."


import torch


def find_similar_question(new_question):
    # Vector hóa câu hỏi mới
    new_vector = bag_of_words(tokenize(new_question), all_words)
    new_vector = new_vector.reshape(1, -1)

    # Chuyển đổi new_vector thành Tensor
    new_vector_tensor = torch.tensor(new_vector, dtype=torch.float32).to(device)

    # Tìm câu hỏi tương tự trong cache
    for cached_question in cache.keys():
        cached_vector = bag_of_words(tokenize(cached_question), all_words)
        cached_vector = cached_vector.reshape(1, -1)

        # Chuyển đổi cached_vector thành Tensor
        cached_vector_tensor = torch.tensor(cached_vector, dtype=torch.float32).to(
            device
        )

        # Tính độ tương đồng
        similarity = torch.cosine_similarity(new_vector_tensor, cached_vector_tensor)

        if similarity.item() > 0.8:  # Thay đổi ngưỡng này nếu cần
            return cache[cached_question]

    return None


def get_response(msg):
    similar_response = find_similar_question(msg)
    if similar_response:
        return similar_response
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return get_response_from_cohere(msg)


if __name__ == "__main__":
    print("Chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
