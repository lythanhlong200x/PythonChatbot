import random
import json
import torch
import os
import time
from flask import Flask, request, jsonify
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import google.generativeai as gen_ai
from dotenv import load_dotenv

# Load environment variables

app = Flask(__name__)

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents
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

# Initialize neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Configure Google Gemini model
GOOGLE_API_KEY = "AIzaSyA73rSuqrn6kp_9M85ICgkSv1UtdmhQO-g"
gen_ai.configure(api_key=GOOGLE_API_KEY)
model_ai = gen_ai.GenerativeModel("gemini-pro")


# def format_response(response_text):
#     # Thay thế dấu "*" bằng các thẻ HTML hoặc định dạng khác
#     formatted_response = (
#         response_text.replace(" * ", "<br> - ")
#         .replace("**", "<br>")
#         .replace(":**", "<br>")
#     )
#     return formatted_response
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message")
    response = get_response(msg)  # Gọi hàm xử lý câu hỏi
    return jsonify({"response": response})


def format_response(response_text):
    # Thay thế dấu ** đầu tiên bằng <strong>, giữ nguyên dấu * nếu có
    segments = response_text.split("**")

    formatted_segments = []
    for i, segment in enumerate(segments):
        if i % 2 == 1:  # Nếu là đoạn giữa (giữa các dấu **)
            # Kiểm tra xem có dấu * ở đầu đoạn không
            if segment.startswith(" * "):
                # Thay dấu * đầu tiên bằng khoảng trắng và giữ nguyên đoạn
                formatted_segments.append(segment.replace(" * ", "<br>", 1).strip())
            else:
                formatted_segments.append(f"<strong>{segment.strip()}</strong>")
        else:
            # Kiểm tra dấu chấm ở cuối đoạn
            if segment.strip().endswith("."):
                formatted_segments.append(
                    segment.strip() + "<br>"
                )  # Thêm <br> nếu có dấu chấm ở cuối
            else:
                formatted_segments.append(segment.strip())
    formatted_response = "".join(formatted_segments) + "<br>"
    formatted_response = formatted_response.replace(
        segments[1], segments[1] + "<br>", 1
    )

    # Thay thế dấu * bằng dấu ngắt dòng
    formatted_response = formatted_response.replace("*", "<br> - ")

    return formatted_response


def get_response_from_gemini(msg):
    try:
        # Gọi API Gemini để nhận phản hồi
        response = model_ai.generate_content(msg)
        return format_response(response.text.strip())
        # return response.text.strip()
    except Exception as e:
        print(f"Có lỗi khi gọi Gemini: {e}")
        return "Xin lỗi, tôi không thể trả lời câu hỏi này ngay bây giờ."


use_ai_bot = False


def get_response(msg):
    global use_ai_bot
    if msg.lower() == "ai bot" or msg.lower() == "ai chat":
        use_ai_bot = True
        return "🤖Chế độ AI Bot đã được kích hoạt. Mọi câu hỏi tiếp theo sẽ được trả lời bởi mô hình AI.🤖"

    # Kiểm tra nếu người dùng muốn quay lại chế độ chatbot hỗ trợ
    if msg.lower() == "support bot" or msg.lower() == "support chat":
        use_ai_bot = False
        return "🧠Chế độ Support Bot đã được kích hoạt. Mọi câu hỏi tiếp theo sẽ được trả lời bởi mô hình hỗ trợ."
    # Xử lý câu hỏi bằng mô hình

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if use_ai_bot:
        return get_response_from_gemini(msg)
    else:
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])

        # Nếu không có phản hồi nào từ mô hình, sử dụng Gemini
        return (
            "Xin lỗi, tôi không thể trả lời câu hỏi này. "
            "<br>"
            "Mọi thắc mắc liên hệ trang Hỗ trợ hoặc chuyển sang chatbot AI."
            "<br>"
            'Để chuyển đổi giữa 2 loại chatbot AI và Support vui lòng nhập lệnh: AI/Support + Chat hoặc AI/Support + Bot ".'
        )


if __name__ == "__main__":
    print("Chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"Bot: {resp}")
