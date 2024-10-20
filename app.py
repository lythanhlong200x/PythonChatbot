from flask import Flask, render_template, request, jsonify
from chat_testing import get_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is actual valid
    if not text:
        return (
            jsonify({"answer": "Xin vui lòng nhập một câu hỏi hợp lệ."}),
            400,
        )  # Trả về mã lỗi 400 nếu không hợp lệ

    try:
        response = get_response(text)
    except Exception as e:
        return (
            jsonify({"answer": "Có lỗi xảy ra: " + str(e)}),
            500,
        )  # Trả về mã lỗi 500 nếu có lỗi

    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
