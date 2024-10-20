# Chọn hình ảnh cơ sở
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt vào trong container
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào trong container
COPY . .

# Chạy ứng dụng
CMD ["python", "app.py"]