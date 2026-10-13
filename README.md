# Google Dịch - PhoMT AI Translator (Vie2Eng)

<p align="center">
  <img src="static/logo.png" alt="Google Dịch Logo" width="80" height="80">
  <br>
  <b>Hệ thống Dịch thuật Song ngữ Anh ⇄ Việt (PhoMT Marian NMT)</b>
</p>

---

## 🌟 Giới thiệu
**Vie2Eng** là hệ thống dịch thuật hai chiều Việt - Anh ứng dụng kiến trúc mạng Transformer Seq2Seq cục bộ. Dự án sở hữu giao diện người dùng Material Design 3 mô phỏng trải nghiệm Google Dịch, kết hợp với backend FastAPI hiệu năng cao.

### Tính năng nổi bật
- **Mô hình Transformer kép:** Điều phối suy luận giữa `vienv2.pt` (Việt ➔ Anh) và `enviv4.pt` (Anh ➔ Việt).
- **Thuật toán giải mã đa dạng:** Hỗ trợ cả **Beam Search** ($k=1\dots 6$) với cơ chế chống lặp từ $N$-gram, phạt độ dài và **Greedy Search** tốc độ cao.
- **Tự động nhận diện ngôn ngữ:** Heuristic Language Detection nhận diện tiếng Việt có dấu/không dấu và tiếng Anh.
- **Giao diện đa phương thức:**
  - Nhập liệu bằng giọng nói (Speech-to-Text).
  - Đọc phát âm văn bản (Text-to-Speech).
  - Tải tệp văn bản `.txt`.
  - Bộ nhớ đệm lịch sử dịch và đánh dấu sao (Bookmarks).
  - Chuyển đổi giao diện Sáng / Tối (Dark & Light Mode).
- **Offline / On-premise:** Toàn bộ quá trình suy luận chạy 100% trên máy tính nội bộ, không gửi dữ liệu ra bên ngoài.

---

## 🛠️ Cài đặt & Khởi chạy

### 1. Yêu cầu môi trường
- Python 3.10+
- PyTorch (`torch`)
- Hugging Face Transformers (`transformers`, `sentencepiece`, `sacremoses`)
- FastAPI & Uvicorn (`fastapi`, `uvicorn`, `pydantic`)

Cài đặt các gói phụ thuộc:
```bash
pip install torch transformers sentencepiece sacremoses fastapi uvicorn pydantic
```

### 2. Chuẩn bị Checkpoints Mô hình
Đặt 2 file trọng số đã huấn luyện vào thư mục gốc của dự án:
- `vienv2.pt`: Checkpoint mô hình dịch Việt ➔ Anh.
- `enviv4.pt`: Checkpoint mô hình dịch Anh ➔ Việt.

### 3. Khởi chạy
Nhấp đúp vào file `run_translator.bat` trên Windows, hoặc chạy lệnh:
```bash
python server.py
```
Sau đó mở trình duyệt và truy cập: [http://localhost:8000](http://localhost:8000).

---

## 📂 Cấu trúc Thư mục
```
Vie2Eng/
├── server.py                   # FastAPI backend & NMT Inference Engine
├── CODEBASE_DOCUMENTATION.md   # Tài liệu phân tích kỹ thuật kiến trúc toàn diện
├── run_translator.bat          # Script khởi chạy nhanh trên Windows
├── templates/
│   └── index.html              # Giao diện chính (Material 3)
├── static/
│   ├── app.js                  # Frontend Controller, Audio & Storage logic
│   ├── style.css               # Phong cách giao diện & Dark/Light mode
│   └── logo.png                # Logo thương hiệu
├── notebooks/                  # Notebooks huấn luyện và đánh giá mô hình
└── .gitignore                  # Bỏ qua các file checkpoint mô hình lớn
```

---

## 📄 Bản quyền & Tác giả
Phát triển bởi **NGUYEN HUU KHANH HUNG** (`nhkhanhhung2005@gmail.com`).
Dự án phục vụ mục đích nghiên cứu và học thuật.
