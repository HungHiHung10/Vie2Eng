# Vie2Eng: Cấu trúc mã nguồn

Ứng dụng dịch Việt ⇄ Anh cục bộ bằng MarianNMT CTranslate2 INT8 hoặc các checkpoint PhoMT PyTorch. Backend chạy Python 3.10+; frontend dùng ES modules trực tiếp, không cần build.

## Khởi chạy

- `python server.py` hoặc `run_translator.bat`: chạy tại `http://127.0.0.1:8000`.
- `uvicorn server:app`: sử dụng cùng ứng dụng ASGI.
- `python download_models.py`: tải và kiểm tra model INT8 cùng tokenizer tương ứng.

`server.py` tạo ứng dụng qua `vie2eng.api.create_app()`. Hàm này nhận một dịch vụ dịch tùy chọn để kiểm thử độc lập. Model và tokenizer được nạp khi dịch lần đầu.

## Phân chia backend

| Mô-đun | Trách nhiệm |
| --- | --- |
| `vie2eng/config.py` | Đường dẫn, repository, phiên bản model/tokenizer, giới hạn cache và backend mặc định. |
| `vie2eng/schemas.py` | Kiểm tra hợp đồng request/response bằng Pydantic. |
| `vie2eng/language.py` | Nhận diện Việt/Anh, độc lập với HTTP và thư viện model. |
| `vie2eng/service.py` | Xác định chiều dịch, chọn backend, chuẩn hóa giải mã, tra cache và đo thời gian. |
| `vie2eng/cache.py` | LRU cache có khóa đa luồng, loại mục ít dùng gần đây nhất khi đầy. |
| `vie2eng/backends/ct2.py` | Tải/cache tài nguyên, kiểm tra từ vựng, chia văn bản dài và suy luận INT8 trên CPU. |
| `vie2eng/backends/legacy.py` | Nạp checkpoint cũ khi cần và giải mã Greedy/Beam. |
| `vie2eng/backends/transformer.py` | Positional encoding và kiến trúc Transformer khớp state dictionary cũ. |
| `vie2eng/api.py` | Endpoint, static files, CORS, lỗi HTTP và phiên bản tài nguyên giao diện. |

Mỗi backend cung cấp `translate(text, direction, method, beam_size, max_length)` và thuộc tính `device`. Backend sở hữu model/tokenizer đã nạp; dịch vụ sở hữu cache. Các ứng dụng có thể dùng dịch vụ riêng khi kiểm thử.

## Luồng dịch

1. `POST /api/translate` kiểm tra dữ liệu theo `TranslationRequest`.
2. Dịch vụ làm sạch khoảng trắng, nhận diện nguồn và xác định đích. Nếu hai ngôn ngữ trùng nhau, chọn ngôn ngữ còn lại.
3. Chọn `vi_en`/`en_vi` và backend `marian_ct2`/`legacy`. Beam size 1 được chuẩn hóa thành Greedy.
4. Khóa cache gồm tên model, ngôn ngữ thực tế, phương pháp/beam hiệu lực, độ dài tối đa và văn bản.
5. Nếu chưa có kết quả, backend nạp model khi cần rồi dịch cục bộ. Khóa nạp ngăn các yêu cầu đầu tiên nạp trùng model.
6. Lưu kết quả và trả lại đúng ngôn ngữ, model, phương pháp, thời gian và cờ cache.

Kết quả rỗng vẫn được cache. Yêu cầu suy luận thất bại không được cache. Lỗi backend chuyển thành `TranslationUnavailable`, sau đó HTTP 503 với hướng dẫn chuẩn bị model. Không trả nguyên văn lỗi thư viện vì có thể chứa thông tin xác thực.

## Vòng đời model

Model INT8 và tokenizer được cố định phiên bản riêng trong `config.py`. Hai repository INT8 không kèm tokenizer; backend lấy tokenizer từ repository PhoMT gốc tương ứng và đối chiếu token ID với từ vựng đã chuyển đổi trước khi nạp.

Tài nguyên nằm trong `models/`, hoặc thư mục `VIE2ENG_MODEL_CACHE`. `HF_TOKEN` được đọc từ môi trường khi cần, không gửi xuống trình duyệt. Khi tài nguyên đã tải đủ, suy luận chạy offline. Font và dịch vụ giọng nói của trình duyệt có hành vi mạng riêng.

Checkpoint `.pt` nằm ở thư mục gốc. Kiến trúc và thuật toán giải mã của chúng được giữ nguyên. CTranslate2 dùng CPU INT8; backend cũ chọn thiết bị PyTorch sẵn có.

`GET /api/health` báo trạng thái máy chủ, cấu hình model/thiết bị và số mục cache. Endpoint này không nạp hay xác thực trọng số.

## Phân chia frontend

| Mô-đun | Trách nhiệm |
| --- | --- |
| `static/app.js` | Ghép các mô-đun, hiển thị kết quả, khôi phục bản dịch và gắn sự kiện chung. |
| `static/js/dom.js` | Tập trung việc lấy các phần tử giao diện. |
| `static/js/state.js` | Trạng thái ban đầu và đọc lịch sử/đã lưu. |
| `static/js/languages.js` | Chọn, hoán đổi và đồng bộ ngôn ngữ từ API hoặc bản ghi. |
| `static/js/translation.js` | Gọi API, xử lý lỗi, debounce và thứ tự phản hồi. |
| `static/js/preferences.js` | Giao diện sáng/tối, model và tham số giải mã. |
| `static/js/records.js` | Lịch sử, yêu thích, bộ đếm và hàm hiển thị bản ghi dùng chung. |
| `static/js/media.js` | Nhập/đọc giọng nói và tải văn bản `.txt`. |

Ngôn ngữ đích lấy từ phản hồi API; nút được chọn và giọng đọc cùng sử dụng giá trị đó. Mở lịch sử/đã lưu khôi phục cả cặp ngôn ngữ và văn bản, không tạo yêu cầu dịch mới.

Bộ điều khiển dịch quản lý mã thứ tự yêu cầu và thời gian chờ gõ. Sửa/xóa văn bản hoặc khôi phục bản ghi làm phản hồi trước đó mất hiệu lực. Hoán đổi ngôn ngữ cập nhật cặp ngôn ngữ và văn bản trước khi gửi một yêu cầu.

Các khóa localStorage giữ nguyên: `phomt_history`, `phomt_saved`, `phomt_model_backend`, `phomt_theme`. Dữ liệu cũ tiếp tục đọc được; JSON lịch sử/đã lưu không hợp lệ được thay bằng danh sách rỗng.

## Cập nhật giao diện

HTML tải `app.js` bằng ES module. API thay `__ASSET_VERSION__` bằng mã hash của toàn bộ JavaScript, gồm các mô-đun được import. HTML, JavaScript và CSS dùng `Cache-Control: no-cache`: trình duyệt được lưu nhưng phải kiểm tra lại trước khi tái sử dụng. Không cần tăng chuỗi phiên bản thủ công sau mỗi lần sửa.

Khởi động lại máy chủ Python sau khi thay backend. Frontend không cần build.

## Kiểm thử

Cài `requirements-dev.txt`, sau đó chạy:

```bash
python -m unittest discover -s tests -v
node --experimental-vm-modules tests/test_client.cjs
```

Kiểm thử API đưa backend giả vào `create_app(service)`, không cần mạng hay checkpoint. Các tình huống gồm định tuyến ngôn ngữ/model, tham số, cache, lỗi an toàn, phiên bản JavaScript và loại mục LRU.

Kiểm thử client nạp đồ thị ES modules với DOM giả lập; không kiểm tra bố cục trên trình duyệt thật. Đặt `VIE2ENG_TEST_URL` để kiểm tra thêm HTML và các mô-đun máy chủ thực sự đang phục vụ.

Sau khi chuẩn bị model và checkpoint, chạy kiểm thử suy luận thật bằng PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:VIE2ENG_TEST_MODELS = "1"
python -m unittest discover -s tests -v
```

Các kiểm thử này chạy hai chiều dịch, hai thuật toán, hai backend và kiểm tra giữ nội dung dài/đoạn trống. Các câu mẫu xác nhận thực thi và định tuyến; không thay thế đánh giá chất lượng trên toàn bộ tập dữ liệu.
