# Lab 21 — Evaluation Report

**Học viên**: Hà Việt Khánh — 2A202600055
**Ngày nộp**: 2026-05-07
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen 2.5 3B, quantized 4-bit)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up to power of 2)
- **GPU**: Tesla T4, 14.6 GB VRAM
- **Training cost**: $0.07 (@ $0.35/hr, tổng training time ~12.3 phút)
- **HF Hub link**: N/A

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.02 min   | 7.22 GB   | 1.5577    | 4.7479     |
| 16   | 3,686,400       | 4.27 min   | 6.62 GB   | 1.5161    | 4.5544     |
| 64   | 14,745,600      | 4.01 min   | 8.00 GB   | 1.4768    | 4.3790     |
| Base | -               | -          | -         | -         | -          |

*Ghi chú: VRAM của r=16 thấp hơn r=8 có thể do sự khác biệt trong việc quản lý cache của PyTorch giữa các lần chạy.*

## 3. Loss Curve Analysis
![Loss Curve](results/loss_curve.png)
- Quan sát: Training loss giảm ổn định từ 1.6 xuống khoảng 1.27-1.39 tùy rank. Do tập dữ liệu nhỏ (200 mẫu), mô hình không có dấu hiệu overfitting nghiêm trọng. Eval loss tỉ lệ nghịch với rank (rank càng cao loss càng thấp), cho thấy khả năng khớp dữ liệu tốt hơn khi tăng dung lượng adapter.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu... Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu.
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo)...
**Nhận xét**: Improved. Câu trả lời của mô hình fine-tuned nghe tự nhiên và đầy đủ hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: `if n <= 0: return "N phải là một số dương"`. (Code dùng đệ quy cơ bản)
**Fine-tuned (r=16)**: `if n < 0: raise ValueError("Input phải là một số nguyên dương.")`. (Code dùng vòng lặp tối ưu và xử lý exception)
**Nhận xét**: Improved. Code của mô hình fine-tuned chuyên nghiệp hơn và hiệu quả hơn về mặt thuật toán.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng... 2. Tru... (câu trả lời bị cắt ngắn hoặc chưa rõ ràng)
**Fine-tuned (r=16)**: 1. Chuyển đổi: Hướng tới hành động. 2. Thích ứng: Đa thiết bị. 3. Đơn giản: Dễ hiểu...
**Nhận xét**: Improved. Trình bày súc tích và đúng các tiêu chuẩn thiết kế hiện đại.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA là phương pháp cải thiện hiệu năng bằng cách sử dụng các phép biến đổi thấp độ phức tạp.
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA là hai phương pháp regularization...
**Nhận xét**: Same/Slightly Degraded. Mô hình fine-tuned bị nhầm tên viết tắt của LoRA thành "Layer-wise Adaptive...", tuy nhiên vẫn hiểu đúng bản chất cơ chế nén/lượng hóa của QLoRA.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering là kỹ thuật cải thiện hiệu suất bằng cách cung cấp câu hỏi hoặc câu lệnh dựa vào...
**Fine-tuned (r=16)**: Prompt engineering tập trung vào xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề... RAG kết hợp truy xuất...
**Nhận xét**: Improved. Phân biệt rõ ràng mục đích sử dụng của từng kỹ thuật trong thực tế.

## 5. Conclusion về Rank Trade-off

- **Rank nào cho ROI tốt nhất?**: Rank **r=16** mang lại tỷ lệ ROI tốt nhất. Nó cải thiện đáng kể perplexity so với r=8 (4.55 vs 4.75) trong khi tốn thêm rất ít tài nguyên VRAM.
- **Diminishing returns**: Việc tăng từ r=16 lên r=64 giúp perplexity giảm thêm xuống 4.38, nhưng số lượng tham số huấn luyện tăng gấp 4 lần. Với một dataset nhỏ 200 mẫu, r=16 là đủ để "học" được style mà không cần tiêu tốn quá nhiều tài nguyên.
- **Recommendation**: Nếu deploy production, tôi chọn **r=16**. Nó đảm bảo độ chính xác đủ tốt, tiết kiệm bộ nhớ và tối ưu chi phí vận hành.

## 6. What I Learned
- **Fine-tuning là để dạy "phong cách"**: Nó hiệu quả nhất khi dùng để định hình format trả lời và giọng văn thay vì chỉ nạp kiến thức thô.
- **Sức mạnh của QLoRA**: Giúp việc huấn luyện các mô hình 3B-7B trở nên cực kỳ dễ dàng trên GPU phổ thông như T4.
- **Lựa chọn rank**: Cần cân bằng giữa hiệu năng và dung lượng tham số; rank cao không phải lúc nào cũng mang lại sự cải thiện vượt bậc tương xứng với chi phí.
