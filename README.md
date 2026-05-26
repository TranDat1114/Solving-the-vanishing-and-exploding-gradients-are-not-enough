# Chứng minh RNN bất ổn định bằng các phương pháp số

Repository này chứa các Jupyter Notebook chứng minh sự bất ổn định của RNN (Recurrent Neural Network) thông qua 5 phương pháp phân tích và mô phỏng số.

## Cấu trúc thư mục

```
.
├── 01_Curse_of_Memory/
│   └── curse_of_memory_analysis.ipynb       # Phân tích "curse of memory"
├── 02_Signal_Propagation/
│   └── signal_propagation_analysis.ipynb    # Phân tích truyền tín hiệu
├── 03_Non_Diagonal_Matrix/
│   └── fully_connected_rnn_analysis.ipynb   # Mở rộng sang ma trận không chéo
├── 04_Loss_Landscape/
│   └── teacher_student_loss_landscape.ipynb # Phân tích loss landscape teacher-student
├── 05_Hessian_Analysis/
│   └── hessian_at_optimum.ipynb             # Phân tích Hessian tại nghiệm tối ưu
├── README.md
└── requirements.txt
```

## Cách chứng minh

### 1. Phân tích "Curse of Memory"
- Mô phỏng RNN với memory dài (|λ| → 1)
- Tính độ nhạy của hidden state ∂h_k/∂λ
- Chứng minh phương sai đạo hàm tăng nhanh khi |λ| → 1

### 2. Phân tích truyền tín hiệu (Signal Propagation)
- **Forward pass**: phương sai hidden state E[h²_k] tăng vô hạn khi k tăng
- **Backward pass**: độ nhạy gradient bùng nổ nhanh hơn forward
- Chứng minh việc học khó khăn dù không có exploding gradients

### 3. Ma trận không chéo (Fully-connected RNN)
- Mở rộng phân tích cho ma trận A nhiều chiều
- Chứng minh sự nhạy cảm lan ra toàn bộ tham số

### 4. Loss Landscape (Teacher-Student)
- Mô hình h_{t+1} = λh_t + x_t
- Trực quan hóa loss landscape "sắc nhọn" khi memory dài
- Chứng minh gradient-based optimization bất khả thi

### 5. Phân tích Hessian
- Tính Hessian tại nghiệm tối ưu
- Chứng minh eigenvalues lớn (curse of memory)
- So sánh SGD vs Adam

## Cách chạy

```bash
pip install -r requirements.txt
jupyter notebook
```

Mở từng notebook theo thứ tự để hiểu trình tự lập luận chứng minh.

## Tham khảo

Dựa trên bài báo về curse of memory trong RNN: bài báo chứng minh rằng việc loại bỏ vanishing/exploding gradients **không đủ** để đảm bảo RNN học ổn định. Vấn đề sâu hơn là **curse of memory**: khi memory dài, độ nhạy tham số sẽ tăng vọt, làm loss landscape trở nên khó tối ưu hóa. Đây là lý do tại sao các kiến trúc như LSTM, GRU, deep state-space models (SSMs) lại hoạt động tốt hơn — chúng có cơ chế giảm thiểu curse of memory thông qua **gating, normalization, và reparametrization**.
