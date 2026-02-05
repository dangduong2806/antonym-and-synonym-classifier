import pandas as pd
import numpy as np
import fasttext
import joblib
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
from mlp_inference import parse_label, read_vicon_data, 

CHECKPOINT_PATH = "/kaggle/working/antonym-and-synonym-classifier/saved_checkpoints/lr_model.pkl"
FASTTEXT_PATH = "cc.vi.300.bin"

# Danh sách các files
TEST_FILES = [
    "test_data/400_noun_pairs.txt",
    "test_data/400_verb_pairs.txt",
    "test_data/600_adj_pairs.txt"
]

# Load model
try:
    word_vectors = fasttext.load_model(FASTTEXT_PATH)
    print("Load embedding text model thành công")
except Exception as e:
    print(f"Lỗi load FastText: {e}")
    exit()

print(f"2. Đang load Model phân loại từ {CHECKPOINT_PATH}...")
clf_model = joblib.load(CHECKPOINT_PATH)

# Chạy đánh giá từng file
print("\n=== BẮT ĐẦU ĐÁNH GIÁ (EVALUATION) ===")
print(f"Metrics: Precision, Recall, F1\n")

overall_y_true = []
overall_y_pred = []

for file_path in TEST_FILES:
    X_test, y_test = read_vicon_data(file_path)
    if X_test is None or len(X_test) == 0:
        continue

    # dự đoán
    y_pred = clf_model.predict(X_test)

    # Lưa lại để tính kq tổng hợp
    overall_y_true.extend(y_test)
    overall_y_pred.extend(y_pred)

    # Tính chỉ số cho file hiện tại
    # target_names khớp với quy ước: 0=Antonym, 1=Synonym
    report = classification_report(y_test, y_pred, target_names=['Antonym', 'Synonym'], digits=4)

    print(f"KẾT QUẢ TRÊN FILE: {os.path.basename(file_path)}")
    print("-" * 60)
    print(report)
    print("-" * 60)

# kết quả tổng hợp
if len(overall_y_true) > 0:
    print("\n=== KẾT QUẢ TỔNG HỢP (TẤT CẢ CÁC FILE) ===")
    print(classification_report(overall_y_true, overall_y_pred, target_names=['Antonym', 'Synonym'], digits=4))

    # Tính riêng từng chỉ số macro để so sánh nhanh
    precision, recall, f1, _ = precision_recall_fscore_support(overall_y_true, overall_y_pred, average='macro')
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
else:
    print("Không có dữ liệu test nào được xử lý.")