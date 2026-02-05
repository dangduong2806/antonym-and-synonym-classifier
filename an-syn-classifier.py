import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import fasttext.util
import joblib

AN_PATH = "data/Antonym_vietnamese.txt"
SYN_PATH = "data/Synonym_vietnamese.txt"
MODEL_SAVE_PATH = "saved_checkpoints"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("Đang kiểm tra/tải model FastText tiếng Việt...")
# Lưu ý: Mã ngôn ngữ tiếng Việt là 'vi', không phải 'vie' cho tên file bin
fasttext.util.download_model('vi', if_exists='ignore')

print("Đang load model vào RAM (có thể tốn 4-7GB RAM)...")
word_vectors = fasttext.load_model('cc.vi.300.bin')

def read_txt_data(file_path, label):
    data_list = []
    print(f"Đang đọc file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                parts = line.split()
                if len(parts) >= 2:
                    data_list.append({'word1': parts[0], 'word2': parts[1], 'label': label})
    except FileNotFoundError:
        print(f"⚠ Lỗi: Không tìm thấy file {file_path}")
    return data_list

antonyms = read_txt_data(AN_PATH, 0) # Nhãn 0: trái nghĩa
synonyms = read_txt_data(SYN_PATH, 1) # Nhãn 1: đồng nghĩa

# Gộp thành dataframe
df = pd.DataFrame(antonyms + synonyms)
print(f"Tổng số dữ liệu: {len(df)} cặp từ.")

# Feature engineering
def create_pair_feature(w1, w2, model):
    # Các từ đã được pre-process
    # Nó xử lý được cả từ sai chính tả hoặc từ hiếm
    v1 = model.get_word_vector(w1)
    v2 = model.get_word_vector(w2)
    # Kỹ thuật: Kết hợp phép nhân (độ tương đồng) và phép trừ (khoảng cách)
    return np.concatenate([v1 * v2, np.abs(v1 - v2)])

X = []
y = []
skipped_count = 0

if word_vectors:
    for index, row in df.iterrows():
        feature = create_pair_feature(row['word1'], row['word2'], word_vectors)
        if feature is not None:
            X.append(feature)
            y.append(row['label'])
        else:
            skipped_count += 1

print(f"Số lượng cặp từ có vector: {len(X)}")
print(f"Số lượng bị bỏ qua (không có trong từ điển): {skipped_count}")

X = np.array(X)
y = np.array(y)

# Chia tập train và valid
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Kích thước tập Train: {len(X_train)}")
print(f"Kích thước tập Valid: {len(X_val)}")

# Huấn luyện logistic
print("\n--- 1. Training Logistic Regression ---")
if len(X_train) > 0:
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Đánh giá trên tập valid
    y_pred_lr = lr_model.predict(X_val)
    print(f"Logistic Regression Validation Accuracy: {accuracy_score(y_val, y_pred_lr):.4f}")
    print(classification_report(y_val, y_pred_lr))

    # LƯU MODEL
    lr_path = os.path.join(MODEL_SAVE_PATH, 'lr_model.pkl')
    joblib.dump(lr_model, lr_path)
    print(f"-> Đã lưu LR model tại: {lr_path}")
else:
    print("Không đủ dữ liệu để train LR.")


# HUẤN LUYỆN MLP (MULTI-LAYER PERCEPTRON)
if len(X_train) > 0:
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    mlp_model.fit(X_train, y_train)

    # Đánh giá
    y_pred_mlp = mlp_model.predict(X_val)
    print(f"MLP Validation Accuracy: {accuracy_score(y_val, y_pred_mlp):.4f}")
    print(classification_report(y_val, y_pred_mlp))

    # LƯU MODEL
    mlp_path = os.path.join(MODEL_SAVE_PATH, 'mlp_model.pkl')
    joblib.dump(mlp_model, mlp_path)
    print(f"-> Đã lưu MLP model tại: {mlp_path}")
else:
    print("Không đủ dữ liệu để train MLP.")