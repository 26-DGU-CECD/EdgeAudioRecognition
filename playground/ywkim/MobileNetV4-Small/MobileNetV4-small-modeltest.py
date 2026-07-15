import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import timm

# ==========================================
# 0. 환경 설정 및 경로 정의
# ==========================================
BASE_DATA_DIR = r'D:\EnnaKim\대학교 4학년\대학교 4학년 1학기\종설1\GithubModelCheck\data\drive-download-20260702T124347Z-3-001'
INTEGRATED_CSV_PATH = r'D:\EnnaKim\대학교 4학년\대학교 4학년 1학기\종설1\GithubModelCheck\data\integrated_test_labels.csv'
MODEL_SAVE_PATH = r'D:\EnnaKim\대학교 4학년\대학교 4학년 1학기\종설1\GithubModelCheck\mobilenetv4_small_trained.pth'

# 학습 환경 설정 (GPU가 있다면 cuda, 없다면 cpu)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 사용 중인 디바이스: {DEVICE}")

# 내가 학습 및 검증하고 싶은 '특정 라벨'만 정의
CLASS_NAMES = ['물소리', 'knock', 'dog_bark', 'cat_meow']
#'baby_cry','울음', '비명', '유리깨지는소리'
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# 학습 세팅
EPOCHS = 20  # 테스트용으로 5번만 학습 (제대로 성능을 내려면 20~30 이상 추천)
BATCH_SIZE = 16
LR = 1e-4

# ==========================================
# 1. 80% : 20% 데이터셋 분할 로직
# ==========================================
raw_df = pd.read_csv(INTEGRATED_CSV_PATH)
# 원하지 않는 라벨 제외하고 내가 정한 특정 라벨만 필터링
filtered_df = raw_df[raw_df['label'].isin(CLASS_NAMES)].reset_index(drop=True)

# stratify=filtered_df['label'] 옵션을 주어 80%와 20% 내의 클래스 비율을 똑같이 맞춥니다.
train_df, val_df = train_test_split(
    filtered_df,
    test_size=0.2,
    stratify=filtered_df['label'],
    random_state=42
)

print(f"📊 데이터 분할 완료 -> 총 타겟 데이터: {len(filtered_df)}개")
print(f"   - 학습(Train) 데이터 개수 (80%): {len(train_df)}개")
print(f"   - 검증(Val/Test) 데이터 개수 (20%): {len(val_df)}개")


# ==========================================
# 2. 오디오 Dataset 정의
# ==========================================
class AudioDataset(Dataset):
    def __init__(self, dataframe, base_data_dir, class_to_idx, sample_rate=16000, n_mels=128, duration=4):
        self.df = dataframe
        self.base_data_dir = base_data_dir
        # [추정] 002 폴더 경로도 자동으로 계산되도록 세팅
        self.base_data_dir_002 = base_data_dir.replace('-001', '-002')

        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # [속도 최적화] 주파수 변환기(Resampler)를 저장해둘 캐시 딕셔너리
        self.resamplers = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['filename'].replace('/', os.sep)

        # 1순위: 001 폴더에서 파일 찾기
        audio_path = os.path.join(self.base_data_dir, rel_path)

        # 2순위: 만약 001에 없다면 002 폴더에서 찾기
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.base_data_dir_002, rel_path)

        label_idx = self.class_to_idx[row['label']]

        try:
            # 두 폴더 모두에 진짜 없는 경우에만 무음 처리
            if not os.path.exists(audio_path):
                raise FileNotFoundError("001 및 002 압축 폴더 어디에도 파일이 존재하지 않습니다.")

            waveform, sr = torchaudio.load(audio_path)

            # [속도 최적화] 동일한 오디오 주파수(sr)는 필터를 새로 만들지 않고 재사용하여 렉 제거
            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = T.Resample(sr, self.sample_rate)
                waveform = self.resamplers[sr](waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if waveform.shape[1] > self.num_samples:
                waveform = waveform[:, :self.num_samples]
            elif waveform.shape[1] < self.num_samples:
                waveform = F.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        except Exception as e:
            # 정말 파일이 없는 경우에만 최소한으로 출력되도록 제한
            print(f"⚠️ [파일 접근 실패] -> {row['filename']} | 사유: {e}")
            waveform = torch.zeros((1, self.num_samples))

        mel_spec = self.amplitude_to_db(self.mel_spectrogram(waveform))

        mel_min = mel_spec.min()
        mel_max = mel_spec.max()
        if mel_max - mel_min > 1e-5:
            mel_spec_norm = (mel_spec - mel_min) / (mel_max - mel_min)
        else:
            mel_spec_norm = torch.zeros_like(mel_spec)

        mel_spec_rgb = mel_spec_norm.repeat(3, 1, 1)
        return mel_spec_rgb, label_idx

# ==========================================
# 3. 데이터 로더 세팅
# ==========================================
train_dataset = AudioDataset(train_df, BASE_DATA_DIR, CLASS_TO_IDX)
val_dataset = AudioDataset(val_df, BASE_DATA_DIR, CLASS_TO_IDX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================
# 4. MobileNetV4-Small 모델 정의 및 학습 함수
# ==========================================
# pretrained=True로 설정하면 ImageNet으로 선행 학습된 뼈대를 가져와서 소리 분류 학습 속도가 훨씬 빨라집니다.
model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

print("\n🚀 [학습 시작] MobileNetV4-Small 모델을 학습시킵니다...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (correct / total) * 100
    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 학습 완료된 모델 저장
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 모델 가중치 저장 완료: {MODEL_SAVE_PATH}")

# ==========================================
# 5. [요구사항 2] 20% 검증셋 기반 Precision, Recall, F1 점수 평가
# ==========================================
model.eval()
all_preds = []
all_trues = []

print("\n🔍 [평가 시작] 20% 검증 데이터셋으로 모델 성능을 측정합니다...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(labels.numpy())

precision = precision_score(all_trues, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_trues, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_trues, all_preds, average='weighted', zero_division=0)

print("\n=========================================")
print("      MobileNetV4-Small 최종 검증 보고서   ")
print("=========================================")
print(f"Precision (정밀도) : {precision:.4f}")
print(f"Recall    (재현율) : {recall:.4f}")
print(f"F1-Score  (F1 점수) : {f1:.4f}")
print("=========================================")
print("\n[클래스별 상세 성능 리포트]")
print(classification_report(all_trues, all_preds, target_names=CLASS_NAMES, zero_division=0))

# ==========================================
# 6. [요구사항 1] 단일 데이터 결과 확인 (라벨과 점수만)
# ==========================================
print("\n[요구사항 1] 단일 샘플 예측 결과 테스트")
model.eval()
# 검증셋의 첫 번째 데이터를 가지고 개별 예측을 수행해봅니다.
single_input, single_true = val_dataset[0]
single_input = single_input.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    single_output = model(single_input)
    probabilities = F.softmax(single_output, dim=1)
    score, pred_idx = torch.max(probabilities, dim=1)

print(f"실제 정답: {CLASS_NAMES[single_true]}")
print(f"결과 -> 라벨: {CLASS_NAMES[pred_idx.item()]}, 점수: {score.item():.4f}")