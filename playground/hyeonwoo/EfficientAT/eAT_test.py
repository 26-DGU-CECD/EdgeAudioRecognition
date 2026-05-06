"""
EfficientAT (MobileNetV3) + ESC-50 Inference Pipeline
------------------------------------------------------
EfficientAT 공식 레포(fschmid56/EfficientAT)의 torch.hub를 통해
mn10_as 사전학습 모델을 로드하고 ESC-50으로 추론합니다.

설치 필요 패키지:
    pip install torch torchaudio librosa soundfile pandas tqdm requests
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0. 설정 상수
# ─────────────────────────────────────────────
SAMPLE_RATE   = 32000
CLIP_DURATION = 10.0       # EfficientAT 은 10초 기준 학습
WINDOW_SIZE   = 800
HOP_SIZE      = 320
MEL_BINS      = 128        # EfficientAT 기본값
FMIN          = 0
FMAX          = None       # None → sr/2
CLASSES_NUM   = 527

MODEL_NAME    = "mn10_as"  # 선택 가능: mn10_as / mn20_as / mn40_as / dymn10_as
N_SAMPLES     = 50         # ESC-50에서 추론할 샘플 수

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파일 구조 (PANNs_test.py 와 동일한 방식)
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data"
MODEL_DIR  = BASE_DIR.parent / "models"
RESULT_DIR = BASE_DIR / "results"

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 타겟 위험음 클래스
TARGET_CLASSES = {
    "car_horn":  ["Car horn", "Vehicle horn, car horn, honking"],
    "siren":     ["Siren", "Civil defense siren", "Ambulance (siren)"],
    "alarm":     ["Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Alarm clock"],
    "screaming": ["Screaming", "Shout"],
    "crash":     ["Vehicle collision, crash", "Crash"],
}


# ─────────────────────────────────────────────
# 1. ESC-50 다운로드 (PANNs 코드와 동일)
# ─────────────────────────────────────────────
def download_esc50():
    esc_dir = DATA_DIR / "ESC-50-master"
    if esc_dir.exists():
        print(f"[✓] ESC-50 이미 존재: {esc_dir}")
        return esc_dir

    urls = [
        "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "https://zenodo.org/record/1203745/files/ESC-50-master.zip",
    ]
    zippath = DATA_DIR / "ESC-50-master.zip"

    for url in urls:
        try:
            print(f"[↓] 다운로드 시도: {url}")
            import requests
            with requests.get(url, stream=True, timeout=60,
                              headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zippath, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, ncols=60
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("[→] 압축 해제 중...")
            with zipfile.ZipFile(zippath, "r") as zf:
                zf.extractall(DATA_DIR)
            zippath.unlink()
            break
        except Exception as e:
            print(f"    [!] 실패: {e}")

    print(f"[✓] ESC-50 준비 완료: {esc_dir}")
    return esc_dir


def load_esc50_metadata(esc_dir: Path) -> pd.DataFrame:
    meta_path = esc_dir / "meta" / "esc50.csv"
    df = pd.read_csv(meta_path)
    df["filepath"] = df["filename"].apply(lambda fn: str(esc_dir / "audio" / fn))
    print(f"[✓] ESC-50 메타데이터 로드: {len(df)}개 샘플, {df['category'].nunique()}개 클래스")
    return df


# ─────────────────────────────────────────────
# 2. EfficientAT 모델 로드 (torch.hub)
# ─────────────────────────────────────────────
def load_efficientat_model(model_name: str = MODEL_NAME):
    import zipfile, shutil

    hub_dir     = Path.home() / ".cache" / "torch" / "hub"
    repo_dir    = hub_dir / "fschmid56_EfficientAT_main"
    zip_path    = hub_dir / "main.zip"

    # zip은 있는데 폴더가 없으면 직접 해제
    if zip_path.exists() and not repo_dir.exists():
        print("[→] hub 캐시 zip 압축 해제 중...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(hub_dir)
        # GitHub zip은 내부 폴더명이 'EfficientAT-main' 으로 나옴
        extracted = hub_dir / "EfficientAT-main"
        if extracted.exists():
            shutil.move(str(extracted), str(repo_dir))
        zip_path.unlink(missing_ok=True)
        print("[✓] 압축 해제 완료")

    print(f"[↓] EfficientAT 모델 로드 중: {model_name}")

    model = torch.hub.load(
        str(repo_dir),          # 로컬 경로로 직접 지정
        model_name,
        pretrained=True,
        source="local",         # 로컬 레포 사용
    )
    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] {model_name} 로드 완료 — 파라미터: {total_params/1e6:.2f}M, device: {DEVICE}")
    return model


# ─────────────────────────────────────────────
# 3. AudioSet 레이블 로드
# ─────────────────────────────────────────────
def load_audioset_labels() -> list:
    labels_path = MODEL_DIR / "class_labels_indices.csv"
    labels_url  = (
        "https://raw.githubusercontent.com/qiuqiangkong/"
        "audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
    )

    if not labels_path.exists():
        print("[↓] AudioSet 레이블 다운로드 시도...")
        try:
            import requests
            r = requests.get(labels_url, timeout=20)
            r.raise_for_status()
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print("[✓] 레이블 저장 완료")
        except Exception as e:
            print(f"[!] 레이블 다운로드 실패: {e} → 인덱스 기반 더미 레이블 사용")
            dummy = [
                {"index": i, "mid": f"/m/{i:05d}", "display_name": f"AudioSet_class_{i}"}
                for i in range(CLASSES_NUM)
            ]
            known = {
                48: "Car horn", 49: "Vehicle horn, car horn, honking",
                77: "Siren", 78: "Civil defense siren", 79: "Ambulance (siren)",
                388: "Alarm", 389: "Fire alarm",
                390: "Smoke detector, smoke alarm", 391: "Alarm clock",
                20: "Screaming", 21: "Shout",
                473: "Vehicle collision, crash", 474: "Crash",
            }
            for idx, name in known.items():
                if idx < CLASSES_NUM:
                    dummy[idx]["display_name"] = name
            pd.DataFrame(dummy).to_csv(labels_path, index=False)

    df     = pd.read_csv(labels_path)
    labels = df["display_name"].tolist()
    print(f"[✓] AudioSet 레이블 로드: {len(labels)}개")
    return labels


# ─────────────────────────────────────────────
# 4. 오디오 전처리
# ─────────────────────────────────────────────
def audio_to_logmel(filepath: str) -> np.ndarray:
    """
    오디오 → log-mel spectrogram (time_steps, mel_bins)
    EfficientAT 기본 설정: 128 mel bands, hop 320, window 800
    """
    waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(waveform) < target_len:
        # 짧은 경우 반복 패딩
        repeats = int(np.ceil(target_len / len(waveform)))
        waveform = np.tile(waveform, repeats)[:target_len]
    else:
        waveform = waveform[:target_len]

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=WINDOW_SIZE,
        hop_length=HOP_SIZE,
        n_mels=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel.T.astype(np.float32)   # (time_steps, mel_bins)


# ─────────────────────────────────────────────
# 5. 단일 파일 추론
# ─────────────────────────────────────────────
def infer_single(model, filepath: str, labels: list, top_k: int = 5) -> dict:
    log_mel = audio_to_logmel(filepath)
    # EfficientAT 입력: (batch, time, mel) → 모델 내부에서 처리
    tensor  = torch.from_numpy(log_mel).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

    # EfficientAT 출력 형태 처리 (tuple or tensor)
    if isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        logits = output

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # (527,)

    # Top-K
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds   = [(labels[i], float(probs[i])) for i in top_indices]

    # 위험음 타겟 점수
    target_scores = {}
    for cls_name, audioset_names in TARGET_CLASSES.items():
        score = 0.0
        for aname in audioset_names:
            if aname in labels:
                idx   = labels.index(aname)
                score = max(score, float(probs[idx]))
        target_scores[cls_name] = score

    return {
        "filepath":      filepath,
        "top_k":         top_preds,
        "target_scores": target_scores,
        "raw_probs":     probs,
    }


# ─────────────────────────────────────────────
# 6. ESC-50 배치 추론
# ─────────────────────────────────────────────
def run_esc50_inference(model, df: pd.DataFrame, labels: list,
                        n_samples: int = N_SAMPLES) -> pd.DataFrame:
    danger_cats = ["car_horn", "siren", "engine", "dog", "crying_baby",
                   "crackling_fire", "fireworks", "clapping"]
    priority = df[df["category"].isin(danger_cats)]
    others   = df[~df["category"].isin(danger_cats)]

    n_priority = min(len(priority), n_samples // 2)
    n_other    = n_samples - n_priority
    sample_df  = pd.concat([
        priority.sample(n=n_priority, random_state=42),
        others.sample(n=min(len(others), n_other), random_state=42),
    ]).reset_index(drop=True)

    print(f"\n[→] {len(sample_df)}개 파일 추론 시작 (모델: {MODEL_NAME})...")
    records = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), ncols=70):
        try:
            result = infer_single(model, row["filepath"], labels, top_k=3)
            top1_label, top1_prob = result["top_k"][0]
            records.append({
                "filename":       row["filename"],
                "esc50_category": row["category"],
                "top1_audioset":  top1_label,
                "top1_prob":      round(top1_prob, 4),
                **{f"score_{k}": round(v, 4)
                   for k, v in result["target_scores"].items()},
            })
        except Exception as e:
            print(f"  [!] 오류 {row['filename']}: {e}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 7. 결과 출력
# ─────────────────────────────────────────────
def print_results(results_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(f"  EfficientAT ({MODEL_NAME}) — ESC-50 추론 결과")
    print("=" * 70)

    score_cols = [c for c in results_df.columns if c.startswith("score_")]
    print("\n[위험음 클래스별 평균 confidence score]")
    print(f"  {'클래스':<15} {'평균 score':>12}")
    print("  " + "-" * 30)
    for col in score_cols:
        cls_name   = col.replace("score_", "")
        mean_score = results_df[col].mean()
        bar        = "█" * int(mean_score * 30)
        print(f"  {cls_name:<15} {mean_score:>8.4f}  {bar}")

    print(f"\n[샘플 추론 결과 (상위 10개)]")
    print(f"  {'ESC-50 카테고리':<22} {'Top-1 AudioSet 예측':<35} {'확률':>8}")
    print("  " + "-" * 68)
    for _, row in results_df.head(10).iterrows():
        cat  = row["esc50_category"][:21]
        pred = row["top1_audioset"][:34]
        prob = row["top1_prob"]
        print(f"  {cat:<22} {pred:<35} {prob:>8.4f}")

    out_path = RESULT_DIR / f"efficientat_{MODEL_NAME}_results.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] 결과 저장: {out_path}")
    print("=" * 70)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  EfficientAT ({MODEL_NAME}) × ESC-50 Inference Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # 1) ESC-50
    esc_dir = download_esc50()
    df      = load_esc50_metadata(esc_dir)

    # 2) AudioSet 레이블
    labels = load_audioset_labels()

    # 3) 모델 로드
    model = load_efficientat_model(MODEL_NAME)

    # 4) 단일 파일 예시
    sample_file = df.iloc[0]["filepath"]
    print(f"\n[→] 단일 파일 추론 예시: {Path(sample_file).name}")
    result = infer_single(model, sample_file, labels, top_k=5)
    print(f"  카테고리: {df.iloc[0]['category']}")
    print(f"  Top-5 예측:")
    for rank, (lbl, prob) in enumerate(result["top_k"], 1):
        print(f"    {rank}. {lbl:<40} {prob:.4f}")
    print(f"  위험음 scores: {result['target_scores']}")

    # 5) 배치 추론
    results_df = run_esc50_inference(model, df, labels, n_samples=N_SAMPLES)

    # 6) 결과 출력
    print_results(results_df)

    print("\n[완료] EfficientAT inference 파이프라인 실행 성공!")


if __name__ == "__main__":
    main()