# 기존에 생성된 결과를 같은 설정으로 재현합니다. 모델 추론은 하지 않습니다.
conda activate audio-policy
$pilotCode = "C:\Users\PLASS\Desktop\audio-policy-review-20261015T164106Z-1-001\audio-policy-review\audio_policy_pilot_v1_4"
$pilotOut = "C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\v1_4_expanded_next\experiment"
Set-Location $pilotCode

python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I1 I2 I3 --temporal T0 --timing audio_clock --async-policy wait_all --result-tag B_adaptive_audio
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I1 I2 I3 --temporal T0 --timing independent_worker_sim --async-policy wait_all --result-tag B_adaptive_fifo
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 I3 --temporal T0 T1 T2 --timing audio_clock --async-policy wait_all --result-tag C_temporal_audio
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 I3 --temporal T0 T1 T2 --timing independent_worker_sim --async-policy wait_all --result-tag C_temporal_fifo
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 --temporal T0 --timing independent_worker_sim --async-policy wait_all --result-tag D_wait_all
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 --temporal T0 --timing independent_worker_sim --async-policy bounded --result-tag D_bounded
