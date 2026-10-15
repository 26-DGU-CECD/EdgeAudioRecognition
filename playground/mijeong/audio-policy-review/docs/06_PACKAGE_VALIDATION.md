# 공유 스냅샷 검증 기록

정리된 폴더에서 실제 audio-policy 환경을 사용해 다음을 확인했다.

- 원본 v1.4 테스트 42개 통과.
- 새 project 경로에서 정적 9개 결과를 재현해 기존 expanded 정적 결과와 모든 CSV 값 일치.
- 새 results/followup 경로에서 적응형 FIFO 12개 결과를 재현해 기존 결과와 모든 CSV 값 일치.
- 모든 수집 파일의 SHA256 검증 통과.
- 별도 로컬 Git index에서 core.autocrlf=true 조건으로 .pt/.npz LFS pointer와 원본 SHA256 확인.
- config/manifest JSON이 Git index를 통과해도 원본 byte와 동일함을 확인. .gitattributes의 -text 설정은 fingerprint 보존을 위한 것이다.
- 개인 결과, embedding_cache, WAV가 Git에서 제외되는지 확인.
- 폴더 생성 PowerShell 스크립트 문법 검사 통과.

GitHub 저장소 생성·네트워크 인증·push는 실행하지 않았다. 실제 학습이나 새 원음 GPU 추론도 실행하지 않았다. 당시 체크포인트와 저장 예측으로 재현성을 확인한 기록이다.

검증 때 생성한 .local/check_* 결과는 Git에서 제외된다. 파일별 검증 대상은 manifests/files.sha256.json이며, 의도적인 수정 후 scripts/update_inventory.py로 갱신한다.
