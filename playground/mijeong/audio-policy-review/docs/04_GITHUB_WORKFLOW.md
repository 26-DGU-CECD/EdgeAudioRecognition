# GitHub 공유 절차 — Windows / VS Code

이 절차는 준비된 폴더를 GitHub에 게시하는 방법이다. 수집 스크립트는 GitHub 저장소 생성이나 push를 수행하지 않는다. 명령의 저장소 URL과 사용자 정보는 실제 값으로 바꾼다.

GitHub는 일반 Git의 100MiB 초과 파일을 차단한다. 모델과 npz 예측은 준비된 `.gitattributes`에 따라 Git LFS로 추적한다. LFS 용량·전송량은 GitHub 계정/조직의 사용 정책을 확인한다.

공식 문서: [대용량 파일](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [Git LFS 설정](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage), [로컬 코드 게시](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## 1. 새 공유 폴더 생성

원 작성자 PC에서 Conda audio-policy를 활성화한다. 수집 kit의 실제 경로를 사용한다.

```powershell
conda activate audio-policy
$kit = 'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\github_team_share_kit'
$share = 'C:\Users\PLASS\Desktop\audio_policy_team_share'
& "$kit\New-TeamShare.ps1" -Destination "$share"
Set-Location $share
code .
python scripts/verify_share.py
```

이미 파일이 있는 대상에는 수집 스크립트가 덮어쓰지 않는다. 다른 새 폴더명을 사용한다. PowerShell이 실행 정책으로 ps1을 차단하면 영구 정책을 바꾸지 않고 다음 Python 명령으로 같은 작업을 할 수 있다.

```powershell
python "$kit\build_team_share.py" --destination "$share"
```

## 2A. 새 GitHub 저장소에 첫 push

GitHub에서 팀 저장소를 생성하고 URL을 복사한다. 팀 내부 공유라면 Private으로 생성한다. 새 저장소를 README/.gitignore/license로 초기화하지 않아야 아래 절차가 간단하다.

Private 저장소에서는 Settings의 Collaborators/팀 접근 설정으로 팀원에게 읽기 또는 쓰기 권한을 부여한다. 조직 저장소는 조직의 팀 권한을 사용한다.

```powershell
Set-Location $share
git init -b main
git lfs version
git lfs install --local
git add .gitattributes .gitignore
git add .
git lfs ls-files
git diff --cached --stat
git status --short
```

목록을 확인한 뒤 commit과 push를 실행한다. 원음과 embedding_cache는 목록에 없어야 하고, 모델과 npz는 LFS 목록에 있어야 한다.

```powershell
git commit -m "Share audio policy experiments through v1.4"
$repoUrl = 'https://github.com/YOUR_ORG/YOUR_REPOSITORY.git'
git remote add origin $repoUrl
git push -u origin main
```

Git이 작성자 정보를 요구할 때만 저장소에 본인의 이름·등록 이메일 또는 GitHub noreply 이메일을 설정한다.

```powershell
git config user.name 'YOUR_NAME'
git config user.email 'YOUR_GITHUB_EMAIL'
```

인증은 Git Credential Manager의 브라우저 로그인 또는 조직에서 사용하는 SSH 방식을 따른다. 토큰을 코드나 remote URL에 적지 않는다. 최초 push에서는 원격에 아직 이력이 없으므로 pull을 먼저 하지 않는다.

## 2B. 이미 팀 저장소에 README/commit이 있는 경우

2A와 섞지 않는다. 기존 저장소를 clone하고 새 branch에서 하위 폴더로 수집한다. 기존 README와 이력을 유지한다.

```powershell
$repoUrl = 'https://github.com/YOUR_ORG/YOUR_REPOSITORY.git'
$clone = 'C:\Users\PLASS\Desktop\team_audio_repository'
git clone $repoUrl $clone
Set-Location $clone
git lfs install --local
git switch -c share/audio-policy-v14
& "$kit\New-TeamShare.ps1" -Destination (Join-Path $clone 'audio-policy-review')
git add audio-policy-review
git lfs ls-files
git diff --cached --stat
git commit -m "Add audio policy v1.4 experiment snapshot"
git push -u origin HEAD
```

그 후 GitHub에서 팀의 기본 branch를 대상으로 Pull Request를 만든다. 이 절차에서는 기존 저장소의 기본 branch 이름을 main으로 가정하지 않는다.

## 3. 팀원이 처음 받기 — pull이 아니라 clone

```powershell
$repoUrl = 'https://github.com/YOUR_ORG/YOUR_REPOSITORY.git'
git clone $repoUrl
Set-Location YOUR_REPOSITORY
git lfs install --local
git lfs pull
code .
```

기존 팀 저장소 하위의 audio-policy-review로 추가했다면 그 폴더로 이동한 뒤 환경·검증 명령을 실행한다. 프로젝트 폴더에서 다음 순서로 시작한다.

```powershell
conda create -n audio-policy python=3.11 pip -y
conda activate audio-policy
python -m pip install -r environments/requirements-policy.txt
python scripts/verify_share.py
python scripts/replay_cached.py --stage adaptive --timing independent_worker_sim
```

## 4. 이후 팀 작업 — pull → branch → 수정 → push

아래 main은 새 저장소를 main으로 생성한 경우다. 기존 팀 저장소는 실제 기본 branch 이름으로 바꾼다.

```powershell
git status
git switch main
git pull --ff-only origin main
git lfs pull
git switch -c experiment/YOUR_NAME-adaptive-review
```

분석/수정 후 공유할 결과만 results/team_runs 아래로 복사한다. 새로 추가한 파일의 목록과 SHA를 갱신한 뒤 commit한다.

```powershell
python scripts/update_inventory.py
python scripts/verify_share.py
git add .
git lfs ls-files
git diff --cached --stat
git commit -m "Add adaptive policy review results"
git push -u origin HEAD
```

GitHub에서 Pull Request를 만들어 팀 검토 후 반영한다. 다음 작업 전 기본 branch에서 다시 pull한다. 이미 공유한 branch를 강제로 덮어쓰지 않는다.

## 자주 생기는 문제

- verify_share가 LFS pointer를 보고함: clone 안에서 `git lfs install --local`, `git lfs pull`.
- verify_share가 변경을 보고함: 의도한 편집인지 먼저 확인한다. 의도한 변경일 때만 update_inventory로 기록을 갱신한다.
- 100MiB 오류: embedding_cache가 실수로 staged 됐는지와 `git lfs ls-files`를 확인한다. commit 전에 해결한다. 이 패키지는 해당 cache를 기본 수집에서 제외한다.
- `pull --ff-only` 실패: 로컬·원격 이력이 갈라진 상태다. 작업을 commit한 뒤 별도 branch에서 `git fetch origin`, `git merge origin/main`으로 통합하고 충돌 파일을 검토한다. main은 실제 기본 branch로 바꾼다.
- `push`가 거절됨: 같은 branch의 원격 변경을 먼저 통합한다. 기본 branch 보호 규칙이면 새 branch에 push하고 PR을 사용한다.
- 환경 명령 실패: `python -c "import sys; print(sys.executable)"`로 Conda Python 3인지 확인한다. 원 작성자 PC의 기본 PATH에는 Python 2.7도 있었다.
