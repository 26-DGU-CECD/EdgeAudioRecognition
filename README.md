# Sound Keyring

청각장애인을 위한 소리 인식 키링 연동 Flutter 앱입니다.  
Jetson Nano 기반 키링이 BLE GATT로 전송하는 소리 인식 결과를 앱에서 수신하고, 홈 화면/로그/설정 화면에 표시합니다.

## 주요 기능

- BLE GATT 기반 Sound Keyring 기기 연결
- Jetson Nano에서 전송한 소리 인식 JSON 수신
- 홈 화면에서 현재 감지된 소리 표시
  - 소리 종류
  - 신뢰도(score)
  - dB
  - 추론 시간
  - 방향 각도 기반 원형 포인터
- 로그 화면
  - 오늘, 이번주, 이번달 섹션 표시
- 설정 화면
  - 기기 정보 확인
  - 소리 종류별 알림 ON/OFF
  - 백그라운드 알림 설정 토글
  - 로컬 로그 초기화

## BLE 연동 정보

Jetson Nano BLE peripheral 설정과 앱의 UUID 값이 일치해야 합니다.

```text
Device Name: JHello
Service UUID: 12345678-1234-5678-1234-56789abcdef0
Result Characteristic UUID: 12345678-1234-5678-1234-56789abcdef1
앱은 Result Characteristic의 notify 값을 UTF-8 JSON 문자열로 수신합니다.

수신 예시:

{
  "status": "ok",
  "time": "12:14:53",
  "label": "dog_bark",
  "display_label": "개소리",
  "score": 0.998,
  "infer_sec": 0.118,
  "total_sec": 2.359,
  "db": 47.2,
  "level": "caution",
  "direction": "서",
  "angle": 233.0,
  "angle_raw": 233.0,
  "direction_text": "서쪽 233도",
  "doa_status": "enabled",
  "raw": "dog_bark score=0.998 db=47.2 doa=233",
  "items": []
}
프로젝트 구조
lib/
  main.dart
  main_page.dart
  home_page.dart
  log_page.dart
  settings_page.dart

  setting_device_info_page.dart
  setting_notification_page.dart

  sound_packet.dart
  device_status.dart
  detected_content.dart
  sound_visual.dart

  ble/
    ble_connection_page.dart
    ble_connection_store.dart
    ble_constants.dart
    ble_sound_service.dart
    connection_gate.dart
파일 설명
main.dart
앱의 진입점입니다.
MaterialApp을 생성하고 앱 제목, 테마, 시작 화면을 설정합니다.

현재 시작 화면은 ConnectionGate입니다.

home: const ConnectionGate()
즉 앱이 켜지면 바로 홈 화면으로 가지 않고, 저장된 BLE 연결 정보가 있는지 먼저 확인합니다.

ble/connection_gate.dart
앱 시작 시 연결 상태를 판단하는 화면입니다.

역할:

저장된 BLE 기기 정보 확인
저장된 기기가 있으면 자동 연결 시도
자동 연결 성공 시 MainPage로 이동
저장된 기기가 없거나 연결 실패 시 BleConnectionPage로 이동
앱의 첫 흐름을 제어하는 파일입니다.

ble/ble_connection_page.dart
BLE 기기 검색 및 연결 화면입니다.

역할:

주변 BLE 기기 스캔
Jetson Nano 키링 이름 또는 Service UUID 기준으로 필터링
사용자가 기기를 선택하면 BLE 연결 시도
연결 성공 시 MainPage로 이동
기기가 없어도 기기 연결 없이 앱 시작하기 -> 버튼으로 메인 화면 진입 가능
ble/ble_constants.dart
BLE 연결에 필요한 고정값을 관리합니다.

포함 내용:

Jetson BLE 기기 이름
Service UUID
Result Characteristic UUID
Jetson Nano Python 코드의 UUID와 반드시 일치해야 합니다.

ble/ble_sound_service.dart
BLE 연결과 데이터 수신을 담당하는 핵심 서비스입니다.

역할:

BLE 기기 연결
GATT Service 탐색
Result Characteristic 탐색
Characteristic notify 구독
UTF-8 byte 데이터를 문자열로 변환
JSON 파싱
SoundPacket 또는 DeviceStatus로 분류
Stream을 통해 앱 화면에 데이터 전달
수신된 소리 데이터는 아래 Stream으로 전달됩니다.

Stream<SoundPacket> get soundPackets
기기 상태 정보는 아래 Stream으로 전달됩니다.

Stream<DeviceStatus> get deviceStatuses
ble/ble_connection_store.dart
마지막으로 연결한 BLE 기기 정보를 로컬에 저장합니다.

사용 패키지:

shared_preferences
저장하는 정보:

BLE deviceId
BLE deviceName
앱을 다시 켰을 때 자동 연결을 시도하기 위해 사용합니다.

main_page.dart
앱의 메인 탭 구조와 전체 상태를 관리합니다.

역할:

하단 탭 관리
홈
로그
설정
현재 감지된 소리 상태 관리
전체 로그 목록 관리
BLE 수신 Stream 구독
소리별 알림 차단 목록 관리
백그라운드 알림 설정 상태 관리
로컬 로그 초기화 처리
BleSoundService에서 들어온 SoundPacket은 이 파일에서 받아서 홈과 로그에 반영됩니다.

home_page.dart
홈 화면 UI를 담당합니다.

역할:

현재 듣는 중 상태 표시
소리 감지 결과 표시
방향 원형 UI 표시
방향 각도에 따라 원 둘레 포인터 표시
infer, dB, score 정보 표시
테스트용 소리 수신 버튼 제공
이 파일 안에는 방향 원을 그리는 DirectionCircle, DirectionCirclePainter도 포함되어 있습니다.

detected_content.dart
홈 화면 중앙에 들어가는 콘텐츠를 담당합니다.

포함 내용:

ListeningContent
소리를 듣고 있는 기본 상태 UI
DetectedContent
소리 감지 시 중앙에 표시되는 박스 UI
소리 이미지/아이콘
소리 이름
score 표시
sound_visual.dart
소리 종류에 맞는 시각 요소를 표시합니다.

현재는 임시 이미지 또는 아이콘 구조로 사용하며, 추후 실제 asset 이미지로 교체할 수 있습니다.

역할:

asset 이미지 표시
이미지가 없거나 실패하면 placeholder 표시
sound_packet.dart
Jetson Nano에서 전송하는 소리 인식 JSON을 Dart 객체로 변환하는 모델 파일입니다.

포함 클래스:

SoundPacket
TopKItem
SoundPacket 주요 필드:

status
time
label
displayLabel
score
inferSec
totalSec
db
level
direction
angle
angleRaw
directionText
doaStatus
raw
items
또한 홈 화면에 표시할지 판단하는 조건도 포함합니다.

bool get isDisplayable {
  return status == 'ok' && score >= 0.7 && db >= 45.0;
}
device_status.dart
키링 또는 Jetson Nano의 상태 정보를 담는 모델 파일입니다.

포함 정보:

연결 상태
기기 이름
배터리
상태 메시지
기기 정보 설정 페이지에서 표시됩니다.

log_page.dart
로그 탭 화면입니다.

역할:

수신된 SoundPacket 목록 표시
로그를 오늘, 이번주, 이번달 섹션으로 구분 표시
현재 SoundPacket에는 날짜 정보가 없고 시간 문자열만 있으므로, 실제 날짜 분류는 아직 제한적입니다.
추후 로그 저장 시 DateTime receivedAt 같은 필드를 추가하면 정확한 날짜별 분류가 가능합니다.

settings_page.dart
설정 탭의 메인 목록 화면입니다.

역할:

기기 정보 페이지로 이동
알림 설정 페이지로 이동
백그라운드 기기 알림 토글
로컬 로그 초기화
실제 휴대폰 설정 앱처럼 항목을 누르면 세부 페이지로 이동하는 구조입니다.

setting_device_info_page.dart
기기 정보 상세 페이지입니다.

표시 항목:

연결 상태
기기 이름
배터리
상태 메시지
설정 탭에서 기기 정보 항목을 누르면 이동합니다.

setting_notification_page.dart
알림 설정 상세 페이지입니다.

역할:

소리 종류 목록 표시
각 소리별 알림 ON/OFF 설정
OFF로 설정한 소리는 홈 화면 알림으로 표시되지 않음
로그에는 계속 기록됨
대상 소리 종류:

총
경보
자전거
물소리
울음
비명
유리깨지는소리
화재경보
아기 우는 소리
개소리
고양이소리
실행 방법
cd C:\CookAndroid\untitled
C:\Users\stell\flutter\bin\flutter.bat pub get
C:\Users\stell\flutter\bin\flutter.bat run
Android 실기기에서 테스트하려면:

개발자 옵션 활성화
USB 디버깅 켜기
PC와 휴대폰 USB 연결
Android Studio 또는 flutter run으로 설치
Android 권한
BLE 스캔 및 연결을 위해 Android 권한이 필요합니다.

<uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" android:maxSdkVersion="30" />
참고
현재 Wi-Fi 연결 페이지는 사용하지 않습니다.
백그라운드 지속 수신은 Android Foreground Service 연동이 추가로 필요합니다.
iOS 설치는 Mac과 Xcode가 필요합니다.
소리별 이미지는 추후 실제 asset 이미지로 교체할 수 있습니다.
