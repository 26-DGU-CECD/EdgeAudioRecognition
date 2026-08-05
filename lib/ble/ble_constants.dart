import 'package:flutter_blue_plus/flutter_blue_plus.dart';

/// Jetson BLE advertising name.
/// Jetson Python code uses --name JHello by default.
const String jetsonDeviceName = 'JHello';

/// User-facing device name.
const String keyringDisplayName = 'miimo';

/// Jetson SERVICE_UUID.
/// This identifies the SoundKey BLE GATT service.
final Guid jetsonServiceUuid = Guid('12345678-1234-5678-1234-56789abcdef0');

/// Jetson HELLO_CHAR_UUID.
/// The app uses this characteristic as the SoundPacket JSON notify channel.
final Guid soundResultCharUuid = Guid('12345678-1234-5678-1234-56789abcdef1');

/// BLE 스캔 디버그 모드.
///
/// true 이면
/// - startScan에 withServices(하드웨어 필터)를 넣지 않는다
/// - 이름/서비스 UUID로 결과를 걸러내지 않는다
/// - 발견된 모든 기기의 platformName / advName / remoteId / serviceUuids를 로그로 남긴다
///
/// 라즈베리파이 광고가 정상적으로 잡히는 것을 확인한 뒤 false로 되돌린다.
const bool kBleScanDebugShowAll = false;
