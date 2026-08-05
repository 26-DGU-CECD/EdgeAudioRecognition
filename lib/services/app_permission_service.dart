import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';

import 'local_notification_service.dart';

/// BLE 스캔 직전에 권한 상태를 확인한 결과.
class BlePermissionReport {
  const BlePermissionReport({
    required this.statuses,
    required this.locationServiceEnabled,
  });

  /// 권한 이름 -> 상태 문자열
  final Map<String, PermissionStatus> statuses;

  /// Android 11 이하에서 BLE 스캔에 필요한 위치 서비스(GPS) on/off
  final bool locationServiceEnabled;

  /// 위치 권한 허용 여부.
  /// Android 11 이하에서는 이게 false면 BLE 스캔 결과가 항상 0건이다.
  bool get locationGranted => statuses[_kLocation]?.isGranted ?? false;

  /// 스캔에 반드시 필요한 권한이 모두 허용됐는지.
  /// Android 12 미만에서 bluetoothScan/Connect는 permission_handler가
  /// granted로 돌려주므로, 그 경우 locationWhenInUse가 실질 조건이 된다.
  bool get canScan {
    final scan = statuses[_kScan];
    final connect = statuses[_kConnect];
    final location = statuses[_kLocation];

    final scanOk = scan == null || scan.isGranted;
    final connectOk = connect == null || connect.isGranted;
    // Android 12+ 에서는 neverForLocation 플래그 덕분에 위치 권한이 없어도 스캔된다.
    // 위치 권한이 거부돼도 scan/connect가 허용이면 통과시킨다.
    final locationOk = scanOk || (location != null && location.isGranted);

    return scanOk && connectOk && locationOk;
  }

  @override
  String toString() {
    final buffer = StringBuffer('[BLE-PERM] ');
    statuses.forEach((name, status) {
      buffer.write('$name=${status.name} ');
    });
    buffer.write('locationService=');
    buffer.write(locationServiceEnabled ? 'enabled' : 'disabled');
    buffer.write(' canScan=$canScan');
    return buffer.toString();
  }

  static const String _kScan = 'BLUETOOTH_SCAN';
  static const String _kConnect = 'BLUETOOTH_CONNECT';
  static const String _kLocation = 'LOCATION_WHEN_IN_USE';
}

class AppPermissionService {
  static Future<void> requestRuntimePermissions() async {
    if (kIsWeb || !Platform.isAndroid) {
      return;
    }

    await [
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      Permission.locationWhenInUse,
      Permission.notification,
    ].request();

    await LocalNotificationService.instance
        .requestAndroidNotificationPermission();
  }

  /// BLE 스캔 직전에 호출한다.
  /// 스캔에 필요한 권한만 요청하고, 결과를 진단용으로 돌려준다.
  static Future<BlePermissionReport> ensureBlePermissions() async {
    if (kIsWeb || !Platform.isAndroid) {
      return const BlePermissionReport(
        statuses: <String, PermissionStatus>{},
        locationServiceEnabled: true,
      );
    }

    final requested = await [
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      // Android 11 이하에서는 위치 권한이 없으면 BLE 스캔 결과가 0건으로 나온다.
      Permission.locationWhenInUse,
    ].request();

    final locationServiceEnabled =
        await Permission.location.serviceStatus.isEnabled;

    final report = BlePermissionReport(
      statuses: <String, PermissionStatus>{
        BlePermissionReport._kScan:
            requested[Permission.bluetoothScan] ?? PermissionStatus.denied,
        BlePermissionReport._kConnect:
            requested[Permission.bluetoothConnect] ?? PermissionStatus.denied,
        BlePermissionReport._kLocation:
            requested[Permission.locationWhenInUse] ?? PermissionStatus.denied,
      },
      locationServiceEnabled: locationServiceEnabled,
    );

    debugPrint(report.toString());

    return report;
  }
}
