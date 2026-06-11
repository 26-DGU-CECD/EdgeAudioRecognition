import 'dart:io';

import 'package:permission_handler/permission_handler.dart';

import 'local_notification_service.dart';

class AppPermissionService {
  static Future<void> requestRuntimePermissions() async {
    if (!Platform.isAndroid) {
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
}
