import 'package:flutter/cupertino.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// BLE 자동 연결에 필요한 정보 저장
class BleConnectionStore {
  static const _deviceIdKey = 'ble_device_id';
  static const _deviceNameKey = 'ble_device_name';

  static Future<void> save({
    required String deviceId,
    required String deviceName,
}) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_deviceIdKey, deviceId);
    await prefs.setString(_deviceNameKey, deviceName);
  }

  static Future<String?> loadDeviceId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_deviceIdKey);
  }

  static Future<String?> loadDeviceName() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_deviceNameKey);
  }

  static Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_deviceIdKey);
    await prefs.remove(_deviceNameKey);
  }
}