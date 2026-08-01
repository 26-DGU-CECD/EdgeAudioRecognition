/// 키링의 상태 정보 패킷
class DeviceStatus {
  final String connection;
  final String deviceName;
  final int? battery;
  final String message;

  const DeviceStatus({
    required this.connection,
    required this.deviceName,
    this.battery,
    this.message = '',
  });

  factory DeviceStatus.fromJson(Map<String, dynamic> json) {
    return DeviceStatus(
      connection: json['connection']?.toString() ?? 'unknown',
      deviceName:
          (json['device_name'] ?? json['deviceName'])?.toString() ?? '',
      battery: parseBatteryPercent(
        json['battery_percent'] ?? json['battery'],
      ),
      message: json['message']?.toString() ?? '',
    );
  }

  /// BLE JSON의 배터리 값을 0~100 범위의 정수 퍼센트로 정규화합니다.
  /// Raspberry Pi가 정수, 실수, 문자열 중 어떤 형태로 보내도 처리합니다.
  static int? parseBatteryPercent(dynamic value) {
    final num? parsed = switch (value) {
      num number => number,
      String text => num.tryParse(text.trim()),
      _ => null,
    };

    if (parsed == null) {
      return null;
    }

    return parsed.round().clamp(0, 100).toInt();
  }

  Map<String, dynamic> toJson() {
    return {
      'connection': connection,
      'device_name': deviceName,
      'battery': battery,
      'message': message,
    };
  }
}
