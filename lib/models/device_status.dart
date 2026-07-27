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
      connection: json['connection'] ?? 'unknown',
      deviceName: json['device_name'] ?? json['deviceName'] ?? '',
      battery: json['battery'],
      message: json['message'] ?? '',
    );
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
