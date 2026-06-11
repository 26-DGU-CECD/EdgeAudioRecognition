import 'package:flutter_blue_plus/flutter_blue_plus.dart';

/// Jetson BLE advertising name.
/// Jetson Python code uses --name JHello by default.
const String jetsonDeviceName = 'JHello';

/// User-facing device name.
const String keyringDisplayName = 'miimo 키링';

/// Jetson SERVICE_UUID.
/// This identifies the SoundKey BLE GATT service.
final Guid jetsonServiceUuid = Guid('12345678-1234-5678-1234-56789abcdef0');

/// Jetson HELLO_CHAR_UUID.
/// The app uses this characteristic as the SoundPacket JSON notify channel.
final Guid soundResultCharUuid = Guid('12345678-1234-5678-1234-56789abcdef1');
