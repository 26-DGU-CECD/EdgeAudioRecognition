import 'dart:async';
import 'dart:convert';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'ble_constants.dart';
import 'ble_connection_store.dart';
import 'package:untitled/models/device_status.dart';
import 'package:untitled/models/sound_packet.dart';

/// BLE 연결, notify 구독, json 파싱
class BleSoundService {
  BleSoundService._();

  static final BleSoundService instance = BleSoundService._();

  BluetoothDevice? _device;

  StreamSubscription<List<int>>? _valueSub;
  StreamSubscription<BluetoothConnectionState>? _connectionSub;

  final StreamController<SoundPacket> _soundPacketController =
      StreamController<SoundPacket>.broadcast();
  final StreamController<DeviceStatus> _deviceStatusController =
      StreamController<DeviceStatus>.broadcast();
  final StreamController<String> _logController =
      StreamController<String>.broadcast();

  Stream<SoundPacket> get soundPackets => _soundPacketController.stream;

  Stream<DeviceStatus> get deviceStatuses => _deviceStatusController.stream;

  Stream<String> get logs => _logController.stream;

  BluetoothDevice? get device => _device;

  bool get isConnected => _device != null;

  static SoundPacket? soundPacketFromBleJson(Map<String, dynamic> json) {
    final type = json['type'];

    if (type == 'sound_packet') {
      final payload = json['payload'];
      if (payload is Map) {
        return SoundPacket.fromJson(_stringKeyMap(payload));
      }
      return null;
    }

    if (json['status'] == 'ok') {
      return SoundPacket.fromJson(json);
    }

    return null;
  }

  static DeviceStatus? deviceStatusFromBleJson(Map<String, dynamic> json) {
    if (json['type'] == 'device_status') {
      return DeviceStatus.fromJson(json);
    }

    return null;
  }

  /// 사용자가 BLE 연결 페이지에서 선택한 기기에 연결
  Future<void> connect(BluetoothDevice device, {String? deviceName}) async {
    await disconnect();

    _device = device;
    _log('BLE 연결: ${deviceName ?? device.remoteId.str}');

    await device.connect(
      license: License.nonprofit,
      autoConnect: false,
      timeout: const Duration(seconds: 12),
    );

    _connectionSub = device.connectionState.listen((state) {
      _log('BLE 상태: $state');

      if (state == BluetoothConnectionState.disconnected) {
        if (_device == device) {
          _device = null;
        }
        unawaited(_valueSub?.cancel());
        _valueSub = null;

        _deviceStatusController.add(
          DeviceStatus(
            connection: 'disconnected',
            deviceName: deviceName ?? device.remoteId.str,
            message: 'Bluetooth 연결이 끊어졌습니다.',
          ),
        );
      }
    });

    await _discoverResultCharacteristic(device);

    await BleConnectionStore.save(
      deviceId: device.remoteId.str,
      deviceName: deviceName ?? device.remoteId.str,
    );

    _deviceStatusController.add(
      DeviceStatus(
        connection: 'connected',
        deviceName: deviceName ?? device.remoteId.str,
        message: 'Bluetooth 연결 완료',
      ),
    );

    _log('Bluetooth 연결 완료');
  }

  /// 저장된 deviceId로 자동 재연결
  /// 실패 시 false를 반환하고 블루투스 연결 페이지로 연결
  Future<bool> connectSavedDevice() async {
    final savedDeviceId = await BleConnectionStore.loadDeviceId();

    if (savedDeviceId == null) {
      return false;
    }

    try {
      final device = BluetoothDevice.fromId(savedDeviceId);
      await connect(device, deviceName: keyringDisplayName);
      return true;
    } catch (e) {
      _log('자동 연결 실패: $e');
      return false;
    }
  }

  /// SoundKey Service와 Result Characteristic을 찾고 notify 켜기
  Future<void> _discoverResultCharacteristic(BluetoothDevice device) async {
    final services = await device.discoverServices();

    BluetoothCharacteristic? target;

    for (final service in services) {
      if (service.uuid == jetsonServiceUuid) {
        for (final characteristic in service.characteristics) {
          if (characteristic.uuid == soundResultCharUuid) {
            target = characteristic;
            break;
          }
        }
      }
    }

    if (target == null) {
      throw Exception('SoundKey result characteristic을 찾지 못했습니다.');
    }

    _valueSub?.cancel();
    _valueSub = target.onValueReceived.listen(_handleBytes);

    /// read면 현재 값을 한번 읽음
    if (target.properties.read) {
      final value = await target.read();
      _handleBytes(value);
    }

    /// notify면 이후 패킷 자동 수신
    if (target.properties.notify) {
      await target.setNotifyValue(true);
    } else {
      throw Exception('Result characteristic이 notify를 지원하지 않습니다.');
    }
  }

  /// Jetson에서 받은 byte 배열을 UTF-8 JSON으로 해석
  void _handleBytes(List<int> bytes) {
    if (bytes.isEmpty) return;

    final text = utf8.decode(bytes, allowMalformed: true);
    _log('BLE 수신: $text');

    try {
      final decoded = jsonDecode(text);

      if (decoded is! Map<String, dynamic>) {
        _log('JSON 객체가 아닙니다.');
        return;
      }

      _handleJson(decoded);
    } catch (e) {
      /// Jetson 테스트 코드처럼 "안녕 1" 같은 일반 문자열이 오면 여기로 옵니다.
      _deviceStatusController.add(
        DeviceStatus(
          connection: 'connected',
          deviceName: _device == null ? '' : keyringDisplayName,
          message: text,
        ),
      );
    }
  }

  /// JSON 메시지 타입에 따라 SoundPacket 또는 DeviceStatus로 분리
  void _handleJson(Map<String, dynamic> json) {
    final status = deviceStatusFromBleJson(json);
    if (status != null) {
      _deviceStatusController.add(status);
      return;
    }

    final packet = soundPacketFromBleJson(json);
    if (packet != null) {
      _soundPacketController.add(packet);
      return;
    }

    _log('알 수 없는 BLE JSON 형식: $json');
  }

  Future<void> disconnect() async {
    await _valueSub?.cancel();
    await _connectionSub?.cancel();

    final device = _device;
    _device = null;

    if (device != null) {
      try {
        await device.disconnect();
      } catch (_) {}
    }
  }

  void _log(String message) {
    _logController.add(message);
  }

  static Map<String, dynamic> _stringKeyMap(Map source) {
    return source.map((key, value) => MapEntry(key.toString(), value));
  }
}
