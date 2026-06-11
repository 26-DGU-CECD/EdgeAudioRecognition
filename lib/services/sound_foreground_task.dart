import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui';

import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';

import '../ble/ble_connection_store.dart';
import '../ble/ble_constants.dart';
import '../ble/ble_sound_service.dart';
import '../models/device_status.dart';
import '../models/sound_packet.dart';
import 'alert_rule_service.dart';
import 'app_permission_service.dart';
import 'local_notification_service.dart';

const String foregroundNotificationTitle = '환경음 감지 디바이스 연결 중';
const String foregroundNotificationText = 'Jetson에서 위험음 알림을 수신하고 있습니다.';

@pragma('vm:entry-point')
void startSoundForegroundTask() {
  FlutterForegroundTask.setTaskHandler(SoundForegroundTaskHandler());
}

class SoundForegroundServiceController {
  static void initialize() {
    if (!Platform.isAndroid) {
      return;
    }

    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'sound_keyring_foreground',
        channelName: 'Sound Keyring foreground service',
        channelDescription:
            'Shows while Sound Keyring is connected to the Jetson device.',
        channelImportance: NotificationChannelImportance.LOW,
        priority: NotificationPriority.LOW,
        onlyAlertOnce: true,
        showWhen: false,
      ),
      iosNotificationOptions: const IOSNotificationOptions(
        showNotification: false,
        playSound: false,
      ),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.repeat(5000),
        autoRunOnBoot: false,
        autoRunOnMyPackageReplaced: false,
        allowWakeLock: true,
        allowWifiLock: false,
        allowAutoRestart: true,
        stopWithTask: false,
      ),
    );
  }

  static Future<bool> start() async {
    if (!Platform.isAndroid) {
      return false;
    }

    initialize();
    await AppPermissionService.requestRuntimePermissions();

    if (await FlutterForegroundTask.isRunningService) {
      return true;
    }

    final result = await FlutterForegroundTask.startService(
      serviceId: 1001,
      serviceTypes: const [ForegroundServiceTypes.connectedDevice],
      notificationTitle: foregroundNotificationTitle,
      notificationText: foregroundNotificationText,
      callback: startSoundForegroundTask,
    );

    return result is ServiceRequestSuccess;
  }

  static Future<void> stop() async {
    if (!Platform.isAndroid) {
      return;
    }

    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.stopService();
    }
  }

  static Future<bool> isRunning() async {
    if (!Platform.isAndroid) {
      return false;
    }
    return FlutterForegroundTask.isRunningService;
  }
}

class SoundForegroundTaskHandler extends TaskHandler {
  final AlertRuleService _alertRules = AlertRuleService();

  BluetoothDevice? _device;
  StreamSubscription<List<int>>? _valueSub;
  StreamSubscription<BluetoothConnectionState>? _connectionSub;

  bool _connecting = false;
  DateTime? _lastReconnectAttempt;

  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    DartPluginRegistrant.ensureInitialized();
    await LocalNotificationService.instance.initialize();
    await _connectSavedDevice();
  }

  @override
  void onRepeatEvent(DateTime timestamp) {
    unawaited(_ensureConnected());
  }

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    await _disconnect();
  }

  @override
  void onReceiveData(Object data) {
    if (data == 'stop') {
      unawaited(_disconnect());
    }
  }

  Future<void> _ensureConnected() async {
    if (_connecting || _device != null) {
      return;
    }

    final now = DateTime.now();
    final lastAttempt = _lastReconnectAttempt;
    if (lastAttempt != null && now.difference(lastAttempt).inSeconds < 5) {
      return;
    }

    _lastReconnectAttempt = now;
    await _connectSavedDevice();
  }

  Future<void> _connectSavedDevice() async {
    final savedDeviceId = await BleConnectionStore.loadDeviceId();

    if (savedDeviceId == null || savedDeviceId.isEmpty) {
      _sendStatus(
        const DeviceStatus(
          connection: 'disconnected',
          deviceName: '',
          message: '저장된 Bluetooth 기기가 없습니다.',
        ),
      );
      return;
    }

    const deviceName = keyringDisplayName;
    _connecting = true;

    try {
      await _disconnect();

      final device = BluetoothDevice.fromId(savedDeviceId);
      _device = device;

      await device.connect(
        license: License.nonprofit,
        autoConnect: false,
        timeout: const Duration(seconds: 12),
      );

      _connectionSub = device.connectionState.listen((state) {
        if (state == BluetoothConnectionState.disconnected) {
          _device = null;
          unawaited(_valueSub?.cancel());
          _valueSub = null;
          _sendStatus(
            DeviceStatus(
              connection: 'disconnected',
              deviceName: deviceName,
              message: 'Foreground Service BLE 연결이 끊어졌습니다.',
            ),
          );
        }
      });

      await _discoverResultCharacteristic(device);

      _sendStatus(
        DeviceStatus(
          connection: 'connected',
          deviceName: deviceName,
          message: 'Foreground Service BLE 연결 완료',
        ),
      );
    } catch (e) {
      _device = null;
      _sendStatus(
        DeviceStatus(
          connection: 'disconnected',
          deviceName: deviceName,
          message: 'Foreground Service BLE 연결 실패: $e',
        ),
      );
    } finally {
      _connecting = false;
    }
  }

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

    await _valueSub?.cancel();
    _valueSub = target.onValueReceived.listen(_handleBytes);

    if (target.properties.read) {
      final value = await target.read();
      _handleBytes(value);
    }

    if (target.properties.notify) {
      await target.setNotifyValue(true);
    } else {
      throw Exception('Result characteristic이 notify를 지원하지 않습니다.');
    }
  }

  void _handleBytes(List<int> bytes) {
    if (bytes.isEmpty) {
      return;
    }

    final text = utf8.decode(bytes, allowMalformed: true);

    try {
      final decoded = jsonDecode(text);
      if (decoded is! Map) {
        return;
      }

      final json = _stringKeyMap(decoded);
      final status = BleSoundService.deviceStatusFromBleJson(json);
      if (status != null) {
        _sendStatus(status);
        return;
      }

      final packet = BleSoundService.soundPacketFromBleJson(json);
      if (packet == null) {
        return;
      }

      _sendPacket(packet);
      unawaited(_maybeNotify(packet));
    } catch (_) {
      _sendStatus(
        DeviceStatus(
          connection: 'connected',
          deviceName: _device == null ? '' : keyringDisplayName,
          message: text,
        ),
      );
    }
  }

  Future<void> _maybeNotify(SoundPacket packet) async {
    if (!await _alertRules.shouldNotify(packet)) {
      return;
    }

    await LocalNotificationService.instance.showSoundAlert(
      title: _alertRules.titleFor(packet),
      body: _alertRules.bodyFor(packet),
      labelKey: packet.notificationLabelKey,
    );
  }

  void _sendPacket(SoundPacket packet) {
    FlutterForegroundTask.sendDataToMain({
      'type': 'sound_packet',
      'payload': packet.toJson(),
    });
  }

  void _sendStatus(DeviceStatus status) {
    FlutterForegroundTask.sendDataToMain({
      'type': 'device_status',
      'payload': status.toJson(),
    });
  }

  Future<void> _disconnect({bool disconnectDevice = true}) async {
    await _valueSub?.cancel();
    _valueSub = null;
    await _connectionSub?.cancel();
    _connectionSub = null;

    final device = _device;
    _device = null;

    if (disconnectDevice && device != null) {
      try {
        await device.disconnect();
      } catch (_) {}
    }
  }

  Map<String, dynamic> _stringKeyMap(Map source) {
    return source.map((key, value) => MapEntry(key.toString(), value));
  }
}
