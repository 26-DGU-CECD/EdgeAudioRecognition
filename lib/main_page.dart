import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';

import 'ble/ble_connection_page.dart';
import 'ble/ble_sound_service.dart';
import 'models/device_status.dart';
import 'models/sound_packet.dart';
import 'pages/home_page.dart';
import 'pages/log_page.dart';
import 'pages/settings_page.dart';
import 'services/alert_settings_store.dart';
import 'services/sound_foreground_task.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key, required this.title});

  final String title;

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  static const double minMainDisplayScore = 0.30;

  int selectedIndex = 0;
  SoundPacket? currentPacket;
  DeviceStatus? deviceStatus;

  final List<SoundPacket> logs = [];
  Set<String> mutedLabelKeys = {};

  bool backgroundAlertsEnabled = false;

  StreamSubscription<SoundPacket>? _soundSub;
  StreamSubscription<DeviceStatus>? _statusSub;

  @override
  void initState() {
    super.initState();
    _loadSavedSettings();
    FlutterForegroundTask.addTaskDataCallback(_onForegroundTaskData);

    _soundSub = BleSoundService.instance.soundPackets.listen((packet) {
      setState(() {
        _addPacket(packet);
      });
    });

    _statusSub = BleSoundService.instance.deviceStatuses.listen((status) {
      setState(() {
        deviceStatus = status;
      });
    });
  }

  @override
  void dispose() {
    FlutterForegroundTask.removeTaskDataCallback(_onForegroundTaskData);
    _soundSub?.cancel();
    _statusSub?.cancel();
    super.dispose();
  }

  Future<void> _loadSavedSettings() async {
    final savedBackgroundAlertsEnabled =
        await AlertSettingsStore.loadBackgroundAlertsEnabled();
    final savedMutedLabelKeys = await AlertSettingsStore.loadMutedLabelKeys();

    if (!mounted) return;

    setState(() {
      backgroundAlertsEnabled = savedBackgroundAlertsEnabled;
      mutedLabelKeys = savedMutedLabelKeys;
    });

    if (savedBackgroundAlertsEnabled) {
      await SoundForegroundServiceController.start();
    }
  }

  void _onForegroundTaskData(Object data) {
    if (!mounted || data is! Map) {
      return;
    }

    final payload = data['payload'];
    if (payload is! Map) {
      return;
    }

    final payloadJson = _stringKeyMap(payload);

    switch (data['type']) {
      case 'sound_packet':
        setState(() {
          _addPacket(SoundPacket.fromJson(payloadJson));
        });
        break;
      case 'device_status':
        setState(() {
          deviceStatus = DeviceStatus.fromJson(payloadJson);
        });
        break;
    }
  }

  void _addPacket(SoundPacket packet) {
    logs.insert(0, packet);

    final isMuted = mutedLabelKeys.contains(packet.notificationLabelKey);
    final shouldDisplayOnMain = packet.score >= minMainDisplayScore && !isMuted;
    currentPacket = shouldDisplayOnMain ? packet : null;
  }

  Future<void> toggleMutedLabel(String label) async {
    final labelKey = AlertSettingsStore.labelKeyForDisplayLabel(label);

    setState(() {
      if (mutedLabelKeys.contains(labelKey)) {
        mutedLabelKeys.remove(labelKey);
      } else {
        mutedLabelKeys.add(labelKey);
        if (currentPacket?.notificationLabelKey == labelKey) {
          currentPacket = null;
        }
      }
    });

    await AlertSettingsStore.saveMutedLabelKeys(mutedLabelKeys);
  }

  Future<void> toggleBackgroundAlerts(bool enabled) async {
    setState(() {
      backgroundAlertsEnabled = enabled;
    });

    await AlertSettingsStore.saveBackgroundAlertsEnabled(enabled);

    if (enabled) {
      await BleSoundService.instance.disconnect();
      final started = await SoundForegroundServiceController.start();

      if (!started && mounted) {
        await AlertSettingsStore.saveBackgroundAlertsEnabled(false);
        setState(() {
          backgroundAlertsEnabled = false;
        });
        unawaited(BleSoundService.instance.connectSavedDevice());
      }
      return;
    }

    await SoundForegroundServiceController.stop();
    unawaited(BleSoundService.instance.connectSavedDevice());
  }

  Future<void> openBleConnectionPage() async {
    if (backgroundAlertsEnabled) {
      await AlertSettingsStore.saveBackgroundAlertsEnabled(false);
      await SoundForegroundServiceController.stop();
    }

    await BleSoundService.instance.disconnect();

    if (!mounted) return;

    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const BleConnectionPage()),
      (_) => false,
    );
  }

  void clearLogs() {
    setState(() {
      logs.clear();
      currentPacket = null;
    });
  }

  String imagePathForLabel(String label) {
    switch (label) {
      case '경보':
      case '화재경보':
        return 'assets/soundimages/alarm_emoji.png';
      case '개소리':
        return 'assets/soundimages/dog_emoji.png';
      default:
        return 'assets/icon_soundkeyring.png';
    }
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomePage(packet: currentPacket, imagePathForLabel: imagePathForLabel),
      LogPage(logs: logs, imagePathForLabel: imagePathForLabel),
      SettingsPage(
        deviceStatus: deviceStatus,
        soundLabels: knownKoreanSoundLabels,
        mutedLabelKeys: mutedLabelKeys,
        backgroundAlertsEnabled: backgroundAlertsEnabled,
        logCount: logs.length,
        onToggleMutedLabel: toggleMutedLabel,
        onToggleBackgroundAlerts: toggleBackgroundAlerts,
        onReconnectDevice: () {
          unawaited(openBleConnectionPage());
        },
        onClearLogs: clearLogs,
      ),
    ];

    return Scaffold(
      body: pages[selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: selectedIndex,
        onTap: (index) {
          setState(() {
            selectedIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: '홈'),
          BottomNavigationBarItem(icon: Icon(Icons.list_alt), label: '로그'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: '설정'),
        ],
      ),
    );
  }

  Map<String, dynamic> _stringKeyMap(Map source) {
    return source.map((key, value) => MapEntry(key.toString(), value));
  }
}
