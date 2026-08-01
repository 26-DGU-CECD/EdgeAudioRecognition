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
import 'ui/app_bottom_nav.dart';
import 'ui/push_banner.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key, required this.title});

  final String title;

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> with WidgetsBindingObserver {
  int selectedIndex = 0;
  SoundPacket? currentPacket;
  DeviceStatus? deviceStatus;

  /// 데모 소리를 실행할 때 홈 헤더에만 표시하는 가상 배터리 잔량입니다.
  /// 실제 BLE DeviceStatus는 변경하지 않습니다.
  int? demoBatteryPercent;

  /// 키링 연결 여부. 미연결이면 홈에 프레임 K(미연결)를 보여줍니다.
  bool connected = false;

  final List<SoundPacket> logs = [];

  /// 음소거된 표시 라벨 집합(UI 기준). 백그라운드 필터는 라벨 키로 저장됩니다.
  final Set<String> mutedLabels = {};

  bool backgroundAlertsEnabled = false;

  /// 인앱 헤즈업 배너(P3)로 표시할 패킷. null이면 배너 없음.
  SoundPacket? bannerPacket;

  /// 포그라운드/백그라운드 판단용
  AppLifecycleState _lifecycle = AppLifecycleState.resumed;

  StreamSubscription<SoundPacket>? _soundSub;
  StreamSubscription<DeviceStatus>? _statusSub;

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addObserver(this);

    // connect()의 'connected' 상태는 이 페이지 진입 전에 방출되므로
    // 현재 연결 여부는 서비스에서 직접 가져온다.
    connected = BleSoundService.instance.isConnected;

    // 백그라운드 포그라운드 서비스가 UI로 전달하는 패킷/상태 수신
    FlutterForegroundTask.addTaskDataCallback(_onForegroundTaskData);

    _soundSub = BleSoundService.instance.soundPackets.listen((packet) {
      setState(() {
        demoBatteryPercent = null;
        _addPacket(packet);
      });
    });

    _statusSub = BleSoundService.instance.deviceStatuses.listen((status) {
      setState(() {
        deviceStatus = status;
        connected = status.connection == 'connected';
      });
    });

    _loadSavedSettings();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    FlutterForegroundTask.removeTaskDataCallback(_onForegroundTaskData);
    _soundSub?.cancel();
    _statusSub?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _lifecycle = state;
  }

  Future<void> _loadSavedSettings() async {
    final savedBackgroundAlertsEnabled =
        await AlertSettingsStore.loadBackgroundAlertsEnabled();
    final savedMutedKeys = await AlertSettingsStore.loadMutedLabelKeys();

    if (!mounted) return;

    setState(() {
      backgroundAlertsEnabled = savedBackgroundAlertsEnabled;
      mutedLabels
        ..clear()
        ..addAll(knownKoreanSoundLabels.where(
          (label) => savedMutedKeys
              .contains(AlertSettingsStore.labelKeyForDisplayLabel(label)),
        ));
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

    final payloadJson =
        payload.map((key, value) => MapEntry(key.toString(), value));

    switch (data['type']) {
      case 'sound_packet':
        setState(() {
          demoBatteryPercent = null;
          _addPacket(SoundPacket.fromJson(payloadJson));
        });
        break;
      case 'device_status':
        setState(() {
          deviceStatus = DeviceStatus.fromJson(payloadJson);
          connected = deviceStatus?.connection == 'connected';
        });
        break;
    }
  }

  void _addPacket(SoundPacket packet) {
    logs.insert(0, packet);

    final isMuted = mutedLabels.contains(packet.displayLabel);
    final shouldAlert = packet.isDisplayable && !isMuted;
    currentPacket = shouldAlert ? packet : null;

    if (shouldAlert) _alert(packet);
  }

  /// 포그라운드면 인앱 헤즈업 배너(P3). 백그라운드에서는 포그라운드 서비스가
  /// 자체적으로 OS 알림을 표시하므로 여기서는 배너만 처리한다.
  void _alert(SoundPacket packet) {
    final foreground = _lifecycle == AppLifecycleState.resumed;
    if (foreground) {
      bannerPacket = packet;
    }
  }

  void receivePacket(Map<String, dynamic> json) {
    final packet = SoundPacket.fromJson(json);
    final battery = DeviceStatus.parseBatteryPercent(
      json['battery_percent'] ?? json['battery'],
    );
    setState(() {
      demoBatteryPercent = battery;
      _addPacket(packet);
    });
  }

  void resetToListening() {
    setState(() {
      currentPacket = null;
      demoBatteryPercent = null;
    });
  }

  Future<void> toggleMutedLabel(String label) async {
    setState(() {
      if (mutedLabels.contains(label)) {
        mutedLabels.remove(label);
      } else {
        mutedLabels.add(label);
        if (currentPacket?.displayLabel == label) {
          currentPacket = null;
        }
      }
    });

    final keys =
        mutedLabels.map(AlertSettingsStore.labelKeyForDisplayLabel).toSet();
    await AlertSettingsStore.saveMutedLabelKeys(keys);
  }

  Future<void> toggleBackgroundAlerts(bool enabled) async {
    setState(() {
      backgroundAlertsEnabled = enabled;
    });

    await AlertSettingsStore.saveBackgroundAlertsEnabled(enabled);

    if (enabled) {
      // 포그라운드 서비스가 BLE를 전담하므로 UI측 연결은 해제한다.
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

  Future<void> _goToReconnect() async {
    if (backgroundAlertsEnabled) {
      await AlertSettingsStore.saveBackgroundAlertsEnabled(false);
      await SoundForegroundServiceController.stop();
    }

    await BleSoundService.instance.disconnect();

    if (!mounted) return;

    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(
        builder: (_) =>
            const BleConnectionPage(showBackgroundAlertConsentOnConnect: true),
      ),
      (_) => false,
    );
  }

  void clearLogs() {
    setState(() {
      logs.clear();
      currentPacket = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomePage(
        packet: currentPacket,
        deviceStatus: deviceStatus,
        demoBatteryPercent: demoBatteryPercent,
        connected: connected,
        recentLogs: logs,
        onReset: resetToListening,
        onMuteCurrent: () {
          final label = currentPacket?.displayLabel;
          if (label != null) toggleMutedLabel(label);
        },
        onShowAllLogs: () {
          setState(() {
            selectedIndex = 1;
          });
        },
        onReconnect: () {
          unawaited(_goToReconnect());
        },
        onMockDog: () {
          receivePacket({
            'status': 'ok',
            'time': '12:14:53',
            'label': 'dog_bark',
            'display_label': '개·고양이 울음소리',
            'score': 0.998,
            'infer_sec': 0.118,
            'total_sec': 2.359,
            'db': 47.2,
            'level': 'caution',
            'angle': 233.0,
            'battery': 76,
            'doa_status': 'enabled',
            'raw': 'dog_bark score=0.998 db=47.2 doa=233',
            'items': [],
          });
        },
        onMockDanger: () {
          receivePacket({
            'status': 'ok',
            'time': '12:20:17',
            'label': 'alarm',
            'display_label': '응급·도난·화재 경보음',
            'score': 0.963,
            'infer_sec': 0.139,
            'total_sec': 2.411,
            'db': 68.1,
            'level': 'danger',
            'angle': 12.0,
            'battery': 15,
            'doa_status': 'enabled',
            'raw': 'alarm score=0.963 db=68.1 doa=12',
            'items': [],
          });
        },
      ),
      LogPage(logs: logs),
      SettingsPage(
        deviceStatus: deviceStatus,
        soundLabels: knownKoreanSoundLabels,
        mutedLabels: mutedLabels,
        backgroundAlertsEnabled: backgroundAlertsEnabled,
        logCount: logs.length,
        onToggleMutedLabel: toggleMutedLabel,
        onToggleBackgroundAlerts: toggleBackgroundAlerts,
        onClearLogs: clearLogs,
      ),
    ];

    return Scaffold(
      body: Stack(
        children: [
          IndexedStack(
            index: selectedIndex,
            children: pages,
          ),
          if (bannerPacket != null)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: HeadsUpBanner(
                key: ValueKey(bannerPacket),
                packet: bannerPacket!,
                onDismiss: () {
                  if (mounted) setState(() => bannerPacket = null);
                },
                onTap: () {
                  setState(() => selectedIndex = 0);
                },
              ),
            ),
        ],
      ),
      bottomNavigationBar: AppBottomNav(
        currentIndex: selectedIndex,
        onTap: (index) {
          setState(() {
            selectedIndex = index;
          });
        },
      ),
    );
  }
}
