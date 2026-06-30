import 'dart:async';

import 'package:flutter/material.dart';

import 'app_prefs.dart';
import 'ble/ble_connection_page.dart';
import 'ble/ble_sound_service.dart';
import 'device_status.dart';
import 'home_page.dart';
import 'log_page.dart';
import 'services/notification_service.dart';
import 'settings_page.dart';
import 'sound_packet.dart';
import 'ui/app_bottom_nav.dart';
import 'ui/push_banner.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key, required this.title});

  final String title;

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> with WidgetsBindingObserver {
  static const List<String> soundLabels = [
    '공사장 소음',
    '총소리',
    '응급·도난·화재 경보음',
    '자전거 접근 소리',
    '차량 경적',
    '물 흐르는 소리',
    '노크 소리',
    '가전제품 작동음',
    '아기 울음소리',
    '개·고양이 울음소리',
    '사람 울음 및 비명',
    '유리 깨지는 소리',
  ];

  int selectedIndex = 0;
  SoundPacket? currentPacket;
  DeviceStatus? deviceStatus;

  /// 키링 연결 여부. 미연결이면 홈에 프레임 K(미연결)를 보여줍니다.
  bool connected = false;

  final List<SoundPacket> logs = [];
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

    _soundSub = BleSoundService.instance.soundPackets.listen((packet) {
      setState(() {
        _addPacket(packet);
      });
    });

    _statusSub = BleSoundService.instance.deviceStatuses.listen((status) {
      setState(() {
        deviceStatus = status;
        connected = status.connection == 'connected';
      });
    });

    AppPrefs.isBackgroundAlertsEnabled().then((enabled) {
      if (!mounted) return;
      setState(() {
        backgroundAlertsEnabled = enabled;
      });
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _soundSub?.cancel();
    _statusSub?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _lifecycle = state;
  }

  void _addPacket(SoundPacket packet) {
    logs.insert(0, packet);

    final isMuted = mutedLabels.contains(packet.displayLabel);
    final shouldAlert = packet.isDisplayable && !isMuted;
    currentPacket = shouldAlert ? packet : null;

    if (shouldAlert) _alert(packet);
  }

  /// 포그라운드면 인앱 헤즈업 배너(P3), 백그라운드면 OS 푸시(P1/P2/P3).
  void _alert(SoundPacket packet) {
    final foreground = _lifecycle == AppLifecycleState.resumed;
    if (foreground) {
      bannerPacket = packet;
    } else if (backgroundAlertsEnabled) {
      NotificationService.instance.showForPacket(packet);
    }
  }

  void receivePacket(Map<String, dynamic> json) {
    final packet = SoundPacket.fromJson(json);
    setState(() {
      _addPacket(packet);
    });
  }

  void resetToListening() {
    setState(() {
      currentPacket = null;
    });
  }

  void toggleMutedLabel(String label) {
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
  }

  void toggleBackgroundAlerts(bool enabled) {
    setState(() {
      backgroundAlertsEnabled = enabled;
    });
    AppPrefs.setBackgroundAlertsEnabled(enabled);
  }

  void _goToReconnect() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const BleConnectionPage()),
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
        onReconnect: _goToReconnect,
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
            'doa_status': 'enabled',
            'raw': 'alarm score=0.963 db=68.1 doa=12',
            'items': [],
          });
        },
      ),
      LogPage(logs: logs),
      SettingsPage(
        deviceStatus: deviceStatus,
        soundLabels: soundLabels,
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
