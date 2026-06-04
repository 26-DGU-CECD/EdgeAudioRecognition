import 'dart:async';

import 'package:flutter/material.dart';

import 'ble/ble_sound_service.dart';
import 'models/device_status.dart';
import 'pages/home_page.dart';
import 'pages/log_page.dart';
import 'pages/settings_page.dart';
import 'models/sound_packet.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key, required this.title});

  final String title;

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  static const List<String> soundLabels = [
    '총',
    '경보',
    '자전거',
    '물소리',
    '울음',
    '비명',
    '유리깨지는소리',
    '화재경보',
    '아기 우는 소리',
    '개소리',
    '고양이소리',
  ];

  int selectedIndex = 0;
  SoundPacket? currentPacket;
  DeviceStatus? deviceStatus;

  final List<SoundPacket> logs = [];
  final Set<String> mutedLabels = {};

  bool backgroundAlertsEnabled = false;

  StreamSubscription<SoundPacket>? _soundSub;
  StreamSubscription<DeviceStatus>? _statusSub;

  @override
  void initState() {
    super.initState();

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
    _soundSub?.cancel();
    _statusSub?.cancel();
    super.dispose();
  }

  void _addPacket(SoundPacket packet) {
    logs.insert(0, packet);

    final isMuted = mutedLabels.contains(packet.displayLabel);
    currentPacket = (!isMuted) ? packet : null;
    // currentPacket = packet.isDisplayable && !isMuted ? packet : null;
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
      HomePage(
        packet: currentPacket,
        imagePathForLabel: imagePathForLabel,
        onReset: resetToListening,
        onMockDog: () {
          receivePacket({
            'status': 'ok',
            'time': '12:14:53',
            'label': 'dog_bark',
            'display_label': '개소리',
            'score': 0.998,
            'infer_sec': 0.118,
            'total_sec': 2.359,
            'db': 47.2,
            'level': 'caution',
            'direction': '서',
            'angle': 233.0,
            'angle_raw': 233.0,
            'direction_text': '서쪽 233도',
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
            'display_label': '경보',
            'score': 0.963,
            'infer_sec': 0.139,
            'total_sec': 2.411,
            'db': 68.1,
            'level': 'danger',
            'direction': '북',
            'angle': 12.0,
            'angle_raw': 12.0,
            'direction_text': '북쪽 12도',
            'doa_status': 'enabled',
            'raw': 'alarm score=0.963 db=68.1 doa=12',
            'items': [],
          });
        },
      ),
      LogPage(
        logs: logs,
        imagePathForLabel: imagePathForLabel,
      ),
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
}
