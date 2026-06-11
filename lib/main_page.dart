import 'dart:async';

import 'package:flutter/material.dart';

import 'ble/ble_sound_service.dart';
import 'models/device_status.dart';
import 'models/sound_packet.dart';
import 'pages/home_page.dart';
import 'pages/log_page.dart';
import 'pages/settings_page.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key, required this.title});

  final String title;

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
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
    currentPacket = isMuted ? null : packet;
    // currentPacket = packet.isDisplayable && !isMuted ? packet : null;
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
      ),
      LogPage(
        logs: logs,
        imagePathForLabel: imagePathForLabel,
      ),
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
