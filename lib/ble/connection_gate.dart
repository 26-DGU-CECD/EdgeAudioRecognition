import 'package:flutter/material.dart';
import 'ble_sound_service.dart';
import 'ble_connection_page.dart';
import 'ble_connection_store.dart';
import 'package:untitled/main_page.dart';
import 'package:untitled/pages/background_alert_consent_page.dart';
import 'package:untitled/services/alert_settings_store.dart';
import 'package:untitled/services/sound_foreground_task.dart';

/// 앱 시작 시 저장된 ble 기기가 있으면 메인으로 보내고, 없으면 ble 연결 화면
class ConnectionGate extends StatefulWidget {
  const ConnectionGate({super.key});

  @override
  State<ConnectionGate> createState() => _ConnectionGateState();
}

class _ConnectionGateState extends State<ConnectionGate> {
  @override
  void initState() {
    super.initState();
    _routeBySavedConnection();
  }

  Future<void> _routeBySavedConnection() async {
    final savedDeviceId = await BleConnectionStore.loadDeviceId();
    final backgroundAlertsEnabled =
        await AlertSettingsStore.loadBackgroundAlertsEnabled();

    if (backgroundAlertsEnabled && savedDeviceId != null) {
      await SoundForegroundServiceController.start();

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => const MainPage(title: 'Demo Home Page'),
        ),
      );
      return;
    }

    final connected = await BleSoundService.instance.connectSavedDevice();

    if (!mounted) return;

    if (connected) {
      final shouldShowBackgroundAlertConsent =
          await BackgroundAlertConsentPage.shouldShow();

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => shouldShowBackgroundAlertConsent
              ? const BackgroundAlertConsentPage()
              : const MainPage(title: 'Demo Home Page'),
        ),
      );
      return;
    }
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const BleConnectionPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    return const Scaffold(body: Center(child: CircularProgressIndicator()));
  }
}
