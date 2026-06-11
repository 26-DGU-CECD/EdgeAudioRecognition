import 'package:flutter/material.dart';

import '../ble/ble_sound_service.dart';
import '../main_page.dart';
import '../services/alert_settings_store.dart';
import '../services/sound_foreground_task.dart';

class BackgroundAlertConsentPage extends StatefulWidget {
  const BackgroundAlertConsentPage({super.key});

  static Future<bool> shouldShow() {
    return AlertSettingsStore.shouldShowBackgroundAlertConsent();
  }

  @override
  State<BackgroundAlertConsentPage> createState() =>
      _BackgroundAlertConsentPageState();
}

class _BackgroundAlertConsentPageState
    extends State<BackgroundAlertConsentPage> {
  bool _isSubmitting = false;

  Future<void> _allowBackgroundAlerts() async {
    setState(() {
      _isSubmitting = true;
    });

    await AlertSettingsStore.saveBackgroundAlertsEnabled(true);
    await BleSoundService.instance.disconnect();
    final started = await SoundForegroundServiceController.start();

    if (!mounted) return;

    if (!started) {
      await AlertSettingsStore.saveBackgroundAlertsEnabled(false);
      await BleSoundService.instance.connectSavedDevice();

      if (!mounted) return;

      setState(() {
        _isSubmitting = false;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('백그라운드 알림을 시작하지 못했습니다.')));
      return;
    }

    await AlertSettingsStore.saveBackgroundAlertConsentSeen(true);

    if (!mounted) return;

    _openMainPage();
  }

  Future<void> _skip() async {
    await AlertSettingsStore.saveBackgroundAlertConsentSeen(true);

    if (!mounted) return;

    _openMainPage();
  }

  void _openMainPage() {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(
        builder: (_) => const MainPage(title: 'Sound Keychain'),
      ),
      (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('백그라운드 알림')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Spacer(),
              Icon(
                Icons.notifications_active,
                size: 72,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 24),
              const Text(
                '백그라운드 알림을 동의해야 앱을 사용하지 않아도 자동으로 알림을 받을 수 있습니다.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              const Text(
                'miimo 키링이 위험음을 감지하면 휴대폰 알림으로 알려드립니다.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 15, color: Colors.black54),
              ),
              const Spacer(),
              FilledButton.icon(
                onPressed: _isSubmitting ? null : _allowBackgroundAlerts,
                icon: const Icon(Icons.check),
                label: Text(_isSubmitting ? '설정 중...' : '허용하고 시작하기'),
              ),
              const SizedBox(height: 8),
              TextButton(
                onPressed: _isSubmitting ? null : _skip,
                child: const Text('나중에 하기'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
