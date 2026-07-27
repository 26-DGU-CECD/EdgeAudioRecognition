import 'package:flutter/material.dart';

import '../ble/ble_sound_service.dart';
import '../main_page.dart';
import '../services/alert_settings_store.dart';
import '../services/sound_foreground_task.dart';
import '../ui/app_colors.dart';

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

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('백그라운드 알림을 시작하지 못했습니다.')),
      );
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
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(26, 8, 26, 28),
          child: Column(
            children: [
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      width: 128,
                      height: 128,
                      decoration: const BoxDecoration(
                        color: AppColors.primarySoft,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.notifications_active,
                          size: 64, color: AppColors.primary),
                    ),
                    const SizedBox(height: 30),
                    const Text(
                      '앱을 닫아도\n계속 지켜드릴게요',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 25,
                        height: 1.35,
                        fontWeight: FontWeight.w900,
                        color: AppColors.textPrimary,
                        letterSpacing: -0.2,
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      '백그라운드 알림을 켜면 앱을 사용하지 않을\n때에도 miimo 키링이 위험음을 감지해\n휴대폰 알림으로 바로 알려드려요.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 15,
                        height: 1.6,
                        fontWeight: FontWeight.w500,
                        color: AppColors.textSecondary,
                      ),
                    ),
                    const SizedBox(height: 26),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 14),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF6F8FA),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(Icons.verified_user,
                              size: 20, color: AppColors.success),
                          SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              '소리 분석은 키링 안에서 처리되며, 녹음 내용은 휴대폰에 저장되지 않아요.',
                              style: TextStyle(
                                fontSize: 13,
                                height: 1.5,
                                fontWeight: FontWeight.w500,
                                color: AppColors.textSecondary,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: FilledButton.icon(
                  onPressed: _isSubmitting ? null : _allowBackgroundAlerts,
                  icon: _isSubmitting
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2.4,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.check, size: 22),
                  label: Text(_isSubmitting ? '설정 중...' : '허용하고 시작하기'),
                  style: FilledButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18)),
                    textStyle: const TextStyle(
                        fontSize: 16, fontWeight: FontWeight.w800),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                height: 52,
                child: TextButton(
                  onPressed: _isSubmitting ? null : _skip,
                  child: const Text(
                    '나중에 하기',
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textMuted,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
