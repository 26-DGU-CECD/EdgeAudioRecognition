import 'package:flutter/material.dart';

import '../app_prefs.dart';
import '../ble/connection_gate.dart';
import '../ui/app_colors.dart';

class BackgroundConsentPage extends StatelessWidget {
  const BackgroundConsentPage({super.key});

  Future<void> _proceed(BuildContext context, bool allow) async {
    await AppPrefs.setBackgroundAlertsEnabled(allow);
    if (!context.mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const ConnectionGate()),
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
                  onPressed: () => _proceed(context, true),
                  icon: const Icon(Icons.check, size: 22),
                  label: const Text('허용하고 시작하기'),
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
                  onPressed: () => _proceed(context, false),
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
