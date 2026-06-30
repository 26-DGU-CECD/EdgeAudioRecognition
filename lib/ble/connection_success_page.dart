import 'package:flutter/material.dart';
import 'package:untitled/main_page.dart';
import 'package:untitled/ui/app_colors.dart';

/// 프레임 L — BLE 연결 성공 피드백.
/// 연결 직후 보여주고, "감지 시작하기"를 누르면 메인 화면으로 진입합니다.
class ConnectionSuccessPage extends StatefulWidget {
  final String deviceName;
  final int? battery;

  /// 메인 화면으로 넘길 때 사용할 타이틀.
  final String mainTitle;

  const ConnectionSuccessPage({
    super.key,
    required this.deviceName,
    this.battery,
    this.mainTitle = 'Sound Keychain',
  });

  @override
  State<ConnectionSuccessPage> createState() => _ConnectionSuccessPageState();
}

class _ConnectionSuccessPageState extends State<ConnectionSuccessPage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2600),
    )..repeat();
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  void _start() {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => MainPage(title: widget.mainTitle)),
      (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final battery = widget.battery;

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(26, 8, 26, 28),
          child: Column(
            children: [
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _successMark(),
                    const SizedBox(height: 28),
                    const Text(
                      '연결 완료!',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 27,
                        fontWeight: FontWeight.w900,
                        color: AppColors.textPrimary,
                        letterSpacing: -0.2,
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      '이제 miimo 키링이 주변 소리를\n실시간으로 감지하기 시작해요.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 15,
                        height: 1.6,
                        fontWeight: FontWeight.w500,
                        color: AppColors.textSecondary,
                      ),
                    ),
                    const SizedBox(height: 26),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _chip(
                          icon: Icons.memory,
                          iconColor: AppColors.primary,
                          label: widget.deviceName,
                        ),
                        const SizedBox(width: 10),
                        battery != null
                            ? _chip(
                                icon: Icons.battery_full,
                                iconColor: AppColors.success,
                                label: '$battery%',
                              )
                            : _chip(
                                icon: Icons.check_circle,
                                iconColor: AppColors.success,
                                label: '연결됨',
                              ),
                      ],
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: FilledButton.icon(
                  onPressed: _start,
                  icon: const Icon(Icons.hearing, size: 21),
                  label: const Text('감지 시작하기'),
                  style: FilledButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(18),
                    ),
                    textStyle: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
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

  Widget _successMark() {
    return SizedBox(
      width: 210,
      height: 210,
      child: AnimatedBuilder(
        animation: _pulse,
        builder: (_, child) {
          return Stack(
            alignment: Alignment.center,
            children: [
              _ring(_pulse.value),
              _ring((_pulse.value + 0.5) % 1.0),
              child!,
            ],
          );
        },
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: 128,
              height: 128,
              decoration: const BoxDecoration(
                color: AppColors.success,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Color(0x52169A4A),
                    blurRadius: 30,
                    offset: Offset(0, 12),
                  ),
                ],
              ),
              child: const Icon(Icons.check, size: 72, color: Colors.white),
            ),
            Positioned(
              bottom: 6,
              right: 6,
              child: Container(
                width: 60,
                height: 60,
                padding: const EdgeInsets.all(6),
                decoration: const BoxDecoration(
                  color: AppColors.card,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: Color(0x29141628),
                      blurRadius: 20,
                      offset: Offset(0, 8),
                    ),
                  ],
                ),
                child: Image.asset('assets/miimo.png', fit: BoxFit.contain),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _ring(double t) {
    final scale = 0.55 + t * 1.1;
    final opacity = (0.5 * (1 - t)).clamp(0.0, 1.0);
    return Opacity(
      opacity: opacity,
      child: Transform.scale(
        scale: scale,
        child: Container(
          width: 210,
          height: 210,
          decoration: BoxDecoration(
            color: AppColors.success.withValues(alpha: 0.12),
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }

  Widget _chip({
    required IconData icon,
    required Color iconColor,
    required String label,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 11),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 19, color: iconColor),
          const SizedBox(width: 7),
          Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}
