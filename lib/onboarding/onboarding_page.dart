import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../app_prefs.dart';
import '../home_page.dart' show CompassView;
import '../ui/app_colors.dart';
import 'background_consent_page.dart';

class OnboardingPage extends StatefulWidget {
  const OnboardingPage({super.key});

  @override
  State<OnboardingPage> createState() => _OnboardingPageState();
}

class _OnboardingPageState extends State<OnboardingPage>
    with SingleTickerProviderStateMixin {
  final _controller = PageController();
  late final AnimationController _wave;
  int _index = 0;

  late final List<_Slide> _slides = [
    _Slide(
      visual: _circleVisual(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Image.asset('assets/miimo.png', fit: BoxFit.contain),
        ),
      ),
      title: 'miimo를 만나보세요',
      body: '주변 소리를 대신 들어주는\n작고 똑똑한 소리 친구예요.',
    ),
    _Slide(
      visual: _circleVisual(child: _waveform()),
      title: '소리를 실시간으로 인식해요',
      body: '화재경보·총소리·유리 깨짐 등\n12종의 위험한 소리를 구분해요.',
    ),
    _Slide(
      visual: const CompassView(
        angle: 88,
        color: AppColors.caution,
        soft: AppColors.cautionSoft,
        icon: Icons.explore,
        size: 260,
      ),
      title: '어디서 났는지 알려줘요',
      body: '소리가 들린 방향을 나침반으로\n한눈에 보여드려요.',
    ),
    _Slide(
      visual: _circleVisual(
        background: AppColors.dangerSoft,
        child: const Icon(Icons.notifications_active,
            size: 96, color: AppColors.danger),
      ),
      title: '위험하면 즉시 알림',
      body: '앱을 닫아도 백그라운드에서\n휴대폰 알림으로 바로 전해드려요.',
    ),
    _Slide(
      visual: _circleVisual(
        inset: 16,
        child: Stack(
          alignment: Alignment.center,
          clipBehavior: Clip.none,
          children: [
            Padding(
              padding: const EdgeInsets.all(22),
              child: Image.asset('assets/miimo.png', fit: BoxFit.contain),
            ),
            Positioned(
              bottom: 6,
              right: 2,
              child: Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  shape: BoxShape.circle,
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x591B6EF3),
                      blurRadius: 20,
                      offset: Offset(0, 8),
                    ),
                  ],
                ),
                child: const Icon(Icons.bluetooth,
                    size: 28, color: Colors.white),
              ),
            ),
          ],
        ),
      ),
      title: '이제 연결만 하면 끝!',
      body: 'miimo 키링을 휴대폰과 연결하고\n바로 사용해 보세요.',
    ),
  ];

  @override
  void initState() {
    super.initState();
    _wave = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    _wave.dispose();
    super.dispose();
  }

  bool get _isLast => _index == _slides.length - 1;

  void _next() {
    if (_isLast) {
      _finish();
    } else {
      _controller.nextPage(
        duration: const Duration(milliseconds: 320),
        curve: Curves.easeOut,
      );
    }
  }

  Future<void> _finish() async {
    await AppPrefs.setOnboardingSeen(true);
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const BackgroundConsentPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            // skip
            Align(
              alignment: Alignment.centerRight,
              child: AnimatedOpacity(
                opacity: _isLast ? 0 : 1,
                duration: const Duration(milliseconds: 200),
                child: TextButton(
                  onPressed: _isLast ? null : _finish,
                  child: const Text(
                    '건너뛰기',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textMuted,
                    ),
                  ),
                ),
              ),
            ),
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _slides.length,
                onPageChanged: (i) => setState(() => _index = i),
                itemBuilder: (_, i) {
                  final slide = _slides[i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 26),
                    child: Column(
                      children: [
                        Expanded(child: Center(child: slide.visual)),
                        Text(
                          slide.title,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            fontSize: 27,
                            height: 1.3,
                            fontWeight: FontWeight.w900,
                            color: AppColors.textPrimary,
                            letterSpacing: -0.2,
                          ),
                        ),
                        const SizedBox(height: 14),
                        Text(
                          slide.body,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            fontSize: 15,
                            height: 1.6,
                            fontWeight: FontWeight.w500,
                            color: AppColors.textSecondary,
                          ),
                        ),
                        const Spacer(),
                      ],
                    ),
                  );
                },
              ),
            ),
            _dots(),
            const SizedBox(height: 18),
            Padding(
              padding: const EdgeInsets.fromLTRB(26, 0, 26, 28),
              child: SizedBox(
                width: double.infinity,
                height: 56,
                child: FilledButton.icon(
                  onPressed: _next,
                  icon: _isLast
                      ? const Icon(Icons.bluetooth_searching, size: 22)
                      : const SizedBox.shrink(),
                  label: Text(_isLast ? '키링 연결하고 시작하기' : '다음'),
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
            ),
          ],
        ),
      ),
    );
  }

  Widget _dots() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        for (var i = 0; i < _slides.length; i++) ...[
          AnimatedContainer(
            duration: const Duration(milliseconds: 220),
            width: i == _index ? 22 : 7,
            height: 7,
            decoration: BoxDecoration(
              color: i == _index ? AppColors.primary : const Color(0xFFD6DCE4),
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          const SizedBox(width: 7),
        ],
      ],
    );
  }

  static Widget _circleVisual({
    required Widget child,
    Color background = AppColors.primarySoft,
    double inset = 0,
  }) {
    return SizedBox(
      width: 210,
      height: 210,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Padding(
            padding: EdgeInsets.all(inset),
            child: Container(
              decoration: BoxDecoration(
                color: background,
                shape: BoxShape.circle,
              ),
            ),
          ),
          child,
        ],
      ),
    );
  }

  Widget _waveform() {
    const heights = [38.0, 70.0, 100.0, 130.0, 100.0, 70.0, 38.0];
    const phases = [0.0, 0.15, 0.3, 0.45, 0.3, 0.15, 0.0];
    return AnimatedBuilder(
      animation: _wave,
      builder: (_, _) {
        return Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            for (var i = 0; i < heights.length; i++) ...[
              _waveBar(heights[i], phases[i]),
              const SizedBox(width: 8),
            ],
          ],
        );
      },
    );
  }

  Widget _waveBar(double maxHeight, double phase) {
    final v = (math.sin((_wave.value + phase) * 2 * math.pi) + 1) / 2;
    final scale = 0.4 + v * 0.6;
    return Container(
      width: 9,
      height: maxHeight * scale,
      decoration: BoxDecoration(
        color: AppColors.primary,
        borderRadius: BorderRadius.circular(5),
      ),
    );
  }
}

class _Slide {
  final Widget visual;
  final String title;
  final String body;

  const _Slide({
    required this.visual,
    required this.title,
    required this.body,
  });
}
