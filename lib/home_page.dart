import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'device_status.dart';
import 'sound_packet.dart';
import 'ui/app_colors.dart';
import 'ui/app_widgets.dart';
import 'ui/sound_style.dart';

/// 홈 화면 — 실시간 감지 (대기 / 감지됨 / 기기 미연결)
class HomePage extends StatefulWidget {
  final SoundPacket? packet;
  final DeviceStatus? deviceStatus;

  /// 키링 연결 여부. false면 미연결(프레임 K) 화면을 보여줍니다.
  final bool connected;
  final List<SoundPacket> recentLogs;
  final VoidCallback onReset;
  final VoidCallback onMuteCurrent;
  final VoidCallback onShowAllLogs;
  final VoidCallback onReconnect;
  final VoidCallback onMockDog;
  final VoidCallback onMockDanger;

  const HomePage({
    super.key,
    required this.packet,
    required this.deviceStatus,
    required this.connected,
    required this.recentLogs,
    required this.onReset,
    required this.onMuteCurrent,
    required this.onShowAllLogs,
    required this.onReconnect,
    required this.onMockDog,
    required this.onMockDanger,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  late final AnimationController _pulse;
  late final AnimationController _wave;
  late final AnimationController _blink;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2600),
    )..repeat();
    _wave = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat();
    _blink = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    _wave.dispose();
    _blink.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final packet = widget.packet;
    // packet이 있으면 감지됨, 연결돼 있으면 대기, 아니면 미연결(K).
    final isListening = packet == null && widget.connected;

    return SafeArea(
      bottom: false,
      child: Column(
        children: [
          _header(isListening),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(18, 6, 18, 18),
              child: packet != null
                  ? _detectedView(packet)
                  : widget.connected
                      ? _listeningView()
                      : _disconnectedView(),
            ),
          ),
        ],
      ),
    );
  }

  // ----- header -----
  Widget _header(bool isListening) {
    final battery = widget.deviceStatus?.battery;
    final connected = widget.connected;
    final subtitle = !connected
        ? '감지가 멈춰 있어요'
        : isListening
            ? '소리를 기다리는 중'
            : '듣는 중 · 오늘 ${widget.recentLogs.length}건';

    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 6, 18, 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '실시간 감지',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w900,
                    color: AppColors.textPrimary,
                    letterSpacing: -0.2,
                  ),
                ),
                const SizedBox(height: 5),
                Row(
                  children: [
                    _statusDot(connected),
                    const SizedBox(width: 7),
                    Text(
                      subtitle,
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                        color: connected
                            ? AppColors.textSecondary
                            : AppColors.textMuted,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 9),
            decoration: BoxDecoration(
              color: connected ? AppColors.primarySoft : AppColors.dangerSoft,
              borderRadius: BorderRadius.circular(999),
            ),
            child: Row(
              children: [
                Icon(
                  connected
                      ? Icons.bluetooth_connected
                      : Icons.bluetooth_disabled,
                  size: 18,
                  color: connected ? AppColors.primary : AppColors.danger,
                ),
                const SizedBox(width: 6),
                Text(
                  connected
                      ? (battery != null ? '$battery%' : '연결')
                      : '끊김',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                    color: connected ? AppColors.primary : AppColors.danger,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _statusDot(bool connected) {
    final dot = Container(
      width: 9,
      height: 9,
      decoration: BoxDecoration(
        color: connected ? AppColors.success : AppColors.textMuted,
        shape: BoxShape.circle,
      ),
    );
    if (!connected) return dot;
    return FadeTransition(
      opacity: Tween(begin: 1.0, end: 0.25).animate(_blink),
      child: dot,
    );
  }

  // ----- detected -----
  Widget _detectedView(SoundPacket packet) {
    final style = LevelStyle.of(packet.level);
    final dir = '${packet.directionLabel} 방향';

    return Column(
      children: [
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.fromLTRB(20, 22, 20, 20),
          decoration: BoxDecoration(
            color: AppColors.card,
            borderRadius: BorderRadius.circular(30),
            border: Border.all(color: AppColors.cardBorder),
            boxShadow: const [
              BoxShadow(
                color: Color(0x0F141628),
                blurRadius: 30,
                offset: Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            children: [
              // level chip
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 15, vertical: 8),
                decoration: BoxDecoration(
                  color: style.soft,
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      packet.isDanger ? Icons.error : Icons.warning_rounded,
                      size: 18,
                      color: style.color,
                    ),
                    const SizedBox(width: 7),
                    Text(
                      '${style.label} · 방금 감지됨',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: style.color,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 18),
              CompassView(
                angle: packet.angle,
                color: style.color,
                soft: style.soft,
                icon: iconForSoundLabel(packet.displayLabel),
              ),
              const SizedBox(height: 16),
              Text(
                packet.displayLabel,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 27,
                  fontWeight: FontWeight.w900,
                  color: AppColors.textPrimary,
                  letterSpacing: -0.2,
                ),
              ),
              const SizedBox(height: 5),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.navigation,
                      size: 17, color: AppColors.textSecondary),
                  const SizedBox(width: 6),
                  Text(
                    '$dir · ${packet.angle.toStringAsFixed(0)}°',
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  _metric('신뢰도', '${(packet.score * 100).toStringAsFixed(0)}%'),
                  const SizedBox(width: 8),
                  _metric('음량', '${packet.db.toStringAsFixed(0)}dB'),
                  const SizedBox(width: 8),
                  _metric('반응', '${packet.inferSec.toStringAsFixed(2)}초'),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),
        _muteButton(),
        const SizedBox(height: 14),
        _demoRow(),
      ],
    );
  }

  Widget _metric(String label, String value) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 13, horizontal: 8),
        decoration: BoxDecoration(
          color: const Color(0xFFF6F8FA),
          borderRadius: BorderRadius.circular(18),
        ),
        child: Column(
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: AppColors.textMuted,
              ),
            ),
            const SizedBox(height: 3),
            Text(
              value,
              style: const TextStyle(
                fontSize: 19,
                fontWeight: FontWeight.w900,
                color: AppColors.textPrimary,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _muteButton() {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: OutlinedButton.icon(
        onPressed: widget.onMuteCurrent,
        icon: const Icon(Icons.notifications_off, size: 20),
        label: const Text('이 소리 알림 끄기'),
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.textSecondary,
          backgroundColor: AppColors.card,
          side: const BorderSide(color: AppColors.divider),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
          textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
        ),
      ),
    );
  }

  // ----- disconnected (프레임 K) -----
  Widget _disconnectedView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 30),
        Center(
          child: SizedBox(
            width: 188,
            height: 188,
            child: Stack(
              children: [
                Container(
                  width: 188,
                  height: 188,
                  decoration: const BoxDecoration(
                    color: Color(0xFFEDEFF3),
                    shape: BoxShape.circle,
                  ),
                  alignment: Alignment.center,
                  child: const Icon(
                    Icons.hearing_disabled,
                    size: 84,
                    color: AppColors.textFaint,
                  ),
                ),
                Positioned(
                  bottom: 8,
                  right: 14,
                  child: Container(
                    width: 54,
                    height: 54,
                    decoration: const BoxDecoration(
                      color: AppColors.card,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: Color(0x38D92D20),
                          blurRadius: 20,
                          offset: Offset(0, 8),
                        ),
                      ],
                    ),
                    alignment: Alignment.center,
                    child: const Icon(
                      Icons.link_off,
                      size: 30,
                      color: AppColors.danger,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 18),
        const Text(
          '키링과 연결이 끊겼어요',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w900,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 7),
        const Text(
          '연결되어 있지 않으면 주변 소리를\n감지하거나 알려드릴 수 없어요.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 14,
            height: 1.6,
            fontWeight: FontWeight.w500,
            color: AppColors.textSecondary,
          ),
        ),
        const SizedBox(height: 30),
        _reconnectGuide(),
        const SizedBox(height: 12),
        SizedBox(
          height: 56,
          child: FilledButton.icon(
            onPressed: widget.onReconnect,
            icon: const Icon(Icons.sync, size: 21),
            label: const Text('다시 연결하기'),
            style: FilledButton.styleFrom(
              backgroundColor: AppColors.primary,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
              ),
              textStyle:
                  const TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
            ),
          ),
        ),
        const SizedBox(height: 18),
        _demoRow(),
      ],
    );
  }

  Widget _reconnectGuide() {
    return Container(
      padding: const EdgeInsets.fromLTRB(17, 16, 17, 16),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.lightbulb, size: 19, color: AppColors.caution),
              SizedBox(width: 8),
              Text(
                '이렇게 확인해 보세요',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                  color: AppColors.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _guideStep(1, '키링이 가까이 있고 전원이 켜져 있나요?'),
          const SizedBox(height: 10),
          _guideStep(2, '휴대폰 블루투스가 켜져 있나요?'),
        ],
      ),
    );
  }

  Widget _guideStep(int number, String text) {
    return Row(
      children: [
        Container(
          width: 24,
          height: 24,
          decoration: const BoxDecoration(
            color: AppColors.primarySoft,
            shape: BoxShape.circle,
          ),
          alignment: Alignment.center,
          child: Text(
            '$number',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w900,
              color: AppColors.primary,
            ),
          ),
        ),
        const SizedBox(width: 11),
        Expanded(
          child: Text(
            text,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: AppColors.textSecondary,
            ),
          ),
        ),
      ],
    );
  }

  // ----- listening -----
  Widget _listeningView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 16),
        Center(
          child: SizedBox(
            width: 208,
            height: 208,
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
              child: Container(
                width: 128,
                height: 128,
                decoration: BoxDecoration(
                  color: AppColors.card,
                  shape: BoxShape.circle,
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x2E1B6EF3),
                      blurRadius: 30,
                      offset: Offset(0, 10),
                    ),
                  ],
                ),
                child: const Icon(Icons.hearing,
                    size: 62, color: AppColors.primary),
              ),
            ),
          ),
        ),
        const SizedBox(height: 18),
        _waveform(),
        const SizedBox(height: 18),
        const Text(
          '듣고 있어요',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w900,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 5),
        const Text(
          '주변 소리를 실시간으로 분석합니다',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
            color: AppColors.textSecondary,
          ),
        ),
        const SizedBox(height: 26),
        _recentSection(),
        const SizedBox(height: 18),
        _demoRow(),
      ],
    );
  }

  Widget _ring(double t) {
    // t: 0 -> 1, scale .55 -> 1.65, opacity .5 -> 0
    final scale = 0.55 + t * 1.1;
    final opacity = (0.5 * (1 - t)).clamp(0.0, 1.0);
    return Opacity(
      opacity: opacity,
      child: Transform.scale(
        scale: scale,
        child: Container(
          width: 208,
          height: 208,
          decoration: BoxDecoration(
            color: AppColors.primary.withValues(alpha: 0.12),
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }

  Widget _waveform() {
    const phases = [0.0, 0.15, 0.3, 0.45, 0.6, 0.3, 0.1];
    return SizedBox(
      height: 30,
      child: AnimatedBuilder(
        animation: _wave,
        builder: (_, _) {
          return Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              for (final phase in phases) ...[
                _waveBar(phase),
                const SizedBox(width: 5),
              ],
            ],
          );
        },
      ),
    );
  }

  Widget _waveBar(double phase) {
    final v = (math.sin((_wave.value + phase) * 2 * math.pi) + 1) / 2;
    final scale = 0.35 + v * 0.65;
    return Container(
      width: 5,
      height: 30 * scale,
      decoration: BoxDecoration(
        color: AppColors.primary,
        borderRadius: BorderRadius.circular(3),
      ),
    );
  }

  Widget _recentSection() {
    final recent = widget.recentLogs.take(3).toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 0, 4, 10),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '최근 감지',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w800,
                  color: AppColors.textPrimary,
                ),
              ),
              GestureDetector(
                onTap: widget.onShowAllLogs,
                child: const Text(
                  '전체보기',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: AppColors.primary,
                  ),
                ),
              ),
            ],
          ),
        ),
        if (recent.isEmpty)
          Container(
            padding: const EdgeInsets.symmetric(vertical: 28),
            decoration: BoxDecoration(
              color: AppColors.card,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.cardBorder),
            ),
            alignment: Alignment.center,
            child: const Text(
              '아직 감지된 소리가 없어요',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: AppColors.textMuted,
              ),
            ),
          )
        else
          for (final p in recent) ...[
            DetectionTile(packet: p),
            const SizedBox(height: 10),
          ],
      ],
    );
  }

  // ----- demo trigger row (데모 입력) -----
  Widget _demoRow() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.fromLTRB(4, 0, 4, 8),
          child: Text(
            '데모 입력',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w800,
              color: AppColors.textMuted,
            ),
          ),
        ),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _demoChip('개소리', widget.onMockDog),
            _demoChip('위험음', widget.onMockDanger),
            _demoChip('대기 상태', widget.onReset),
          ],
        ),
      ],
    );
  }

  Widget _demoChip(String label, VoidCallback onTap) {
    return OutlinedButton(
      onPressed: onTap,
      style: OutlinedButton.styleFrom(
        foregroundColor: AppColors.textSecondary,
        backgroundColor: AppColors.card,
        side: const BorderSide(color: AppColors.divider),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
        textStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
      ),
      child: Text(label),
    );
  }
}

/// 감지 방향 나침반 (디자인 A 프레임)
class CompassView extends StatelessWidget {
  final double angle; // 0=북(=뒤쪽/6시 방향), 시계방향
  final Color color;
  final Color soft;
  final IconData icon;
  final double size;

  const CompassView({
    super.key,
    required this.angle,
    required this.color,
    required this.soft,
    required this.icon,
    this.size = 196,
  });

  @override
  Widget build(BuildContext context) {
    final centerSize = size * 0.53;
    final iconSize = size * 0.265;
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: Size(size, size),
            painter: _CompassPainter(angle: angle, color: color),
          ),
          const _CompassLabel(
              text: '앞쪽', top: 8, color: AppColors.textFaint),
          _CompassLabel(text: '뒤쪽', bottom: 8, color: color, bold: true),
          Container(
            width: centerSize,
            height: centerSize,
            decoration: BoxDecoration(color: soft, shape: BoxShape.circle),
            child: Icon(icon, size: iconSize, color: color),
          ),
        ],
      ),
    );
  }
}

class _CompassLabel extends StatelessWidget {
  final String text;
  final double? top, bottom;
  final Color color;
  final bool bold;

  const _CompassLabel({
    required this.text,
    required this.color,
    this.top,
    this.bottom,
    this.bold = false,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: top,
      bottom: bottom,
      child: Text(
        text,
        style: TextStyle(
          fontSize: bold ? 13 : 12,
          fontWeight: bold ? FontWeight.w900 : FontWeight.w700,
          color: color,
        ),
      ),
    );
  }
}

class _CompassPainter extends CustomPainter {
  final double angle;
  final Color color;

  _CompassPainter({required this.angle, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.width / 2 - 8;

    // base ring
    canvas.drawCircle(
      center,
      radius,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6
        ..color = AppColors.cardBorder,
    );

    // 0도(=북)를 6시 방향(뒤쪽)에 오도록 회전
    final rad = (angle + 90) * math.pi / 180;

    // highlighted arc segment around direction
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      rad - (17 * math.pi / 180),
      34 * math.pi / 180,
      false,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 9
        ..strokeCap = StrokeCap.round
        ..color = color,
    );

    // pointer dot
    final dot = Offset(
      center.dx + radius * math.cos(rad),
      center.dy + radius * math.sin(rad),
    );
    canvas.drawCircle(
      dot,
      13,
      Paint()..color = color.withValues(alpha: 0.16),
    );
    canvas.drawCircle(dot, 7, Paint()..color = color);
  }

  @override
  bool shouldRepaint(covariant _CompassPainter old) =>
      old.angle != angle || old.color != color;
}
