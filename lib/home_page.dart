import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/sound_packet.dart';
import '../widgets/detected_content.dart';

/// 홈 화면 UI
class HomePage extends StatelessWidget {
  final SoundPacket? packet;
  final String Function(String label) imagePathForLabel;
  final VoidCallback onReset;
  final VoidCallback onMockDog;
  final VoidCallback onMockDanger;

  const HomePage({
    super.key,
    required this.packet,
    required this.imagePathForLabel,
    required this.onReset,
    required this.onMockDog,
    required this.onMockDanger,
  });

  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    final isListening = packet == null;
    final color = packet?.isDanger == true ? Colors.redAccent : Colors.cyanAccent;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Sound Recognition'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(
              child: Text(
                isListening ? 'LISTENING' : packet!.level.toUpperCase(),
                style: TextStyle(color: color, fontWeight: FontWeight.bold),
              ),
            ),
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: DirectionCircle(
                  angle: packet?.angle,
                  isDanger: packet?.isDanger == true,
                  child: isListening
                      ? const ListeningContent()
                      : DetectedContent(
                    packet: packet!,
                    imagePath: imagePathForLabel(packet!.displayLabel),
                    color: color,
                  ),
                ),
              ),
            ),
            InfoPanel(packet: packet),
            const SizedBox(height: 12,),

            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ElevatedButton(onPressed: onReset, child: const Text('듣기 상태')),
                ElevatedButton(onPressed: onMockDog, child: Text('개소리 수신 테스트')),
                ElevatedButton(onPressed: onMockDanger, child: const Text('위험 수신 테스트')),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// 방향 표시 원 그리기
class DirectionCircle extends StatelessWidget {
  final double? angle;
  final bool isDanger;
  final Widget child;

  const DirectionCircle({
    super.key,
    required this.angle,
    required this.child,
    this.isDanger = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 340,
      height: 340,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: const Size(340, 340),
            painter: DirectionCirclePainter(
              angle: angle,
              isDanger: isDanger,
            ),
          ),
          child,
        ],
      ),
    );
  }
}

/// 원, 방향 바늘 그리기
class DirectionCirclePainter extends CustomPainter {
  final double? angle;
  final bool isDanger;

  DirectionCirclePainter({
    required this.angle,
    this.isDanger = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final outerRadius = size.width / 2;
    final innerRadius = outerRadius * 0.88;

    final baseColor = isDanger ? Colors.redAccent : Colors.cyan;

    final outerPaint = Paint()
      ..color = baseColor.withOpacity(0.35)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final innerPaint = Paint()
      ..color = Colors.purple.withOpacity(0.25)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    canvas.drawCircle(center, outerRadius - 4, outerPaint);
    canvas.drawCircle(center, innerRadius, innerPaint);

    if (angle != null) {
      _drawDirectionHighlight(
        canvas,
        center,
        outerRadius,
        innerRadius,
        angle!,
        baseColor,
      );

      _drawOuterPointer(
        canvas,
        center,
        outerRadius,
        angle!,
        baseColor,
      );
    }
  }

  void _drawDirectionHighlight(
      Canvas canvas,
      Offset center,
      double outerRadius,
      double innerRadius,
      double angle,
      Color color,
      ) {
    final startAngle = (angle - 90 - 16) * math.pi / 180;
    const sweepAngle = 32 * math.pi / 180;

    final rect = Rect.fromCircle(
      center: center,
      radius: (outerRadius + innerRadius) / 2,
    );

    final highlightPaint = Paint()
      ..color = color.withOpacity(0.28)
      ..style = PaintingStyle.stroke
      ..strokeWidth = outerRadius - innerRadius
      ..strokeCap = StrokeCap.round;

    final glowPaint = Paint()
      ..color = color.withOpacity(0.13)
      ..style = PaintingStyle.stroke
      ..strokeWidth = (outerRadius - innerRadius) + 18
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);

    canvas.drawArc(rect, startAngle, sweepAngle, false, glowPaint);
    canvas.drawArc(rect, startAngle, sweepAngle, false, highlightPaint);
  }

  void _drawOuterPointer(
      Canvas canvas,
      Offset center,
      double outerRadius,
      double angle,
      Color color,
      ) {
    final radians = (angle - 90) * math.pi / 180;

    final direction = Offset(
      math.cos(radians),
      math.sin(radians),
    );

    final perpendicular = Offset(
      -math.sin(radians),
      math.cos(radians),
    );

    final tip = Offset(
      center.dx + (outerRadius + 18) * direction.dx,
      center.dy + (outerRadius + 18) * direction.dy,
    );

    final baseCenter = Offset(
      center.dx + (outerRadius - 8) * direction.dx,
      center.dy + (outerRadius - 8) * direction.dy,
    );

    const pointerWidth = 34.0;

    final left = Offset(
      baseCenter.dx + perpendicular.dx * pointerWidth / 2,
      baseCenter.dy + perpendicular.dy * pointerWidth / 2,
    );

    final right = Offset(
      baseCenter.dx - perpendicular.dx * pointerWidth / 2,
      baseCenter.dy - perpendicular.dy * pointerWidth / 2,
    );

    final pointerPath = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(left.dx, left.dy)
      ..lineTo(right.dx, right.dy)
      ..close();

    final glowPathPaint = Paint()
      ..color = color.withOpacity(0.20)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);

    final pointerPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    canvas.drawPath(pointerPath, glowPathPaint);
    canvas.drawPath(pointerPath, pointerPaint);

    final dotCenter = Offset(
      center.dx + (outerRadius - 20) * direction.dx,
      center.dy + (outerRadius - 20) * direction.dy,
    );

    canvas.drawCircle(
      dotCenter,
      15,
      Paint()
        ..color = color.withOpacity(0.18)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8),
    );

    canvas.drawCircle(
      dotCenter,
      6,
      Paint()..color = color,
    );
  }

  @override
  bool shouldRepaint(covariant DirectionCirclePainter oldDelegate) {
    return oldDelegate.angle != angle || oldDelegate.isDanger != isDanger;
  }
}

/// 홈화면 하단 소리 정보
class InfoPanel extends StatelessWidget {
  final SoundPacket? packet;

  const InfoPanel({super.key, required this.packet});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(child: infoBox('INFER', '${packet?.inferSec ?? 0.0}s')),
            Expanded(child: infoBox('DB', '${packet?.db ?? 0.0}dB')),
            /// Expanded(child: infoBox('SCORE', packet == null ? '0.0%' : '${(packet!.score *100).toStringAsFixed(1)}%',),),
          ],
        ),
      ],
    );
  }

  Widget infoBox(String title, String value) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 4),
      padding: const EdgeInsets.all(12),
      color: const Color(0xFFF3F5F7),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontSize: 11)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}