import 'package:flutter/material.dart';

import '../sound_packet.dart';
import 'app_colors.dart';
import 'sound_style.dart';

/// 둥근 모서리 색상 아이콘 칩 (소리 픽토그램 용)
class IconSquare extends StatelessWidget {
  final IconData icon;
  final Color color;
  final Color background;
  final double size;
  final double radius;
  final double iconSize;

  const IconSquare({
    super.key,
    required this.icon,
    required this.color,
    required this.background,
    this.size = 48,
    this.radius = 14,
    this.iconSize = 25,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(radius),
      ),
      alignment: Alignment.center,
      child: Icon(icon, size: iconSize, color: color),
    );
  }
}

/// 소리 픽토그램. 디자인 PNG 에셋이 있으면 그걸 쓰고, 없으면 Material 아이콘 칩으로 대체.
class SoundIcon extends StatelessWidget {
  final String label;
  final String level;
  final double size;
  final double radius;
  final double iconSize;

  const SoundIcon({
    super.key,
    required this.label,
    required this.level,
    this.size = 48,
    this.radius = 14,
    this.iconSize = 25,
  });

  @override
  Widget build(BuildContext context) {
    final style = LevelStyle.of(level);
    final asset = assetForSoundLabel(label);

    Widget fallback() => IconSquare(
          icon: iconForSoundLabel(label),
          color: style.color,
          background: style.soft,
          size: size,
          radius: radius,
          iconSize: iconSize,
        );

    if (asset == null) return fallback();

    return Image.asset(
      asset,
      width: size,
      height: size,
      fit: BoxFit.contain,
      errorBuilder: (_, _, _) => fallback(),
    );
  }
}

/// 위험도 뱃지 (긴급 / 주의 / 정보)
class LevelBadge extends StatelessWidget {
  final String level;

  const LevelBadge({super.key, required this.level});

  @override
  Widget build(BuildContext context) {
    final style = LevelStyle.of(level);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
      decoration: BoxDecoration(
        color: style.soft,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        style.label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w800,
          color: style.color,
        ),
      ),
    );
  }
}

/// 감지 기록 / 최근 감지 공용 리스트 타일
class DetectionTile extends StatelessWidget {
  final SoundPacket packet;
  final String? trailingTime;
  final VoidCallback? onTap;

  const DetectionTile({
    super.key,
    required this.packet,
    this.trailingTime,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final time = trailingTime ?? packet.time;

    return Material(
      color: AppColors.card,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(18),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 13),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.cardBorder),
          ),
          child: Row(
            children: [
              SoundIcon(
                label: packet.displayLabel,
                level: packet.level,
              ),
              const SizedBox(width: 13),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      packet.displayLabel,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Row(
                      children: [
                        const Icon(Icons.navigation,
                            size: 14, color: AppColors.textMuted),
                        const SizedBox(width: 6),
                        Flexible(
                          child: Text(
                            _subtitle(packet),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                              color: AppColors.textMuted,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  LevelBadge(level: packet.level),
                  const SizedBox(height: 6),
                  Text(
                    time,
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF717A88),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _subtitle(SoundPacket p) {
    final dir = p.directionLabel;
    final db = '${p.db.toStringAsFixed(0)}dB';
    return '$dir · $db';
  }
}

/// 섹션 구분 라벨 (오늘 / 이번 주 ...)
class SectionDivider extends StatelessWidget {
  final String title;
  final String? count;

  const SectionDivider({super.key, required this.title, this.count});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 14, 4, 8),
      child: Row(
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w900,
              color: AppColors.textPrimary,
              letterSpacing: 0.3,
            ),
          ),
          if (count != null) ...[
            const SizedBox(width: 8),
            Text(
              count!,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: AppColors.textFaint,
              ),
            ),
          ],
          const SizedBox(width: 8),
          const Expanded(child: Divider(color: AppColors.divider, height: 1)),
        ],
      ),
    );
  }
}
