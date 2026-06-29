import 'package:flutter/material.dart';

import 'app_colors.dart';

/// 위험도(level)에 따른 색/뱃지 텍스트 묶음
class LevelStyle {
  final String label; // 긴급 / 주의 / 정보
  final Color color; // 강조색 (아이콘, 텍스트)
  final Color soft; // 배경색

  const LevelStyle({
    required this.label,
    required this.color,
    required this.soft,
  });

  static LevelStyle of(String level) {
    switch (level) {
      case 'danger':
        return const LevelStyle(
          label: '긴급',
          color: AppColors.danger,
          soft: AppColors.dangerSoft,
        );
      case 'caution':
        return const LevelStyle(
          label: '주의',
          color: AppColors.cautionText,
          soft: AppColors.cautionSoft,
        );
      default:
        return const LevelStyle(
          label: '정보',
          color: AppColors.primary,
          soft: AppColors.primarySoft,
        );
    }
  }
}

/// 소리 라벨에 어울리는 Material 아이콘 매핑 (디자인의 픽토그램 대체)
/// 소리 라벨 → 디자인 에셋(assets/sounds/*.png) 경로. 매칭되는 게 없으면 null.
String? assetForSoundLabel(String label) {
  final l = label.replaceAll(' ', '');
  String? slug;
  if (l.contains('유리')) {
    slug = 'glass-break';
  } else if (l.contains('총')) {
    slug = 'gunshot';
  } else if (l.contains('응급') ||
      l.contains('도난') ||
      l.contains('화재') ||
      l.contains('경보')) {
    slug = 'emergency-alarm';
  } else if (l.contains('공사') || l.contains('소음')) {
    slug = 'construction-noise';
  } else if (l.contains('자전거')) {
    slug = 'bicycle-approach';
  } else if (l.contains('경적') || l.contains('차량') || l.contains('자동차')) {
    slug = 'car-horn';
  } else if (l.contains('노크')) {
    slug = 'knock';
  } else if (l.contains('가전')) {
    slug = 'appliance';
  } else if (l.contains('물')) {
    slug = 'water-running';
  } else if (l.contains('아기')) {
    slug = 'baby-cry';
  } else if (l.contains('개') || l.contains('고양이') || l.contains('동물')) {
    slug = 'pet-cry';
  } else if (l.contains('비명') || l.contains('사람') || l.contains('울음')) {
    slug = 'scream';
  }
  return slug == null ? null : 'assets/sounds/$slug.png';
}

IconData iconForSoundLabel(String label) {
  final l = label.replaceAll(' ', '');
  if (l.contains('유리')) return Icons.broken_image;
  if (l.contains('총')) return Icons.crisis_alert;
  if (l.contains('응급') ||
      l.contains('도난') ||
      l.contains('화재') ||
      l.contains('경보')) {
    return Icons.emergency;
  }
  if (l.contains('공사') || l.contains('소음')) return Icons.construction;
  if (l.contains('자전거')) return Icons.directions_bike;
  if (l.contains('경적') || l.contains('차량') || l.contains('자동차')) {
    return Icons.directions_car;
  }
  if (l.contains('노크')) return Icons.door_front_door;
  if (l.contains('가전')) return Icons.kitchen;
  if (l.contains('물')) return Icons.water_drop;
  if (l.contains('아기')) return Icons.child_care;
  if (l.contains('개') || l.contains('고양이') || l.contains('동물')) {
    return Icons.pets;
  }
  if (l.contains('비명') || l.contains('사람') || l.contains('울음')) {
    return Icons.record_voice_over;
  }
  return Icons.graphic_eq;
}
