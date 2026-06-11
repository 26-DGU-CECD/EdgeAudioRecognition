const List<String> knownKoreanSoundLabels = [
  '총',
  '경보',
  '자전거',
  '물소리',
  '울음',
  '비명',
  '유리깨지는소리',
  '화재경보',
  '아기 우는 소리',
  '개소리',
  '고양이소리',
];

const Map<String, String> _soundLabelKoMap = {
  'gun': '총',
  'gunshot': '총',
  'gun_shot': '총',
  'explosion': '총',
  'alarm': '경보',
  'alarm_siren': '경보',
  'siren': '경보',
  'warning_alarm': '경보',
  'bicycle': '자전거',
  'bike': '자전거',
  'water': '물소리',
  'water_sound': '물소리',
  'cry': '울음',
  'crying': '울음',
  'scream': '비명',
  'glass_break': '유리깨지는소리',
  'glass_breaking': '유리깨지는소리',
  'glass_shatter': '유리깨지는소리',
  'fire_alarm': '화재경보',
  'baby_cry': '아기 우는 소리',
  'baby_crying': '아기 우는 소리',
  'dog': '개소리',
  'dog_bark': '개소리',
  'cat': '고양이소리',
  'cat_meow': '고양이소리',
};

String displayNameForSound(String? label, [String? displayLabel]) {
  final display = displayLabel?.trim();
  if (display != null && display.isNotEmpty) {
    if (knownKoreanSoundLabels.contains(display)) {
      return display;
    }

    final normalizedDisplay = display.toLowerCase().replaceAll(' ', '_');
    final mappedDisplay = _soundLabelKoMap[normalizedDisplay];
    if (mappedDisplay != null) {
      return mappedDisplay;
    }
  }

  final rawLabel = label?.trim();
  if (rawLabel == null || rawLabel.isEmpty) {
    return display?.isNotEmpty == true ? display! : '소리 감지';
  }

  if (knownKoreanSoundLabels.contains(rawLabel)) {
    return rawLabel;
  }

  final normalizedLabel = rawLabel.toLowerCase().replaceAll(' ', '_');
  return _soundLabelKoMap[normalizedLabel] ?? rawLabel;
}

double _toDouble(dynamic value) {
  if (value is num) {
    return value.toDouble();
  }
  if (value is String) {
    return double.tryParse(value) ?? 0.0;
  }
  return 0.0;
}

/// top-k 결과가 오면 이 클래스로 변환합니다.
class TopKItem {
  final String label;
  final String displayLabel;
  final double score;
  final String direction;

  TopKItem({
    required this.label,
    required this.displayLabel,
    required this.score,
    required this.direction,
  });

  factory TopKItem.fromJson(Map<String, dynamic> json) {
    final label = json['label']?.toString() ?? '';
    final displayLabel = json['display_label']?.toString();

    return TopKItem(
      label: label,
      displayLabel: displayNameForSound(label, displayLabel),
      score: _toDouble(json['score']),
      direction: json['direction']?.toString() ?? '',
    );
  }
}

/// 키링으로부터 수신한 JSON 전체입니다.
class SoundPacket {
  final String status;
  final String time;
  final String label;
  final String displayLabel;
  final double score;
  final double inferSec;
  final double totalSec;
  final double db;
  final String level;
  final String direction;
  final double angle;
  final double angleRaw;
  final String directionText;
  final String doaStatus;
  final String raw;
  final List<TopKItem> items;

  SoundPacket({
    required this.status,
    required this.time,
    required this.label,
    required this.displayLabel,
    required this.score,
    required this.inferSec,
    required this.totalSec,
    required this.db,
    required this.level,
    required this.direction,
    required this.angle,
    required this.angleRaw,
    required this.directionText,
    required this.doaStatus,
    required this.raw,
    required this.items,
  });

  factory SoundPacket.fromJson(Map<String, dynamic> json) {
    final rawItems = json['items'];
    final label = json['label']?.toString() ?? '';
    final displayLabel = json['display_label']?.toString();

    return SoundPacket(
      status: json['status']?.toString() ?? '',
      time: json['time']?.toString() ?? '',
      label: label,
      displayLabel: displayNameForSound(label, displayLabel),
      score: _toDouble(json['score']),
      inferSec: _toDouble(json['infer_sec']),
      totalSec: _toDouble(json['total_sec']),
      db: _toDouble(json['db']),
      level: json['level']?.toString() ?? 'info',
      direction: json['direction']?.toString() ?? '',
      angle: _toDouble(json['angle']),
      angleRaw: _toDouble(json['angle_raw']),
      directionText: json['direction_text']?.toString() ?? '',
      doaStatus: json['doa_status']?.toString() ?? '',
      raw: json['raw']?.toString() ?? '',
      items: rawItems is List
          ? rawItems
              .whereType<Map<String, dynamic>>()
              .map(TopKItem.fromJson)
              .toList()
          : [],
    );
  }

  /// 알림 기준 threshold
  /* bool get isDisplayable {
    return status == 'ok' && score >= 0.7 && db >= 30.0;
  } */

  /// 알림 위험도
  bool get isDanger {
    return level == 'danger';
  }
}
