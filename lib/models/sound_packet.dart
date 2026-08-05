/// 라벨을 사용자에게 노출하기 위한 최소 신뢰도(85%).
/// 홈 화면 표시, 배너, OS 알림, 감지 기록이 모두 이 값을 공유한다.
/// 클래스 구분 없이 전부 같은 임계값을 쓴다.
const double kMinConfidenceScore = 0.85;

/// 해당 라벨에 적용할 최소 신뢰도를 돌려준다.
/// 지금은 모든 클래스가 [kMinConfidenceScore]로 균일하다.
double minConfidenceScoreFor(String? label, [String? displayLabel]) {
  return kMinConfidenceScore;
}

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
  '경적',
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
  'bicycle_bell': '자전거',
  'bike_bell': '자전거',
  'water': '물소리',
  'water_sound': '물소리',
  'cry': '울음',
  'crying': '울음',
  'scream': '비명',
  'glass_break': '유리깨지는소리',
  'glass_breaking': '유리깨지는소리',
  'glass_shatter': '유리깨지는소리',
  'fire_alarm': '화재경보',
  'knock': '노크',
  'baby_cry': '아기 우는 소리',
  'baby_crying': '아기 우는 소리',
  'dog': '개소리',
  'dog_bark': '개소리',
  'cat': '고양이소리',
  'cat_meow': '고양이소리',
  'car_horn': '경적',
  'vehicle_horn': '경적',
  'horn': '경적',
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

String soundLabelKeyFor(String? label, [String? displayLabel]) {
  final displayLabelKey = displayNameForSound(label, displayLabel).trim();
  if (displayLabelKey.isEmpty) {
    return 'unknown';
  }

  return displayLabelKey.toLowerCase().replaceAll(RegExp(r'\s+'), '_');
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

bool _toBool(dynamic value) {
  if (value is bool) {
    return value;
  }
  if (value is num) {
    return value != 0;
  }
  if (value is String) {
    final normalized = value.trim().toLowerCase();
    return normalized == 'true' || normalized == '1' || normalized == 'yes';
  }
  return false;
}

/// top-k 결과가 오면 이 클래스로 변환합니다.
class TopKItem {
  final String label;
  final String displayLabel;
  final double score;

  TopKItem({
    required this.label,
    required this.displayLabel,
    required this.score,
  });

  factory TopKItem.fromJson(Map<String, dynamic> json) {
    final label = json['label']?.toString() ?? '';
    final displayLabel = json['display_label']?.toString();

    return TopKItem(
      label: label,
      displayLabel: displayNameForSound(label, displayLabel),
      score: _toDouble(json['score']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'label': label,
      'display_label': displayLabel,
      'score': score,
    };
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
  final double angle;
  final double angleRaw;
  final bool hasDirectionAngle;
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
    required this.angle,
    required this.angleRaw,
    required this.hasDirectionAngle,
    required this.directionText,
    required this.doaStatus,
    required this.raw,
    required this.items,
  });

  factory SoundPacket.fromJson(Map<String, dynamic> json) {
    final rawItems = json['items'];
    final label = json['label']?.toString() ?? '';
    final displayLabel = json['display_label']?.toString();
    final rawAngle = json['angle'];

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
      angle: _toDouble(rawAngle),
      angleRaw: _toDouble(json['angle_raw']),
      hasDirectionAngle: _toBool(json['has_doa']) || rawAngle != null,
      directionText: json['direction_text']?.toString() ?? '',
      doaStatus: json['doa_status']?.toString() ?? '',
      raw: json['raw']?.toString() ?? '',
      items: rawItems is List
          ? rawItems
                .whereType<Map>()
                .map((item) => TopKItem.fromJson(_stringKeyMap(item)))
                .toList()
          : [],
    );
  }

  String get notificationLabel {
    final display = displayLabel.trim();
    if (display.isNotEmpty) {
      return display;
    }
    return displayNameForSound(label);
  }

  String get notificationLabelKey {
    return soundLabelKeyFor(label, displayLabel);
  }

  Map<String, dynamic> toJson() {
    return {
      'status': status,
      'time': time,
      'label': label,
      'display_label': displayLabel,
      'score': score,
      'infer_sec': inferSec,
      'total_sec': totalSec,
      'db': db,
      'level': level,
      'angle': angle,
      'angle_raw': angleRaw,
      'has_doa': hasDirectionAngle,
      'direction_text': directionText,
      'doa_status': doaStatus,
      'raw': raw,
      'items': items.map((item) => item.toJson()).toList(),
    };
  }

  /// 이 패킷의 라벨에 적용되는 최소 신뢰도
  double get minConfidenceScore {
    return minConfidenceScoreFor(label, displayLabel);
  }

  /// 신뢰도가 충분해서 라벨을 믿고 보여줄 수 있는지 여부
  bool get isConfident {
    return status == 'ok' && score >= minConfidenceScore;
  }

  /// 알림 기준 threshold (홈 화면 표시 조건)
  bool get isDisplayable {
    return isConfident && db >= 45.0;
  }

  /// 방향(각도)을 한글 방위 텍스트로 변환
  String get directionLabel {
    const labels = ['뒤쪽', '왼쪽', '앞쪽', '오른쪽'];
    final a = ((angle % 360) + 360) % 360;
    final idx = (((a + 45) % 360) ~/ 90).toInt();
    return labels[idx];
  }

  /// 알림 위험도
  bool get isDanger {
    return level == 'danger';
  }
}

Map<String, dynamic> _stringKeyMap(Map source) {
  return source.map((key, value) => MapEntry(key.toString(), value));
}
