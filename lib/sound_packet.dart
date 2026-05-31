/// top-k 결과가 오면 이 클래스로 변환
class TopKItem{
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
    return TopKItem(
      label: json['label'] ?? '',
      displayLabel: json['display_label'] ?? json['label'] ?? '',
      score: (json['score'] ?? 0).toDouble(),
      direction: json['direction'] ?? '',
    );
  }
}

/// 키링으로부터 수신한 json 전체
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

    return SoundPacket(
      status: json['status'] ?? '',
      time: json['time'] ?? '',
      label: json['label'] ?? '',
      displayLabel: json['display_label'] ?? json['label'],
      score: (json['score'] ?? 0).toDouble(),
      inferSec: (json['infer_sec'] ?? 0).toDouble(),
      totalSec: (json['total_sec'] ?? 0).toDouble(),
      db: (json['db'] ?? 0).toDouble(),
      level: json['level'] ?? 'info',
      direction: json['direction'] ?? '',
      angle: (json['angle'] ?? 0).toDouble(),
      angleRaw: (json['angle_raw'] ?? 0).toDouble(),
      directionText: json['direction_text'] ?? '',
      doaStatus: json['doa_status'] ?? '',
      raw: json['raw'] ?? '',
      items: rawItems is List
          ? rawItems.map((itemJson) => TopKItem.fromJson(itemJson)).toList()
          : [],
    );
  }

  /// 알림 기준 threshold
  bool get isDisplayable {
    return status == 'ok' && score >= 0.7 && db >= 45.0;
  }

  /// 알림 위험도
  bool get isDanger {
    return level == 'danger';
  }
}