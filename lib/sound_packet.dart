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

  /// angle(0~360)로부터 사용자 기준 방향 라벨 계산.
  /// 0도(=기기의 북쪽)는 사용자의 뒤쪽(6시 방향)에 대응한다.
  /// 원을 X로 4등분: 0=뒤쪽, 90=왼쪽, 180=앞쪽, 270=오른쪽.
  /// 서버는 angle만 보내면 되고, 방향 텍스트는 앱에서 파생한다.
  String get directionLabel {
    const labels = ['뒤쪽', '왼쪽', '앞쪽', '오른쪽'];
    final a = ((angle % 360) + 360) % 360;
    final idx = (((a + 45) % 360) ~/ 90).toInt();
    return labels[idx];
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