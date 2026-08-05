import '../models/sound_packet.dart';
import 'alert_settings_store.dart';

class AlertRuleService {
  /// 알림 판정은 `packet.isConfident`가 담당한다. 이 값은 그 임계값과
  /// 같은 값을 외부에 노출하기 위한 별칭이다.
  static const double minNotificationScore = kMinConfidenceScore;
  static const Duration duplicateCooldown = Duration(seconds: 10);

  final Map<String, DateTime> _lastNotificationAtByLabelKey = {};

  Future<bool> shouldNotify(SoundPacket packet) async {
    if (!packet.isConfident) {
      return false;
    }

    if (!await AlertSettingsStore.isPacketAllowed(packet)) {
      return false;
    }

    final labelKey = packet.notificationLabelKey;
    final now = DateTime.now();
    final lastNotificationAt = _lastNotificationAtByLabelKey[labelKey];

    if (lastNotificationAt != null &&
        now.difference(lastNotificationAt) < duplicateCooldown) {
      return false;
    }

    _lastNotificationAtByLabelKey[labelKey] = now;
    return true;
  }

  String titleFor(SoundPacket packet) {
    switch (packet.level.trim().toLowerCase()) {
      case 'danger':
        return '위험음 감지';
      case 'caution':
        return '주의음 감지';
      default:
        return '환경음 감지';
    }
  }

  String bodyFor(SoundPacket packet) {
    final label = packet.notificationLabel;
    final directionText = _relativeDirectionTextFor(packet);
    final dbText = '${packet.db.toStringAsFixed(1)}dB';

    return '$label 감지 / $directionText / $dbText';
  }

  String _relativeDirectionTextFor(SoundPacket packet) {
    final angle = packet.hasDirectionAngle
        ? packet.angle
        : _angleFromText(packet.directionText);

    if (angle != null) {
      return _relativeDirectionTextForAngle(angle);
    }

    final directionText = _relativeDirectionTextForCardinal(packet.directionText);

    return directionText ?? '방향 정보 없음';
  }

  double? _angleFromText(String text) {
    final match = RegExp(r'(\d+(?:\.\d+)?)').firstMatch(text);
    if (match == null) {
      return null;
    }

    return double.tryParse(match.group(1)!);
  }

  /// angle 0=뒤쪽, 90=왼쪽, 180=앞쪽, 270=오른쪽 (시계방향).
  /// `SoundPacket.directionLabel`, `home_page.dart`의 CompassView와 같은 규칙이다.
  /// 예전에는 이 함수만 0=앞쪽으로 가정해서 나침반과 정확히 180도 어긋났다.
  String _relativeDirectionTextForAngle(double angle) {
    final normalized = ((angle % 360.0) + 360.0) % 360.0;

    if (normalized < 45.0 || normalized >= 315.0) {
      return '뒤쪽입니다';
    }
    if (normalized < 135.0) {
      return '왼쪽입니다';
    }
    if (normalized < 225.0) {
      return '앞쪽입니다';
    }
    return '오른쪽입니다';
  }

  String? _relativeDirectionTextForCardinal(String text) {
    final normalized = text.trim().toLowerCase();
    if (normalized.isEmpty) {
      return null;
    }

    if (normalized.contains('북') || normalized.contains('north')) {
      return '앞쪽입니다';
    }
    if (normalized.contains('동') || normalized.contains('east')) {
      return '오른쪽입니다';
    }
    if (normalized.contains('남') || normalized.contains('south')) {
      return '뒤쪽입니다';
    }
    if (normalized.contains('서') || normalized.contains('west')) {
      return '왼쪽입니다';
    }

    return null;
  }
}
