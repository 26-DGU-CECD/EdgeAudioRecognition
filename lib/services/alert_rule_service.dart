import '../models/sound_packet.dart';
import 'alert_settings_store.dart';

class AlertRuleService {
  static const double minNotificationScore = 0.30;
  static const Duration duplicateCooldown = Duration(seconds: 10);

  final Map<String, DateTime> _lastNotificationAtByLabelKey = {};

  Future<bool> shouldNotify(SoundPacket packet) async {
    if (packet.score < minNotificationScore) {
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

    final directionText =
        _relativeDirectionTextForCardinal(packet.direction) ??
        _relativeDirectionTextForCardinal(packet.directionText);

    return directionText ?? '방향 정보 없음';
  }

  double? _angleFromText(String text) {
    final match = RegExp(r'(\d+(?:\.\d+)?)').firstMatch(text);
    if (match == null) {
      return null;
    }

    return double.tryParse(match.group(1)!);
  }

  String _relativeDirectionTextForAngle(double angle) {
    final normalized = angle % 360.0;

    if (normalized < 45.0 || normalized >= 315.0) {
      return '앞쪽입니다';
    }
    if (normalized < 135.0) {
      return '오른쪽입니다';
    }
    if (normalized < 225.0) {
      return '뒤쪽입니다';
    }
    return '왼쪽입니다';
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
