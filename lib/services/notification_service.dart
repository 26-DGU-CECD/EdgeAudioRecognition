import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import '../sound_packet.dart';
import '../ui/sound_style.dart';

/// 소리 감지 시 OS 푸시 알림(잠금화면 P1 / 알림센터 P2 / 헤즈업 P3)을 띄운다.
/// 디자인의 둥근 카드는 OS가 렌더링하므로, 레벨 색상·소리 이미지·텍스트로 반영한다.
class NotificationService {
  NotificationService._();

  static final NotificationService instance = NotificationService._();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  bool _ready = false;
  int _idSeq = 0;

  /// 레벨별 채널. 긴급은 헤즈업(P3)이 강하게 뜨도록 max 중요도.
  static const AndroidNotificationChannel _dangerChannel =
      AndroidNotificationChannel(
    'sound_danger',
    '긴급 소리 감지',
    description: '유리 깨짐, 경보음 등 긴급 소리 알림',
    importance: Importance.max,
  );
  static const AndroidNotificationChannel _cautionChannel =
      AndroidNotificationChannel(
    'sound_caution',
    '주의 소리 감지',
    description: '차량 경적 등 주의 소리 알림',
    importance: Importance.high,
  );
  static const AndroidNotificationChannel _infoChannel =
      AndroidNotificationChannel(
    'sound_info',
    '정보 소리 감지',
    description: '물 흐르는 소리 등 일반 소리 알림',
    importance: Importance.high,
  );

  Future<void> init() async {
    const androidInit = AndroidInitializationSettings('@mipmap/ic_launcher');
    const settings = InitializationSettings(android: androidInit);
    await _plugin.initialize(settings);

    final android = _plugin.resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin>();
    if (android != null) {
      await android.createNotificationChannel(_dangerChannel);
      await android.createNotificationChannel(_cautionChannel);
      await android.createNotificationChannel(_infoChannel);
      await android.requestNotificationsPermission();
    }
    _ready = true;
  }

  AndroidNotificationChannel _channelFor(String level) {
    switch (level) {
      case 'danger':
        return _dangerChannel;
      case 'caution':
        return _cautionChannel;
      default:
        return _infoChannel;
    }
  }

  /// 감지된 소리 패킷으로 알림 표시.
  Future<void> showForPacket(SoundPacket packet) async {
    if (!_ready) return;

    final channel = _channelFor(packet.level);
    final style = LevelStyle.of(packet.level);
    final body = '${packet.directionLabel} 방향에서 감지';

    // 소리 이미지(assets/sounds/*.png)를 라지 아이콘으로 사용
    ByteArrayAndroidBitmap? largeIcon;
    final assetPath = assetForSoundLabel(packet.displayLabel);
    if (assetPath != null) {
      try {
        final data = await rootBundle.load(assetPath);
        largeIcon = ByteArrayAndroidBitmap(data.buffer.asUint8List());
      } catch (_) {
        largeIcon = null;
      }
    }

    final androidDetails = AndroidNotificationDetails(
      channel.id,
      channel.name,
      channelDescription: channel.description,
      importance: channel.importance,
      priority: packet.level == 'danger' ? Priority.max : Priority.high,
      category: packet.isDanger
          ? AndroidNotificationCategory.alarm
          : AndroidNotificationCategory.message,
      color: style.color,
      colorized: false,
      largeIcon: largeIcon,
      ticker: packet.displayLabel,
      subText: '소리감지',
      styleInformation: BigTextStyleInformation(
        body,
        contentTitle: packet.displayLabel,
        summaryText: '${style.label} · 소리감지',
      ),
    );

    // 31비트 범위 내 id
    final id = _idSeq = (_idSeq + 1) & 0x7fffffff;
    await _plugin.show(
      id,
      packet.displayLabel,
      body,
      NotificationDetails(android: androidDetails),
    );
  }
}
