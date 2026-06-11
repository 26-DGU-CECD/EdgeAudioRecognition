import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class LocalNotificationService {
  LocalNotificationService._();

  static final LocalNotificationService instance = LocalNotificationService._();

  static const String alertChannelId = 'sound_alerts';
  static const String alertChannelName = 'Sound alerts';
  static const String alertChannelDescription =
      'Notifications for detected environmental sounds.';

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) {
      return;
    }

    const androidSettings = AndroidInitializationSettings(
      '@mipmap/ic_launcher',
    );
    const settings = InitializationSettings(android: androidSettings);

    await _plugin.initialize(settings: settings);

    const channel = AndroidNotificationChannel(
      alertChannelId,
      alertChannelName,
      description: alertChannelDescription,
      importance: Importance.high,
      playSound: true,
      enableVibration: true,
    );

    await _plugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.createNotificationChannel(channel);

    _initialized = true;
  }

  Future<bool?> requestAndroidNotificationPermission() async {
    await initialize();
    return _plugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.requestNotificationsPermission();
  }

  Future<void> showSoundAlert({
    required String title,
    required String body,
    required String labelKey,
  }) async {
    await initialize();

    const androidDetails = AndroidNotificationDetails(
      alertChannelId,
      alertChannelName,
      channelDescription: alertChannelDescription,
      importance: Importance.high,
      priority: Priority.high,
      playSound: true,
      enableVibration: true,
      autoCancel: true,
    );

    await _plugin.show(
      id: _notificationIdFor(labelKey),
      title: title,
      body: body,
      notificationDetails: const NotificationDetails(android: androidDetails),
    );
  }

  int _notificationIdFor(String labelKey) {
    return labelKey.codeUnits.fold<int>(17, (hash, unit) {
          return (hash * 37 + unit) & 0x7fffffff;
        }) %
        2147483647;
  }
}
