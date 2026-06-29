import 'package:shared_preferences/shared_preferences.dart';

/// 온보딩 / 사용자 설정 등 앱 단위 플래그 저장
class AppPrefs {
  static const _onboardingSeenKey = 'onboarding_seen';
  static const _backgroundAlertsKey = 'background_alerts';

  static Future<bool> isOnboardingSeen() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_onboardingSeenKey) ?? false;
  }

  static Future<void> setOnboardingSeen(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_onboardingSeenKey, value);
  }

  static Future<bool> isBackgroundAlertsEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_backgroundAlertsKey) ?? false;
  }

  static Future<void> setBackgroundAlertsEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_backgroundAlertsKey, value);
  }
}
