import 'package:shared_preferences/shared_preferences.dart';

import '../models/sound_packet.dart';

class AlertSettingsStore {
  static const _backgroundAlertsEnabledKey = 'background_alerts_enabled';
  static const _backgroundAlertConsentSeenKey = 'background_alert_consent_seen';
  static const _mutedLabelKeysKey = 'muted_sound_label_keys';

  static Future<bool> loadBackgroundAlertsEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    return prefs.getBool(_backgroundAlertsEnabledKey) ?? false;
  }

  static Future<void> saveBackgroundAlertsEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_backgroundAlertsEnabledKey, enabled);
  }

  static Future<bool> loadBackgroundAlertConsentSeen() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    return prefs.getBool(_backgroundAlertConsentSeenKey) ?? false;
  }

  static Future<void> saveBackgroundAlertConsentSeen(bool seen) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_backgroundAlertConsentSeenKey, seen);
  }

  static Future<bool> shouldShowBackgroundAlertConsent() async {
    final backgroundAlertsEnabled = await loadBackgroundAlertsEnabled();
    final consentSeen = await loadBackgroundAlertConsentSeen();
    return !backgroundAlertsEnabled && !consentSeen;
  }

  static Future<Set<String>> loadMutedLabelKeys() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    return (prefs.getStringList(_mutedLabelKeysKey) ?? const <String>[])
        .toSet();
  }

  static Future<void> saveMutedLabelKeys(Set<String> keys) async {
    final prefs = await SharedPreferences.getInstance();
    final sorted = keys.toList()..sort();
    await prefs.setStringList(_mutedLabelKeysKey, sorted);
  }

  static String labelKeyForDisplayLabel(String label) {
    return soundLabelKeyFor(label);
  }

  static String labelKeyForPacket(SoundPacket packet) {
    return packet.notificationLabelKey;
  }

  static Future<bool> isPacketAllowed(SoundPacket packet) async {
    final mutedKeys = await loadMutedLabelKeys();
    return !mutedKeys.contains(labelKeyForPacket(packet));
  }
}
