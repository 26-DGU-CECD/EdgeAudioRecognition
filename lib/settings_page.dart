import 'package:flutter/material.dart';

import 'device_status.dart';
import 'setting_device_info_page.dart';
import 'setting_notification_page.dart';

class SettingsPage extends StatelessWidget {
  final DeviceStatus? deviceStatus;
  final List<String> soundLabels;
  final Set<String> mutedLabels;
  final bool backgroundAlertsEnabled;
  final int logCount;
  final ValueChanged<String> onToggleMutedLabel;
  final ValueChanged<bool> onToggleBackgroundAlerts;
  final VoidCallback onClearLogs;

  const SettingsPage({
    super.key,
    required this.deviceStatus,
    required this.soundLabels,
    required this.mutedLabels,
    required this.backgroundAlertsEnabled,
    required this.logCount,
    required this.onToggleMutedLabel,
    required this.onToggleBackgroundAlerts,
    required this.onClearLogs,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('설정')),
      body: ListView(
        children: [
          ListTile(
            leading: const Icon(Icons.memory),
            title: const Text('기기 정보'),
            subtitle: Text(deviceStatus?.deviceName.isNotEmpty == true
                ? deviceStatus!.deviceName
                : '연결된 기기 정보 없음'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => SettingDeviceInfoPage(
                    deviceStatus: deviceStatus,
                  ),
                ),
              );
            },
          ),
          ListTile(
            leading: const Icon(Icons.notifications_active),
            title: const Text('알림 설정'),
            subtitle: Text('차단된 소리 ${mutedLabels.length}개'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => SettingNotificationPage(
                    soundLabels: soundLabels,
                    mutedLabels: mutedLabels,
                    onToggleMutedLabel: onToggleMutedLabel,
                  ),
                ),
              );
            },
          ),
          SwitchListTile(
            secondary: const Icon(Icons.notifications_paused),
            value: backgroundAlertsEnabled,
            onChanged: onToggleBackgroundAlerts,
            title: const Text('백그라운드 기기 알림'),
            subtitle: const Text('Foreground Service 연동 후 지속 수신에 사용'),
          ),
          ListTile(
            leading: const Icon(Icons.delete_outline),
            title: const Text('로컬 로그 초기화'),
            subtitle: Text('현재 로그 $logCount개'),
            trailing: OutlinedButton(
              onPressed: onClearLogs,
              child: const Text('초기화'),
            ),
          ),
        ],
      ),
    );
  }
}
