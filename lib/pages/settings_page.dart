import 'package:flutter/material.dart';

import '../models/device_status.dart';
import '../onboarding/onboarding_page.dart';
import 'setting_device_info_page.dart';
import 'setting_notification_page.dart';
import '../ui/app_colors.dart';
import '../ui/app_widgets.dart';

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
    final connected = deviceStatus?.connection == 'connected';
    final deviceName = deviceStatus?.deviceName.isNotEmpty == true
        ? deviceStatus!.deviceName
        : 'miimo 키링';
    final battery = deviceStatus?.battery;

    return SafeArea(
      bottom: false,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(18, 6, 18, 24),
        children: [
          const Padding(
            padding: EdgeInsets.fromLTRB(2, 0, 2, 18),
            child: Text(
              '설정',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w900,
                color: AppColors.textPrimary,
              ),
            ),
          ),

          // device card
          _Card(
            padding: const EdgeInsets.all(16),
            child: InkWell(
              borderRadius: BorderRadius.circular(22),
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) =>
                      SettingDeviceInfoPage(deviceStatus: deviceStatus),
                ),
              ),
              child: Row(
                children: [
                  Container(
                    width: 54,
                    height: 54,
                    decoration: BoxDecoration(
                      color: AppColors.primarySoft,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    padding: const EdgeInsets.all(2),
                    child: Image.asset('assets/miimo.png',
                        fit: BoxFit.contain),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          deviceName,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textPrimary,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Row(
                          children: [
                            Container(
                              width: 8,
                              height: 8,
                              decoration: BoxDecoration(
                                color: connected
                                    ? AppColors.success
                                    : AppColors.textMuted,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 6),
                            Text(
                              connected
                                  ? '연결됨${battery != null ? ' · 배터리 $battery%' : ' · 외부 전원'}'
                                  : '연결 안 됨',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                                color: connected
                                    ? AppColors.success
                                    : AppColors.textMuted,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const Icon(Icons.chevron_right,
                      color: AppColors.textFaint),
                ],
              ),
            ),
          ),
          const SizedBox(height: 18),

          // 알림 section
          const _SectionLabel('알림'),
          _Card(
            child: Column(
              children: [
                _SettingRow(
                  icon: Icons.notifications_active,
                  iconColor: AppColors.caution,
                  iconBg: AppColors.cautionSoft,
                  title: '알림 받을 소리',
                  subtitle: '차단된 소리 ${mutedLabels.length}개',
                  trailing: const Icon(Icons.chevron_right,
                      color: AppColors.textFaint),
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => SettingNotificationPage(
                        soundLabels: soundLabels,
                        mutedLabels: mutedLabels,
                        onToggleMutedLabel: onToggleMutedLabel,
                      ),
                    ),
                  ),
                ),
                const _RowDivider(),
                _SettingRow(
                  icon: Icons.notifications_paused,
                  iconColor: AppColors.primary,
                  iconBg: AppColors.primarySoft,
                  title: '백그라운드 알림',
                  subtitle: '앱이 꺼져 있어도 알려드려요',
                  trailing: Switch(
                    value: backgroundAlertsEnabled,
                    onChanged: onToggleBackgroundAlerts,
                    activeThumbColor: Colors.white,
                    activeTrackColor: AppColors.primary,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 18),

          // 데이터 section
          const _SectionLabel('데이터'),
          _Card(
            child: _SettingRow(
              icon: Icons.delete,
              iconColor: AppColors.danger,
              iconBg: AppColors.dangerSoft,
              title: '감지 기록 초기화',
              subtitle: '현재 기록 $logCount건',
              trailing: OutlinedButton(
                onPressed: () => _confirmClear(context),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.danger,
                  side: const BorderSide(
                      color: AppColors.dangerBorder, width: 1.5),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 8),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(999)),
                  textStyle: const TextStyle(
                      fontSize: 13, fontWeight: FontWeight.w800),
                ),
                child: const Text('초기화'),
              ),
            ),
          ),
          const SizedBox(height: 18),

          // 앱 section
          const _SectionLabel('앱'),
          _Card(
            child: _SettingRow(
              icon: Icons.slideshow,
              iconColor: AppColors.primary,
              iconBg: AppColors.primarySoft,
              title: '온보딩 다시보기',
              subtitle: '앱 사용법을 다시 볼 수 있어요',
              trailing: const Icon(Icons.chevron_right,
                  color: AppColors.textFaint),
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => const OnboardingPage(),
                ),
              ),
            ),
          ),
          const SizedBox(height: 20),
          const Center(
            child: Text(
              'Sound Keyring · v1.0.0',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: AppColors.textFaint,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _confirmClear(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        title: const Text('감지 기록 초기화'),
        content: Text('저장된 $logCount건의 기록을 모두 삭제할까요?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('취소',
                style: TextStyle(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () {
              onClearLogs();
              Navigator.pop(ctx);
            },
            child: const Text('초기화',
                style: TextStyle(color: AppColors.danger)),
          ),
        ],
      ),
    );
  }
}

class _Card extends StatelessWidget {
  final Widget child;
  final EdgeInsets? padding;

  const _Card({required this.child, this.padding});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.cardBorder),
      ),
      clipBehavior: Clip.antiAlias,
      child: child,
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(6, 0, 6, 9),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w800,
          color: AppColors.textMuted,
        ),
      ),
    );
  }
}

class _RowDivider extends StatelessWidget {
  const _RowDivider();

  @override
  Widget build(BuildContext context) {
    return const Divider(height: 1, thickness: 1, color: AppColors.line);
  }
}

class _SettingRow extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final Color iconBg;
  final String title;
  final String subtitle;
  final Widget trailing;
  final VoidCallback? onTap;

  const _SettingRow({
    required this.icon,
    required this.iconColor,
    required this.iconBg,
    required this.title,
    required this.subtitle,
    required this.trailing,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            IconSquare(
              icon: icon,
              color: iconColor,
              background: iconBg,
              size: 42,
              radius: 13,
              iconSize: 22,
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      color: AppColors.textMuted,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            trailing,
          ],
        ),
      ),
    );
  }
}
