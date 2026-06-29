import 'package:flutter/material.dart';

import 'ble/ble_connection_page.dart';
import 'ble/ble_sound_service.dart';
import 'device_status.dart';
import 'ui/app_colors.dart';

class SettingDeviceInfoPage extends StatelessWidget {
  final DeviceStatus? deviceStatus;

  const SettingDeviceInfoPage({super.key, required this.deviceStatus});

  @override
  Widget build(BuildContext context) {
    final status = deviceStatus;
    final connected = status?.connection == 'connected';
    final deviceName = status?.deviceName.isNotEmpty == true
        ? status!.deviceName
        : 'miimo 키링';
    final battery = status?.battery;
    final message = status?.message.isNotEmpty == true
        ? status!.message
        : (connected ? '정상 작동 중' : '-');

    return Scaffold(
      appBar: AppBar(title: const Text('기기 정보')),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(18, 6, 18, 18),
          children: [
            // hero
            Container(
              padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
              decoration: BoxDecoration(
                color: AppColors.card,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: AppColors.cardBorder),
              ),
              child: Column(
                children: [
                  Container(
                    width: 96,
                    height: 96,
                    padding: const EdgeInsets.all(5),
                    decoration: BoxDecoration(
                      color: AppColors.primarySoft,
                      borderRadius: BorderRadius.circular(28),
                    ),
                    child: Image.asset('assets/miimo.png', fit: BoxFit.contain),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    deviceName,
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w900,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      color: connected
                          ? AppColors.successSoft
                          : AppColors.dangerSoft,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: connected
                                ? AppColors.success
                                : AppColors.danger,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 7),
                        Text(
                          connected ? '연결됨' : '연결 끊김',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: connected
                                ? AppColors.successText
                                : AppColors.danger,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 18),
                  if (battery != null) ...[
                    // battery
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          '배터리',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textSecondary,
                          ),
                        ),
                        Text(
                          '$battery%',
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textPrimary,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 7),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: battery / 100,
                        minHeight: 10,
                        backgroundColor: AppColors.cardBorder,
                        valueColor: const AlwaysStoppedAnimation(
                            AppColors.success),
                      ),
                    ),
                  ] else
                    // 보조배터리/외부 전원: 잔량을 읽을 수 없음
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          '전원',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textSecondary,
                          ),
                        ),
                        Row(
                          children: const [
                            Icon(Icons.power, size: 16, color: AppColors.primary),
                            SizedBox(width: 6),
                            Text(
                              '외부 전원 (USB)',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: AppColors.primary,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // detail rows
            Container(
              decoration: BoxDecoration(
                color: AppColors.card,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.cardBorder),
              ),
              clipBehavior: Clip.antiAlias,
              child: Column(
                children: [
                  _row('기기 이름', deviceName),
                  const Divider(height: 1, color: AppColors.line),
                  _row('연결 상태', connected ? '연결됨' : '연결 끊김'),
                  const Divider(height: 1, color: AppColors.line),
                  _row('상태 메시지', message),
                ],
              ),
            ),
            const SizedBox(height: 16),

            SizedBox(
              height: 54,
              child: FilledButton.icon(
                onPressed: () => _goToConnection(context),
                icon: const Icon(Icons.sync, size: 20),
                label: const Text('기기 다시 연결'),
                style: FilledButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16)),
                  textStyle: const TextStyle(
                      fontSize: 15, fontWeight: FontWeight.w800),
                ),
              ),
            ),
            const SizedBox(height: 10),
            SizedBox(
              height: 54,
              child: OutlinedButton.icon(
                onPressed: () => _disconnect(context),
                icon: const Icon(Icons.link_off, size: 20),
                label: const Text('연결 해제'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.danger,
                  backgroundColor: AppColors.card,
                  side: const BorderSide(color: AppColors.dangerBorder),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16)),
                  textStyle: const TextStyle(
                      fontSize: 15, fontWeight: FontWeight.w800),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _row(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
      child: Row(
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: AppColors.textMuted,
            ),
          ),
          const Spacer(),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.right,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _goToConnection(BuildContext context) {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const BleConnectionPage()),
      (_) => false,
    );
  }

  Future<void> _disconnect(BuildContext context) async {
    await BleSoundService.instance.disconnect();
    if (!context.mounted) return;
    _goToConnection(context);
  }
}
