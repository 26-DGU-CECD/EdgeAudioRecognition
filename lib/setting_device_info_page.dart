import 'package:flutter/material.dart';

import '../models/device_status.dart';

class SettingDeviceInfoPage extends StatelessWidget {
  final DeviceStatus? deviceStatus;

  const SettingDeviceInfoPage({
    super.key,
    required this.deviceStatus,
  });

  @override
  Widget build(BuildContext context) {
    final status = deviceStatus;

    return Scaffold(
      appBar: AppBar(title: const Text('기기 정보')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _infoTile('연결 상태', status?.connection ?? 'disconnected'),
          _infoTile(
            '기기 이름',
            status?.deviceName.isNotEmpty == true ? status!.deviceName : '-',
          ),
          _infoTile('배터리', status?.battery == null ? '-' : '${status!.battery}%'),
          _infoTile(
            '메시지',
            status?.message.isNotEmpty == true ? status!.message : '-',
          ),
        ],
      ),
    );
  }

  Widget _infoTile(String title, String value) {
    return ListTile(
      title: Text(title),
      subtitle: Text(value),
    );
  }
}
