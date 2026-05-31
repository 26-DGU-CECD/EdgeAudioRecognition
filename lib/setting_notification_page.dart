import 'package:flutter/material.dart';

class SettingNotificationPage extends StatefulWidget {
  final List<String> soundLabels;
  final Set<String> mutedLabels;
  final ValueChanged<String> onToggleMutedLabel;

  const SettingNotificationPage({
    super.key,
    required this.soundLabels,
    required this.mutedLabels,
    required this.onToggleMutedLabel,
  });

  @override
  State<SettingNotificationPage> createState() =>
      _SettingNotificationPageState();
}

class _SettingNotificationPageState extends State<SettingNotificationPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('알림 설정')),
      body: ListView.builder(
        padding: const EdgeInsets.symmetric(vertical: 8),
        itemCount: widget.soundLabels.length,
        itemBuilder: (context, index) {
          final label = widget.soundLabels[index];
          final enabled = !widget.mutedLabels.contains(label);

          return SwitchListTile(
            value: enabled,
            onChanged: (_) {
              widget.onToggleMutedLabel(label);
              setState(() {});
            },
            title: Text(label),
            subtitle: Text(enabled ? '알림 받음' : '알림 차단됨'),
          );
        },
      ),
    );
  }
}
