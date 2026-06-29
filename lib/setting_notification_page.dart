import 'package:flutter/material.dart';

import 'ui/app_colors.dart';
import 'ui/app_widgets.dart';
import 'ui/sound_style.dart';

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
  // 라벨 → 위험도 분류
  String _levelOf(String label) {
    final l = label.replaceAll(' ', '');
    const danger = ['총', '응급', '도난', '화재', '경보', '유리', '비명', '아기'];
    const caution = ['공사', '자전거', '경적', '차량', '노크', '개', '고양이'];
    if (danger.any((d) => l.contains(d))) return 'danger';
    if (caution.any((c) => l.contains(c))) return 'caution';
    return 'info';
  }

  @override
  Widget build(BuildContext context) {
    final groups = <String, List<String>>{
      'danger': [],
      'caution': [],
      'info': [],
    };
    for (final label in widget.soundLabels) {
      groups[_levelOf(label)]!.add(label);
    }

    return Scaffold(
      appBar: AppBar(title: const Text('알림 받을 소리')),
      body: SafeArea(
        top: false,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // info banner
            Container(
              margin: const EdgeInsets.fromLTRB(18, 0, 18, 12),
              padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 13),
              decoration: BoxDecoration(
                color: AppColors.primarySoft,
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.info, size: 20, color: AppColors.primary),
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      '끈 소리는 홈 알림에 표시되지 않아요. 감지 기록에는 계속 저장됩니다.',
                      style: TextStyle(
                        fontSize: 13,
                        height: 1.5,
                        fontWeight: FontWeight.w500,
                        color: AppColors.primaryText,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(18, 0, 18, 24),
                children: [
                  for (final level in const ['danger', 'caution', 'info'])
                    if (groups[level]!.isNotEmpty)
                      _group(level, groups[level]!),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _group(String level, List<String> labels) {
    final style = LevelStyle.of(level);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(6, 8, 6, 8),
          child: Text(
            style.label,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w800,
              color: style.color,
            ),
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: AppColors.card,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: AppColors.cardBorder),
          ),
          clipBehavior: Clip.antiAlias,
          child: Column(
            children: [
              for (var i = 0; i < labels.length; i++) ...[
                if (i > 0)
                  const Divider(height: 1, thickness: 1, color: AppColors.line),
                _row(labels[i], level),
              ],
            ],
          ),
        ),
        const SizedBox(height: 8),
      ],
    );
  }

  Widget _row(String label, String level) {
    final enabled = !widget.mutedLabels.contains(label);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 11),
      child: Row(
        children: [
          SoundIcon(
            label: label,
            level: level,
            size: 40,
            radius: 12,
            iconSize: 21,
          ),
          const SizedBox(width: 13),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
          ),
          Switch(
            value: enabled,
            onChanged: (_) {
              widget.onToggleMutedLabel(label);
              setState(() {});
            },
            activeThumbColor: Colors.white,
            activeTrackColor: AppColors.primary,
          ),
        ],
      ),
    );
  }
}
