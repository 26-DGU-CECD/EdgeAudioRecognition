import 'package:flutter/material.dart';

import '../models/sound_packet.dart';
import '../ui/app_colors.dart';
import '../ui/app_widgets.dart';

class LogPage extends StatefulWidget {
  final List<SoundPacket> logs;

  const LogPage({super.key, required this.logs});

  @override
  State<LogPage> createState() => _LogPageState();
}

class _LogPageState extends State<LogPage> {
  // null = 전체, otherwise level key
  String? _filter;

  static const _chips = <_Chip>[
    _Chip(label: '전체', level: null),
    _Chip(label: '긴급', level: 'danger'),
    _Chip(label: '주의', level: 'caution'),
    _Chip(label: '정보', level: 'info'),
  ];

  @override
  Widget build(BuildContext context) {
    final filtered = _filter == null
        ? widget.logs
        : widget.logs.where((l) => l.level == _filter).toList();

    return SafeArea(
      bottom: false,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 6, 20, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    const Text(
                      '감지 기록',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w900,
                        color: AppColors.textPrimary,
                      ),
                    ),
                    const Spacer(),
                    Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Text(
                        '총 ${widget.logs.length}건',
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textMuted,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    for (final chip in _chips) ...[
                      _filterChip(chip),
                      const SizedBox(width: 8),
                    ],
                  ],
                ),
              ],
            ),
          ),
          Expanded(
            child: filtered.isEmpty
                ? _emptyState()
                : ListView(
                    padding: const EdgeInsets.fromLTRB(18, 4, 18, 18),
                    children: [
                      SectionDivider(
                        title: '오늘',
                        count: '${filtered.length}건',
                      ),
                      for (final log in filtered) ...[
                        DetectionTile(packet: log),
                        const SizedBox(height: 9),
                      ],
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  Widget _filterChip(_Chip chip) {
    final selected = _filter == chip.level;
    return GestureDetector(
      onTap: () => setState(() => _filter = chip.level),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? AppColors.primary : AppColors.card,
          borderRadius: BorderRadius.circular(999),
          border: selected
              ? null
              : Border.all(color: AppColors.divider),
        ),
        child: Text(
          chip.label,
          style: TextStyle(
            fontSize: 13,
            fontWeight: selected ? FontWeight.w800 : FontWeight.w700,
            color: selected ? Colors.white : AppColors.textSecondary,
          ),
        ),
      ),
    );
  }

  Widget _emptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: const BoxDecoration(
              color: AppColors.primarySoft,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.history,
                size: 34, color: AppColors.primary),
          ),
          const SizedBox(height: 16),
          Text(
            _filter == null ? '아직 감지된 소리가 없어요' : '해당 조건의 기록이 없어요',
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: AppColors.textSecondary,
            ),
          ),
        ],
      ),
    );
  }
}

class _Chip {
  final String label;
  final String? level;
  const _Chip({required this.label, required this.level});
}
