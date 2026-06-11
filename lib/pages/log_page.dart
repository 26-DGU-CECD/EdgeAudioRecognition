import 'package:flutter/material.dart';

import '../models/sound_packet.dart';
import '../widgets/sound_image_view.dart';

class LogPage extends StatelessWidget {
  final List<SoundPacket> logs;
  final String Function(String label) imagePathForLabel;

  const LogPage({
    super.key,
    required this.logs,
    required this.imagePathForLabel,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('로그')),
      body: logs.isEmpty
          ? const Center(child: Text('수신된 로그가 없습니다.'))
          : ListView(
              padding: const EdgeInsets.symmetric(vertical: 8),
              children: [
                _sectionHeader('오늘'),
                ...logs.map(_logTile),
                _sectionHeader('이번주'),
                _emptySectionText('이번주로 분류된 추가 로그가 없습니다.'),
                _sectionHeader('이번달'),
                _emptySectionText('이번달로 분류된 추가 로그가 없습니다.'),
              ],
            ),
    );
  }

  Widget _sectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 8),
      child: Text(
        '----$title----',
        textAlign: TextAlign.center,
        style: const TextStyle(
          fontWeight: FontWeight.bold,
          color: Colors.black54,
        ),
      ),
    );
  }

  Widget _emptySectionText(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: const TextStyle(color: Colors.black38, fontSize: 12),
      ),
    );
  }

  Widget _logTile(SoundPacket log) {
    return ListTile(
      leading: SoundImageView(imagePath: imagePathForLabel(log.displayLabel)),
      title: Text(log.displayLabel),
      subtitle: Text('${log.db}dB | ${log.level}'),
      trailing: Text('${log.time}'),
    );
  }
}
