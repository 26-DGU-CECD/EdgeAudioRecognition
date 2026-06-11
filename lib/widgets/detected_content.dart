import 'package:flutter/material.dart';

import '../models/sound_packet.dart';
import '../widgets/sound_image_view.dart';

class ListeningContent extends StatelessWidget {
  const ListeningContent({super.key});

  @override
  Widget build(BuildContext context) {
    return const Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.graphic_eq, size: 64, color: Colors.cyan),
        SizedBox(height: 12),
        Text(
          '소리를 듣고 있어요',
          style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 8),
        Text('30dB 이상, score 70% 이상일 때 표시'),
      ],
    );
  }
}

class DetectedContent extends StatelessWidget {
  final SoundPacket packet;
  final String imagePath;
  final Color color;

  const DetectedContent({
    super.key,
    required this.packet,
    required this.imagePath,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 230,
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.black12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SoundImageView(imagePath: imagePath),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                packet.displayLabel,
                maxLines: 1,
                softWrap: false,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          /*Text(
            'score: ${(packet.score * 100).toStringAsFixed(1)}%',
            style: const TextStyle(color: Colors.black87),
          ),*/
        ],
      ),
    );
  }
}
