/// 소리 이미지 표시
import 'package:flutter/material.dart';

class SoundImageView extends StatelessWidget {
  final String imagePath;

  const SoundImageView({
    super.key,
    required this.imagePath,
  });

  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    return Image.asset(
      imagePath,
      width: 64,
      height: 64,
      fit: BoxFit.contain,

      errorBuilder: (context, error, stackTrace) {
        return Container(
          width: 64,
          height: 64,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: Colors.white10,
            border: Border.all(color: Colors.white24),
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Icon(
            Icons.image_not_supported_outlined,
            color: Colors.white54,
          ),
        );
      },
    );
  }
}