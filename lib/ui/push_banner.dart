import 'dart:async';

import 'package:flutter/material.dart';

import '../sound_packet.dart';
import 'app_colors.dart';
import 'sound_style.dart';

/// 푸시 알림 디자인(P1/P3 히어로 카드)을 그대로 옮긴 인앱 카드.
/// 흰 카드 + 레벨 색 좌측 보더 + 헤더(소리감지·지금·레벨뱃지) + 소리 이미지 + 방향.
class PushNotificationCard extends StatelessWidget {
  final SoundPacket packet;

  const PushNotificationCard({super.key, required this.packet});

  @override
  Widget build(BuildContext context) {
    final style = LevelStyle.of(packet.level);
    final assetPath = assetForSoundLabel(packet.displayLabel);

    return Container(
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(26),
        boxShadow: const [
          BoxShadow(
            color: Color(0x42141628), // rgba(20,22,40,.26)
            blurRadius: 38,
            offset: Offset(0, 16),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(26),
        child: IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 좌측 레벨 컬러 보더 (5px)
              Container(width: 5, color: style.color),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(15, 15, 16, 15),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _header(style),
                      const SizedBox(height: 11),
                      _body(style, assetPath),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header(LevelStyle style) {
    return Row(
      children: [
        Icon(Icons.hearing, size: 17, color: style.color),
        const SizedBox(width: 7),
        const Text(
          '소리감지',
          style: TextStyle(
            fontWeight: FontWeight.w700,
            fontSize: 12,
            color: AppColors.textSecondary,
          ),
        ),
        const SizedBox(width: 5),
        const Text(
          '· 지금',
          style: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 12,
            color: AppColors.textFaint,
          ),
        ),
        const Spacer(),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
          decoration: BoxDecoration(
            color: style.soft,
            borderRadius: BorderRadius.circular(999),
          ),
          child: Text(
            style.label,
            style: TextStyle(
              fontWeight: FontWeight.w800,
              fontSize: 11,
              color: style.color,
            ),
          ),
        ),
      ],
    );
  }

  Widget _body(LevelStyle style, String? assetPath) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        _soundImage(style, assetPath),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                packet.displayLabel,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontWeight: FontWeight.w900,
                  fontSize: 21,
                  height: 1.05,
                  letterSpacing: -0.2,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: 5),
              Row(
                children: [
                  Icon(Icons.navigation, size: 18, color: style.color),
                  const SizedBox(width: 6),
                  Flexible(
                    child: Text(
                      '${packet.directionLabel} 방향에서 감지',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 15,
                        color: style.color,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _soundImage(LevelStyle style, String? assetPath) {
    if (assetPath != null) {
      return Image.asset(
        assetPath,
        width: 56,
        height: 56,
        fit: BoxFit.contain,
        errorBuilder: (_, _, _) => _iconFallback(style),
      );
    }
    return _iconFallback(style);
  }

  Widget _iconFallback(LevelStyle style) {
    return Container(
      width: 56,
      height: 56,
      decoration: BoxDecoration(color: style.soft, shape: BoxShape.circle),
      child: Icon(iconForSoundLabel(packet.displayLabel),
          size: 28, color: style.color),
    );
  }
}

/// 앱이 켜져 있을 때 화면 상단에서 내려오는 헤즈업 배너(P3).
/// 자동으로 사라지고, 탭하거나 위로 스와이프하면 즉시 닫힌다.
class HeadsUpBanner extends StatefulWidget {
  final SoundPacket packet;
  final VoidCallback onDismiss;
  final VoidCallback? onTap;
  final Duration visibleFor;

  const HeadsUpBanner({
    super.key,
    required this.packet,
    required this.onDismiss,
    this.onTap,
    this.visibleFor = const Duration(seconds: 5),
  });

  @override
  State<HeadsUpBanner> createState() => _HeadsUpBannerState();
}

class _HeadsUpBannerState extends State<HeadsUpBanner>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<Offset> _offset;
  late final Animation<double> _fade;
  Timer? _autoHide;
  bool _closing = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 320),
    );
    _offset = Tween<Offset>(
      begin: const Offset(0, -1.15),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    ));
    _fade = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
    _controller.forward();
    _autoHide = Timer(widget.visibleFor, _dismiss);
  }

  Future<void> _dismiss() async {
    if (_closing) return;
    _closing = true;
    _autoHide?.cancel();
    if (mounted) {
      await _controller.reverse();
    }
    widget.onDismiss();
  }

  @override
  void dispose() {
    _autoHide?.cancel();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      bottom: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
        child: SlideTransition(
          position: _offset,
          child: FadeTransition(
            opacity: _fade,
            child: Dismissible(
              key: ValueKey(widget.packet.hashCode),
              direction: DismissDirection.up,
              onDismissed: (_) {
                _autoHide?.cancel();
                widget.onDismiss();
              },
              child: GestureDetector(
                onTap: () {
                  widget.onTap?.call();
                  _dismiss();
                },
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    PushNotificationCard(packet: widget.packet),
                    const SizedBox(height: 6),
                    Container(
                      width: 40,
                      height: 5,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.85),
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
