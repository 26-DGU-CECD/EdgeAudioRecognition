import 'package:flutter/material.dart';

/// 디자인 시스템 색상 토큰 (소리감지앱 디자인 기준)
class AppColors {
  AppColors._();

  // surface
  static const Color scaffold = Color(0xFFF4F6F9);
  static const Color card = Colors.white;
  static const Color cardBorder = Color(0xFFEEF1F5);
  static const Color line = Color(0xFFF2F4F7);
  static const Color divider = Color(0xFFE6E9EE);

  // text
  static const Color textPrimary = Color(0xFF16151A);
  static const Color textSecondary = Color(0xFF5A6472);
  static const Color textMuted = Color(0xFF9AA3B0);
  static const Color textFaint = Color(0xFFAEB6C2);

  // brand (info / primary)
  static const Color primary = Color(0xFF1B6EF3);
  static const Color primarySoft = Color(0xFFEAF1FE);
  static const Color primaryText = Color(0xFF2B5BA8);

  // danger (긴급)
  static const Color danger = Color(0xFFD92D20);
  static const Color dangerSoft = Color(0xFFFEECEB);
  static const Color dangerBorder = Color(0xFFF3C6C2);

  // caution (주의)
  static const Color caution = Color(0xFFD97706);
  static const Color cautionText = Color(0xFFB45309);
  static const Color cautionSoft = Color(0xFFFFF7E8);

  // success (연결됨)
  static const Color success = Color(0xFF16A34A);
  static const Color successText = Color(0xFF15803D);
  static const Color successSoft = Color(0xFFE7F6ED);
}
