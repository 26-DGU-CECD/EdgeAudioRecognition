import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:untitled/ble/connection_gate.dart';
import 'services/notification_service.dart';
import 'ui/app_colors.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      statusBarBrightness: Brightness.light,
    ),
  );
  await NotificationService.instance.init();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.light,
    ).copyWith(
      primary: AppColors.primary,
      surface: AppColors.scaffold,
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Sound Keyring',
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.light,
        scaffoldBackgroundColor: AppColors.scaffold,
        colorScheme: colorScheme,
        splashColor: AppColors.primarySoft,
        appBarTheme: const AppBarTheme(
          backgroundColor: AppColors.scaffold,
          surfaceTintColor: Colors.transparent,
          foregroundColor: AppColors.textPrimary,
          elevation: 0,
          centerTitle: false,
          titleTextStyle: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: AppColors.textPrimary,
          ),
        ),
      ),
      home: const ConnectionGate(),
    );
  }
}











