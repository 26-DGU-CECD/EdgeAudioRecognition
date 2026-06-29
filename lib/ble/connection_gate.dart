import 'package:flutter/material.dart';
import 'ble_sound_service.dart';
import 'ble_connection_page.dart';
import 'package:untitled/app_prefs.dart';
import 'package:untitled/main_page.dart';
import 'package:untitled/onboarding/onboarding_page.dart';

/// 앱 시작 시 저장된 ble 기기가 있으면 메인으로 보내고, 없으면 ble 연결 화면
class ConnectionGate extends StatefulWidget {
  const ConnectionGate({super.key});

  @override
  State<ConnectionGate> createState() => _ConnectionGateState();
}

class _ConnectionGateState extends State<ConnectionGate> {
  @override
  void initState() {
    super.initState();
    _routeBySavedConnection();
  }

  Future<void> _routeBySavedConnection() async {
    final onboardingSeen = await AppPrefs.isOnboardingSeen();

    if (!mounted) return;

    if (!onboardingSeen) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const OnboardingPage()),
      );
      return;
    }

    final connected = await BleSoundService.instance.connectSavedDevice();

    if(!mounted) return;

    if(connected){
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const MainPage(title: 'Demo Home Page',)),
      );
      return;
    }
    Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const BleConnectionPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 120,
              height: 120,
              padding: const EdgeInsets.all(14),
              decoration: const BoxDecoration(
                color: Color(0xFFEAF1FE),
                shape: BoxShape.circle,
              ),
              child: Image.asset('assets/miimo.png', fit: BoxFit.contain),
            ),
            const SizedBox(height: 28),
            const Text(
              'Sound Keyring',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w900,
                color: Color(0xFF16151A),
              ),
            ),
            const SizedBox(height: 20),
            const SizedBox(
              width: 26,
              height: 26,
              child: CircularProgressIndicator(strokeWidth: 2.5),
            ),
          ],
        ),
      ),
    );
  }
}