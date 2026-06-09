import 'package:flutter/material.dart';
import 'ble_sound_service.dart';
import 'ble_connection_page.dart';
import 'ble_connection_store.dart';
import 'package:untitled/main_page.dart';

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
    // TODO: implement build
    return const Scaffold(
      body: Center(child: CircularProgressIndicator(),),
    );
  }
}