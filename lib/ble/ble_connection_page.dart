import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:untitled/main_page.dart';

import '../pages/background_alert_consent_page.dart';
import 'ble_constants.dart';
import 'ble_sound_service.dart';

class BleConnectionPage extends StatefulWidget {
  const BleConnectionPage({
    super.key,
    this.showBackgroundAlertConsentOnConnect = false,
  });

  final bool showBackgroundAlertConsentOnConnect;

  @override
  State<BleConnectionPage> createState() => _BleConnectionPageState();
}

class _BleConnectionPageState extends State<BleConnectionPage> {
  final List<ScanResult> scanResults = [];

  StreamSubscription<List<ScanResult>>? _scanSub;
  StreamSubscription<String>? _logSub;

  bool isScanning = false;
  bool isConnecting = false;
  String statusText = '키링을 검색하세요.';

  @override
  void initState() {
    super.initState();

    _logSub = BleSoundService.instance.logs.listen((message) {
      if (!mounted) return;
      setState(() {
        statusText = message;
      });
    });

    startScan();
  }

  @override
  void dispose() {
    _scanSub?.cancel();
    _logSub?.cancel();
    FlutterBluePlus.stopScan();
    super.dispose();
  }

  Future<void> startScan() async {
    setState(() {
      scanResults.clear();
      isScanning = true;
      statusText = 'Bluetooth 기기 검색 중...';
    });

    _scanSub?.cancel();
    _scanSub = FlutterBluePlus.scanResults.listen((results) {
      final filtered = results.where((result) {
        final advName = result.advertisementData.advName;
        final platformName = result.device.platformName;

        return advName == jetsonDeviceName ||
            platformName == jetsonDeviceName ||
            result.advertisementData.serviceUuids.contains(jetsonServiceUuid);
      }).toList();

      if (!mounted) return;
      setState(() {
        scanResults
          ..clear()
          ..addAll(filtered);
      });
    });

    try {
      await FlutterBluePlus.startScan(
        timeout: const Duration(seconds: 8),
        withServices: [jetsonServiceUuid],
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        isScanning = false;
        statusText = 'BLE 검색 실패: $e';
      });
      return;
    }

    if (!mounted) return;
    setState(() {
      isScanning = false;
      statusText = scanResults.isEmpty ? '키링을 찾지 못했습니다.' : '키링을 선택해서 연결하세요.';
    });
  }

  Future<void> connect(ScanResult result) async {
    setState(() {
      isConnecting = true;
      statusText = '연결 중...';
    });

    try {
      await FlutterBluePlus.stopScan();

      await BleSoundService.instance.connect(
        result.device,
        deviceName: keyringDisplayName,
      );

      if (!mounted) return;

      await _openNextPage();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        isConnecting = false;
        statusText = '연결 실패: $e';
      });
    }
  }

  Future<void> _openNextPage() async {
    final shouldShowBackgroundAlertConsent =
        widget.showBackgroundAlertConsentOnConnect ||
        await BackgroundAlertConsentPage.shouldShow();

    if (!mounted) return;

    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(
        builder: (_) => shouldShowBackgroundAlertConsent
            ? const BackgroundAlertConsentPage()
            : const MainPage(title: 'Sound Keychain'),
      ),
      (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Bluetooth 키링 연결'),
        actions: [
          IconButton(
            onPressed: isScanning ? null : startScan,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Text(statusText),
            const SizedBox(height: 16),
            if (isConnecting || isScanning) const LinearProgressIndicator(),
            const SizedBox(height: 16),
            Expanded(
              child: scanResults.isEmpty
                  ? const Center(child: Text('검색된 키링이 없습니다.'))
                  : ListView.builder(
                      itemCount: scanResults.length,
                      itemBuilder: (context, index) {
                        final result = scanResults[index];

                        return ListTile(
                          leading: const Icon(Icons.bluetooth),
                          title: const Text(keyringDisplayName),
                          subtitle: Text(
                            '${result.device.remoteId.str} / RSSI ${result.rssi}',
                          ),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: isConnecting ? null : () => connect(result),
                        );
                      },
                    ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: isConnecting
                    ? null
                    : () {
                        Navigator.pushAndRemoveUntil(
                          context,
                          MaterialPageRoute(
                            builder: (_) =>
                                const MainPage(title: 'Sound Keychain'),
                          ),
                          (_) => false,
                        );
                      },
                child: const Text('기기 연결 없이 앱 시작하기 ->'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
