import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:untitled/main_page.dart';
import 'package:untitled/ui/app_colors.dart';

import 'ble_constants.dart';
import 'ble_sound_service.dart';
import 'connection_success_page.dart';

class BleConnectionPage extends StatefulWidget {
  const BleConnectionPage({super.key});

  @override
  State<BleConnectionPage> createState() => _BleConnectionPageState();
}

class _BleConnectionPageState extends State<BleConnectionPage>
    with SingleTickerProviderStateMixin {
  final List<ScanResult> scanResults = [];

  StreamSubscription<List<ScanResult>>? _scanSub;
  late final AnimationController _pulse;

  bool isScanning = false;
  bool isConnecting = false;
  String statusText = '키링을 검색하세요.';

  @override
  void initState() {
    super.initState();

    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2400),
    )..repeat();

    startScan();
  }

  @override
  void dispose() {
    _scanSub?.cancel();
    _pulse.dispose();
    FlutterBluePlus.stopScan();
    super.dispose();
  }

  Future<void> startScan() async {
    setState(() {
      scanResults.clear();
      isScanning = true;
      statusText = '키링을 검색하는 중…';
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
        statusText = '블루투스를 찾을 수 없습니다.';
      });
      return;
    }

    if (!mounted) return;
    setState(() {
      isScanning = false;
      statusText =
          scanResults.isEmpty ? '키링을 찾지 못했습니다.' : '키링을 선택해서 연결하세요.';
    });
  }

  Future<void> connect(ScanResult result) async {
    setState(() {
      isConnecting = true;
      statusText = '연결 중...';
    });

    try {
      await FlutterBluePlus.stopScan();

      final deviceName = _deviceName(result);

      await BleSoundService.instance.connect(
        result.device,
        deviceName: deviceName,
      );

      if (!mounted) return;

      // 연결 성공 피드백(프레임 L)을 거쳐 메인으로 진입.
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ConnectionSuccessPage(deviceName: deviceName),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        isConnecting = false;
        statusText = '블루투스를 찾을 수 없습니다.';
      });
    }
  }

  String _deviceName(ScanResult result) {
    final advName = result.advertisementData.advName;
    final platformName = result.device.platformName;

    if (advName.isNotEmpty) return advName;
    if (platformName.isNotEmpty) return platformName;
    return result.device.remoteId.str;
  }

  void _enterWithoutDevice() {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const MainPage(title: 'Sound Keychain')),
      (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('기기 연결'),
        actions: [
          IconButton(
            onPressed: isScanning ? null : startScan,
            icon: const Icon(Icons.refresh, color: AppColors.primary),
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(22, 6, 22, 22),
          child: Column(
            children: [
              const SizedBox(height: 14),
              _scanVisual(),
              const SizedBox(height: 14),
              Text(
                isScanning ? '키링을 검색하는 중…' : statusText,
                textAlign: TextAlign.center,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w900,
                  color: AppColors.textPrimary,
                ),
              ),
              const SizedBox(height: 5),
              const Text(
                '키링의 전원이 켜져 있는지 확인하세요',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  color: AppColors.textMuted,
                ),
              ),
              const SizedBox(height: 18),
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '검색된 기기',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                    color: AppColors.textMuted,
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Expanded(
                child: scanResults.isEmpty
                    ? _emptyDevices()
                    : ListView.separated(
                        itemCount: scanResults.length,
                        separatorBuilder: (_, _) =>
                            const SizedBox(height: 10),
                        itemBuilder: (_, i) => _deviceCard(scanResults[i]),
                      ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                height: 54,
                child: OutlinedButton(
                  onPressed: isConnecting ? null : _enterWithoutDevice,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppColors.textSecondary,
                    backgroundColor: AppColors.card,
                    side: const BorderSide(color: Color(0xFFD6DCE4)),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16)),
                    textStyle: const TextStyle(
                        fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                  child: const Text('기기 없이 둘러보기'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _scanVisual() {
    return SizedBox(
      width: 150,
      height: 150,
      child: AnimatedBuilder(
        animation: _pulse,
        builder: (_, child) {
          return Stack(
            alignment: Alignment.center,
            children: [
              if (isScanning) _ring(_pulse.value),
              if (isScanning) _ring((_pulse.value + 0.5) % 1.0),
              child!,
            ],
          );
        },
        child: Container(
          width: 88,
          height: 88,
          decoration: BoxDecoration(
            color: AppColors.primary,
            shape: BoxShape.circle,
            boxShadow: const [
              BoxShadow(
                color: Color(0x521B6EF3),
                blurRadius: 28,
                offset: Offset(0, 10),
              ),
            ],
          ),
          child: const Icon(Icons.bluetooth_searching,
              size: 44, color: Colors.white),
        ),
      ),
    );
  }

  Widget _ring(double t) {
    final scale = 0.55 + t * 1.1;
    final opacity = (0.5 * (1 - t)).clamp(0.0, 1.0);
    return Opacity(
      opacity: opacity,
      child: Transform.scale(
        scale: scale,
        child: Container(
          width: 150,
          height: 150,
          decoration: BoxDecoration(
            color: AppColors.primary.withValues(alpha: 0.14),
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }

  Widget _emptyDevices() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (isScanning)
            const SizedBox(
              width: 26,
              height: 26,
              child: CircularProgressIndicator(strokeWidth: 2.5),
            )
          else
            const Icon(Icons.bluetooth_disabled,
                size: 38, color: AppColors.textFaint),
          const SizedBox(height: 12),
          Text(
            isScanning ? '주변 기기를 찾고 있어요…' : '검색된 키링이 없습니다.',
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: AppColors.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  Widget _deviceCard(ScanResult result) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.primary, width: 2),
        boxShadow: const [
          BoxShadow(
            color: Color(0x1A1B6EF3),
            blurRadius: 24,
            offset: Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            padding: const EdgeInsets.all(2),
            decoration: BoxDecoration(
              color: AppColors.primarySoft,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Image.asset('assets/miimo.png', fit: BoxFit.contain),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _deviceName(result),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${result.device.remoteId.str} · RSSI ${result.rssi}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textMuted,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          FilledButton(
            onPressed: isConnecting ? null : () => connect(result),
            style: FilledButton.styleFrom(
              backgroundColor: AppColors.primary,
              foregroundColor: Colors.white,
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              minimumSize: const Size(0, 36),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(999)),
              textStyle:
                  const TextStyle(fontSize: 13, fontWeight: FontWeight.w800),
            ),
            child: isConnecting
                ? const SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white),
                  )
                : const Text('연결'),
          ),
        ],
      ),
    );
  }
}
