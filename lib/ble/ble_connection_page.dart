import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:untitled/main_page.dart';
import 'package:untitled/services/app_permission_service.dart';

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

  /// 이미 로그를 찍은 기기. 같은 기기를 매 광고마다 다시 찍지 않기 위해 사용.
  final Set<String> _loggedDeviceIds = <String>{};

  StreamSubscription<List<ScanResult>>? _scanSub;
  StreamSubscription<String>? _logSub;

  bool isScanning = false;
  bool isConnecting = false;
  String statusText = '키링을 검색하세요.';

  /// Android 11 이하에서 스캔이 조용히 0건으로 끝나는 조건을 화면에 알리기 위한 문구.
  String? _scanWarning;

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
    // 이전 스캔/구독이 남아 있으면 결과가 누락되거나 겹치므로 먼저 정리한다.
    await _scanSub?.cancel();
    _scanSub = null;
    try {
      await FlutterBluePlus.stopScan();
    } catch (_) {}

    _loggedDeviceIds.clear();

    if (!mounted) return;
    setState(() {
      scanResults.clear();
      isScanning = true;
      statusText = 'Bluetooth 권한 확인 중...';
    });

    // 1) 권한. BLUETOOTH_SCAN 없이 startScan하면 Android 12+에서 결과가 0건이다.
    final permissionReport = await AppPermissionService.ensureBlePermissions();
    _debugLog(permissionReport.toString());

    if (!permissionReport.canScan) {
      if (!mounted) return;
      setState(() {
        isScanning = false;
        statusText =
            'Bluetooth 권한이 없습니다. 설정에서 권한을 허용해 주세요.\n$permissionReport';
      });
      return;
    }

    // Android 11 이하에서는 위치 권한과 위치 서비스(GPS)가 둘 다 있어야
    // BLE 스캔 결과가 나온다. 하나라도 없으면 예외 없이 0건으로 끝나므로
    // 로그뿐 아니라 화면에도 경고를 띄운다.
    final warnings = <String>[];

    if (!permissionReport.locationGranted) {
      warnings.add('위치 권한 거부됨');
    }
    if (!permissionReport.locationServiceEnabled) {
      warnings.add('위치 서비스(GPS) 꺼짐');
    }

    if (warnings.isNotEmpty) {
      _scanWarning =
          '⚠ ${warnings.join(' / ')} — Android 11 이하에서는 이 상태면 BLE 기기가 하나도 검색되지 않습니다.';
      _debugLog(_scanWarning!);
    } else {
      _scanWarning = null;
    }

    // 2) 어댑터 상태. 꺼져 있으면 스캔은 무조건 실패한다.
    if (await FlutterBluePlus.isSupported == false) {
      if (!mounted) return;
      setState(() {
        isScanning = false;
        statusText = '이 기기는 Bluetooth를 지원하지 않습니다.';
      });
      return;
    }

    BluetoothAdapterState adapterState = BluetoothAdapterState.unknown;
    try {
      adapterState = await FlutterBluePlus.adapterState
          .where((state) => state != BluetoothAdapterState.unknown)
          .first
          .timeout(const Duration(seconds: 5));
    } on TimeoutException {
      _debugLog('adapterState 확인 타임아웃');
    }
    _debugLog('adapterState=$adapterState');

    if (adapterState != BluetoothAdapterState.on) {
      if (!mounted) return;
      setState(() {
        isScanning = false;
        statusText = 'Bluetooth가 꺼져 있습니다. Bluetooth를 켜고 다시 시도해 주세요.';
      });
      return;
    }

    if (kBleScanDebugShowAll) {
      FlutterBluePlus.setLogLevel(LogLevel.verbose, color: false);
    }

    if (!mounted) return;
    setState(() {
      statusText = 'Bluetooth 기기 검색 중...';
    });

    // 3) 결과 구독은 startScan보다 반드시 먼저.
    _scanSub = FlutterBluePlus.scanResults.listen(
      _handleScanResults,
      onError: (Object error) {
        _debugLog('scanResults 오류: $error');
      },
    );

    // 4) 스캔 시작.
    //    withServices(안드로이드 하드웨어 ScanFilter)는 걸지 않는다.
    //    광고 패킷에 128비트 서비스 UUID가 실려 있지 않으면
    //    이 필터 때문에 OS 단계에서 결과가 전부 버려지기 때문이다.
    //    대신 _isKeyring으로 앱에서 걸러 miimo만 목록에 보여준다.
    try {
      _debugLog('startScan 호출 (앱 필터: ${jetsonServiceUuid.str})');

      await FlutterBluePlus.startScan(
        timeout: const Duration(seconds: 15),
        androidUsesFineLocation: false,
      );
    } catch (e) {
      _debugLog('startScan 실패: $e');
      if (!mounted) return;
      setState(() {
        isScanning = false;
        statusText = 'BLE 검색 실패: $e';
      });
      return;
    }

    _debugLog('스캔 종료. 발견된 기기 ${_loggedDeviceIds.length}개');

    if (!mounted) return;
    setState(() {
      isScanning = false;
      statusText = scanResults.isEmpty ? '키링을 찾지 못했습니다.' : '키링을 선택해서 연결하세요.';
    });
  }

  void _handleScanResults(List<ScanResult> results) {
    for (final result in results) {
      final id = result.device.remoteId.str;
      if (!_loggedDeviceIds.add(id)) continue;

      final adv = result.advertisementData;
      _debugLog(
        '기기 발견 '
        'remoteId=$id '
        'platformName="${result.device.platformName}" '
        'advName="${adv.advName}" '
        'rssi=${result.rssi} '
        'connectable=${adv.connectable} '
        'serviceUuids=${adv.serviceUuids.map((u) => u.str).toList()} '
        'serviceData=${adv.serviceData.keys.map((u) => u.str).toList()} '
        'manufacturerData=${adv.manufacturerData.keys.toList()} '
        'txPower=${adv.txPowerLevel}',
      );
    }

    final visible = kBleScanDebugShowAll
        ? results
        : results.where(_isKeyring).toList();

    if (!mounted) return;
    setState(() {
      scanResults
        ..clear()
        ..addAll(visible);
    });
  }

  bool _isKeyring(ScanResult result) {
    final advName = result.advertisementData.advName;
    final platformName = result.device.platformName;

    return advName == jetsonDeviceName ||
        platformName == jetsonDeviceName ||
        result.advertisementData.serviceUuids.contains(jetsonServiceUuid);
  }

  void _debugLog(String message) {
    debugPrint('[BLE-SCAN] $message');
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

  /// 디버그 모드에서는 실제 광고 이름을 그대로 보여준다.
  String _titleFor(ScanResult result) {
    if (!kBleScanDebugShowAll) {
      return keyringDisplayName;
    }

    final advName = result.advertisementData.advName;
    final platformName = result.device.platformName;

    if (advName.isNotEmpty) return advName;
    if (platformName.isNotEmpty) return platformName;
    return '(이름 없음)';
  }

  String _subtitleFor(ScanResult result) {
    final base = '${result.device.remoteId.str} / RSSI ${result.rssi}';

    if (!kBleScanDebugShowAll) {
      return base;
    }

    final uuids = result.advertisementData.serviceUuids
        .map((u) => u.str)
        .join(', ');

    return '$base\nadvName="${result.advertisementData.advName}" '
        'platformName="${result.device.platformName}"\n'
        'services=[${uuids.isEmpty ? '-' : uuids}]';
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
            if (_scanWarning != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  _scanWarning!,
                  style: const TextStyle(fontSize: 12, color: Colors.red),
                ),
              ),
            if (kBleScanDebugShowAll)
              const Padding(
                padding: EdgeInsets.only(top: 4),
                child: Text(
                  '디버그 모드: 필터 없이 모든 BLE 기기를 표시합니다.',
                  style: TextStyle(fontSize: 12, color: Colors.orange),
                ),
              ),
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
                          isThreeLine: kBleScanDebugShowAll,
                          leading: const Icon(Icons.bluetooth),
                          title: Text(_titleFor(result)),
                          subtitle: Text(
                            _subtitleFor(result),
                            style: const TextStyle(fontSize: 12),
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
