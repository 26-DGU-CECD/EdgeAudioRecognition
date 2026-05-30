## 2026-05-30 iOS BLE Client Progress

Added a native SwiftUI/CoreBluetooth iPhone client under `ios_ble_client`.

Files:

- `ios_ble_client/JetsonBleClient.xcodeproj`
- `ios_ble_client/JetsonBleClient/Info.plist`
- `ios_ble_client/JetsonBleClient/JetsonBleClientApp.swift`
- `ios_ble_client/JetsonBleClient/ContentView.swift`
- `ios_ble_client/JetsonBleClient/BleCentral.swift`
- `ios_ble_client/README.md`

Behavior:

- Requests iOS Bluetooth permission with `NSBluetoothAlwaysUsageDescription`.
- Scans BLE peripherals from a physical iPhone.
- Prioritizes devices named `JHello` or advertising service UUID `12345678-1234-5678-1234-56789abcdef0`.
- Connects to the Jetson BLE peripheral.
- Discovers characteristic UUID `12345678-1234-5678-1234-56789abcdef1`.
- Reads the current value and subscribes to notify.
- Displays latest message and message history.

Important:

- iOS Simulator cannot scan/connect to external Jetson BLE devices.
- The app must be opened/built on macOS with Xcode and run on a physical iPhone.
- Select an Apple developer team in Xcode `Signing & Capabilities` before running on the iPhone.

# Session Handoff

## Project Goal

Jetson will act as a BLE peripheral/server. A Flutter Android app will act as a BLE central/client.

Required flow:

1. Flutter scans nearby BLE devices.
2. User selects/connects to the Jetson device.
3. Jetson periodically sends messages to Flutter via BLE notify.
4. Flutter displays received messages on screen.

Required Flutter screens:

1. Bluetooth connection screen
   - Request Bluetooth permissions.
   - Scan BLE devices.
   - Show Jetson candidates.
   - Connect to selected Jetson.

2. Message screen
   - Show connection status.
   - Show latest received message.
   - Show received message history.
   - Provide disconnect action.

## Current Repo State

Existing Jetson BLE example:

- `jetson/jetson_ble_hello.py`
- `jetson/jetson_ble_hello_README.md`

Existing BLE UUIDs:

```text
Device name: JHello
Service UUID: 12345678-1234-5678-1234-56789abcdef0
Characteristic UUID: 12345678-1234-5678-1234-56789abcdef1
Characteristic flags: read, notify
Default message: 안녕
```

Current Jetson script already supports periodic notify:

```bash
sudo python3 jetson_ble_hello.py --name JHello --message "안녕 from Jetson" --repeat 2
```

## Flutter / Android Setup Decisions

Flutter installed:

```text
Flutter stable 3.44.0
Dart 3.12.0 bundled with Flutter
```

Android setup status before restart:

```text
Android SDK detected: 36.1.0
Remaining issues:
- cmdline-tools component missing
- Android license status unknown
```

Next setup steps after restart:

1. Open Android Studio.
2. Go to `More Actions` -> `SDK Manager` -> `SDK Tools`.
3. Install/check:
   - Android SDK Command-line Tools (latest)
   - Android SDK Platform-Tools
   - Android SDK Build-Tools
   - Android Emulator
4. Reopen PowerShell.
5. Run:

```powershell
flutter doctor --android-licenses
flutter doctor
```

Expected result:

```text
[√] Android toolchain - develop for Android devices
```

## Emulator Choice

Do not use API 24 for the emulator.

Recommended:

```text
AVD system image: API 35 or API 36
Minimum for BLE emulator work: API 31
Android Emulator version: 36.5+
```

Build choices:

```text
Android language: Kotlin
Gradle DSL: Kotlin DSL
```

## Implementation Plan

1. Finish Android SDK setup and licenses.
2. Create Flutter project inside this repo, recommended folder:

```powershell
flutter create mobile
```

3. Add BLE dependencies:

```powershell
cd mobile
flutter pub add flutter_blue_plus
flutter pub add permission_handler
```

4. Add Android permissions in `mobile/android/app/src/main/AndroidManifest.xml`.
5. Implement BLE abstraction with two modes:
   - Mock source for emulator/UI testing.
   - Real BLE source using `flutter_blue_plus`.
6. Implement `BluetoothConnectScreen`.
7. Implement `MessageScreen`.
8. Later, refine Jetson server if needed.

## Important Testing Note

The user wants to test with the Android Emulator. The app should still include a mock message source because actual Jetson BLE discovery from Android Emulator can be environment-dependent. Real end-to-end BLE with Jetson is most reliable on a physical Android phone, but emulator UI/message flow can be developed first.
## 2026-05-30 BLE Client Progress

Android BLE client is now implemented in the existing native Kotlin/Compose app under `bluetooth_flutter`.

Changed files:

- `bluetooth_flutter/app/src/main/AndroidManifest.xml`
  - Declares BLE hardware.
  - Adds `BLUETOOTH_SCAN`, `BLUETOOTH_CONNECT`, and legacy `ACCESS_FINE_LOCATION`.
- `bluetooth_flutter/app/src/main/java/com/example/bluetoothtest/MainActivity.kt`
  - Requests runtime BLE permissions.
  - Scans for service UUID `12345678-1234-5678-1234-56789abcdef0`.
  - Lists discovered devices.
  - Connects to the selected device.
  - Discovers characteristic UUID `12345678-1234-5678-1234-56789abcdef1`.
  - Reads current value and subscribes to notify.
  - Displays latest message and message history.

Build verification:

```powershell
cd bluetooth_flutter
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:Path="$env:JAVA_HOME\bin;$env:Path"
.\gradlew.bat assembleDebug
```

Result: debug build succeeded.

Next recommended test:

```bash
sudo python3 jetson_ble_hello.py --name JHello --message "hello from Jetson" --repeat 2
```

Then install/run the Android app on a physical Android phone for real BLE testing. The Android Emulator remains useful for UI/build checks, but Jetson BLE discovery from the emulator may not work reliably.
