package com.example.bluetoothtest

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothGattService
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.bluetoothtest.ui.theme.BluetoothTestTheme
import java.nio.charset.StandardCharsets
import java.util.UUID

private val JetsonServiceUuid: UUID = UUID.fromString("12345678-1234-5678-1234-56789abcdef0")
private val JetsonCharacteristicUuid: UUID =
    UUID.fromString("12345678-1234-5678-1234-56789abcdef1")
private val ClientCharacteristicConfigUuid: UUID =
    UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

class MainActivity : ComponentActivity() {
    private lateinit var bleClient: JetsonBleClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bleClient = JetsonBleClient(this)
        enableEdgeToEdge()
        setContent {
            BluetoothTestTheme {
                BleApp(bleClient)
            }
        }
    }

    override fun onDestroy() {
        bleClient.close()
        super.onDestroy()
    }
}

data class BleDeviceItem(
    val name: String,
    val address: String,
    val rssi: Int,
)

class JetsonBleClient(private val context: Context) {
    var status by mutableStateOf("Idle")
        private set
    var latestMessage by mutableStateOf("")
        private set

    val devices = mutableStateListOf<BleDeviceItem>()
    val messages = mutableStateListOf<String>()

    private val bluetoothManager =
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val bluetoothAdapter: BluetoothAdapter?
        get() = bluetoothManager.adapter
    private val scanner
        get() = bluetoothAdapter?.bluetoothLeScanner

    private var bluetoothGatt: BluetoothGatt? = null
    private var isScanning = false
    private val mainHandler = Handler(Looper.getMainLooper())

    private val scanCallback = object : ScanCallback() {
        @SuppressLint("MissingPermission")
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device
            val name = result.scanRecord?.deviceName ?: device.name ?: "Unknown"
            val item = BleDeviceItem(name = name, address = device.address, rssi = result.rssi)
            onMain {
                val index = devices.indexOfFirst { it.address == item.address }
                if (index >= 0) {
                    devices[index] = item
                } else {
                    devices.add(item)
                }
            }
        }

        override fun onScanFailed(errorCode: Int) {
            onMain {
                isScanning = false
                status = "Scan failed: $errorCode"
            }
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, statusCode: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    onMain { status = "Connected, discovering services" }
                    gatt.discoverServices()
                }

                BluetoothProfile.STATE_DISCONNECTED -> {
                    onMain { status = "Disconnected" }
                    bluetoothGatt?.close()
                    bluetoothGatt = null
                }
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, statusCode: Int) {
            if (statusCode != BluetoothGatt.GATT_SUCCESS) {
                onMain { status = "Service discovery failed: $statusCode" }
                return
            }
            val characteristic = findJetsonCharacteristic(gatt)
            if (characteristic == null) {
                onMain { status = "Jetson characteristic not found" }
                return
            }
            onMain { status = "Subscribed to Jetson messages" }
            subscribeToNotifications(gatt, characteristic)
            gatt.readCharacteristic(characteristic)
        }

        override fun onCharacteristicRead(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray,
            statusCode: Int,
        ) {
            if (statusCode == BluetoothGatt.GATT_SUCCESS) {
                acceptMessage(value)
            }
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray,
        ) {
            acceptMessage(value)
        }
    }

    fun hasPermissions(): Boolean =
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BLUETOOTH_SCAN,
        ) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.BLUETOOTH_CONNECT,
            ) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    fun startScan() {
        if (!hasPermissions()) {
            status = "Bluetooth permissions required"
            return
        }
        val activeScanner = scanner
        if (activeScanner == null) {
            status = "Bluetooth LE scanner unavailable"
            return
        }
        stopScan()
        devices.clear()
        val filters = listOf(
            ScanFilter.Builder()
                .setServiceUuid(ParcelUuid(JetsonServiceUuid))
                .build(),
        )
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        activeScanner.startScan(filters, settings, scanCallback)
        isScanning = true
        status = "Scanning for JHello"
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        if (!isScanning) return
        scanner?.stopScan(scanCallback)
        isScanning = false
        status = "Scan stopped"
    }

    @SuppressLint("MissingPermission")
    fun connect(address: String) {
        if (!hasPermissions()) {
            status = "Bluetooth permissions required"
            return
        }
        stopScan()
        val device = bluetoothAdapter?.getRemoteDevice(address)
        if (device == null) {
            status = "Device unavailable"
            return
        }
        status = "Connecting to ${safeName(device)}"
        bluetoothGatt?.close()
        bluetoothGatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
    }

    @SuppressLint("MissingPermission")
    fun disconnect() {
        bluetoothGatt?.disconnect()
        bluetoothGatt?.close()
        bluetoothGatt = null
        status = "Disconnected"
    }

    fun close() {
        stopScan()
        disconnect()
    }

    private fun findJetsonCharacteristic(gatt: BluetoothGatt): BluetoothGattCharacteristic? {
        val service: BluetoothGattService = gatt.getService(JetsonServiceUuid) ?: return null
        return service.getCharacteristic(JetsonCharacteristicUuid)
    }

    @SuppressLint("MissingPermission")
    private fun subscribeToNotifications(
        gatt: BluetoothGatt,
        characteristic: BluetoothGattCharacteristic,
    ) {
        gatt.setCharacteristicNotification(characteristic, true)
        val descriptor = characteristic.getDescriptor(ClientCharacteristicConfigUuid) ?: return
        gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
    }

    private fun acceptMessage(value: ByteArray) {
        val decoded = value.toString(StandardCharsets.UTF_8)
        onMain {
            latestMessage = decoded
            messages.add(0, decoded)
        }
    }

    @SuppressLint("MissingPermission")
    private fun safeName(device: BluetoothDevice): String = device.name ?: device.address

    private fun onMain(action: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            action()
        } else {
            mainHandler.post(action)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BleApp(bleClient: JetsonBleClient) {
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        val granted = result.values.all { it }
        if (granted) {
            bleClient.startScan()
        }
    }

    LaunchedEffect(Unit) {
        if (!bleClient.hasPermissions()) {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.BLUETOOTH_SCAN,
                    Manifest.permission.BLUETOOTH_CONNECT,
                ),
            )
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Jetson BLE Client") })
        },
    ) { innerPadding ->
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Status", style = MaterialTheme.typography.labelLarge)
                Text(bleClient.status, style = MaterialTheme.typography.bodyLarge)

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            if (bleClient.hasPermissions()) {
                                bleClient.startScan()
                            } else {
                                permissionLauncher.launch(
                                    arrayOf(
                                        Manifest.permission.BLUETOOTH_SCAN,
                                        Manifest.permission.BLUETOOTH_CONNECT,
                                    ),
                                )
                            }
                        },
                    ) {
                        Text("Scan")
                    }
                    Button(onClick = { bleClient.stopScan() }) {
                        Text("Stop")
                    }
                    Button(onClick = { bleClient.disconnect() }) {
                        Text("Disconnect")
                    }
                }

                Text("Nearby Jetson devices", style = MaterialTheme.typography.titleMedium)
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp),
                ) {
                    items(bleClient.devices, key = { it.address }) { item ->
                        ListItem(
                            headlineContent = { Text(item.name) },
                            supportingContent = { Text("${item.address}  RSSI ${item.rssi}") },
                            trailingContent = {
                                Button(onClick = { bleClient.connect(item.address) }) {
                                    Text("Connect")
                                }
                            },
                        )
                        HorizontalDivider()
                    }
                }

                Text("Latest message", style = MaterialTheme.typography.titleMedium)
                Text(
                    text = bleClient.latestMessage.ifBlank { "No message received" },
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.SemiBold,
                )

                Text("History", style = MaterialTheme.typography.titleMedium)
                LazyColumn(modifier = Modifier.fillMaxWidth()) {
                    items(bleClient.messages) { message ->
                        Text(message, modifier = Modifier.padding(vertical = 6.dp))
                        HorizontalDivider()
                    }
                }

                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Run Jetson as: sudo python3 jetson_ble_hello.py --name JHello --message \"hello from Jetson\" --repeat 2",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun BleAppPreview() {
    BluetoothTestTheme {
        Text("Jetson BLE Client")
    }
}
