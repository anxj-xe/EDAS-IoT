/*
 * EDAS-IoT: Environmental Anomaly Detection & Deployment Readiness System
 * Node Firmware for ESP32 Development Board
 *
 * Supported Sensors:
 *  - BME280 (I2C: SDA=GPIO21, SCL=GPIO22): Temperature, Humidity, Barometric Pressure
 *  - DHT22 (Single-bus: DATA=GPIO4): Temperature, Humidity (Fallback if BME280 unavailable)
 *
 * Communication:
 *  - Wi-Fi 802.11 b/g/n
 *  - HTTP POST with JSON payload conforming to EDAS-IoT API schema (/api/readings)
 *  - Exponential backoff retry logic on network or server drops
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

// Sensor selection: define ONE of the following
#define USE_BME280
// #define USE_DHT22

#ifdef USE_BME280
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
Adafruit_BME280 bme; // I2C address 0x76 or 0x77
#endif

#ifdef USE_DHT22
#include <DHT.h>
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);
#endif

// --- Network Configuration ---
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Server Configuration (Ensure ESP32 can route to this IP on local LAN)
const char* SERVER_URL    = "http://192.168.1.100:8000/api/readings";

// Device Identification
const char* DEVICE_ID     = "environment_node_01";

// Telemetry Timing
const unsigned long SAMPLING_INTERVAL_MS = 10000; // 10 seconds
const int MAX_RETRIES = 3;
const unsigned long INITIAL_BACKOFF_MS = 1000;

unsigned long lastSampleTime = 0;
unsigned long sequenceNumber = 0;

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi] Connection failed. Will retry on next sampling cycle.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n==========================================");
  Serial.println("  EDAS-IoT ESP32 Node Firmware Starting   ");
  Serial.println("==========================================");

  // Initialize selected sensor
#ifdef USE_BME280
  Serial.println("[Sensor] Initializing BME280 on I2C (SDA=21, SCL=22)...");
  if (!bme.begin(0x76) && !bme.begin(0x77)) {
    Serial.println("[Sensor Warning] Could not find a valid BME280 sensor! Check wiring.");
  } else {
    Serial.println("[Sensor] BME280 initialized successfully.");
  }
#endif

#ifdef USE_DHT22
  Serial.println("[Sensor] Initializing DHT22 on GPIO4...");
  dht.begin();
#endif

  connectWiFi();
}

bool transmitPayload(const String& jsonPayload) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[HTTP] Cannot transmit: No Wi-Fi connection.");
      return false;
    }
  }

  HTTPClient http;
  unsigned long backoff = INITIAL_BACKOFF_MS;

  for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    Serial.printf("[HTTP] Transmitting telemetry (Attempt %d/%d)...\n", attempt, MAX_RETRIES);
    int httpResponseCode = http.POST(jsonPayload);

    if (httpResponseCode == 200 || httpResponseCode == 201) {
      String response = http.getString();
      Serial.printf("[HTTP Success] Response code: %d | Response: %s\n", httpResponseCode, response.c_str());
      http.end();
      return true;
    } else {
      Serial.printf("[HTTP Error] Response code: %d | Error: %s\n", httpResponseCode, http.errorToString(httpResponseCode).c_str());
      http.end();

      if (attempt < MAX_RETRIES) {
        Serial.printf("[HTTP] Backing off for %lu ms before retry...\n", backoff);
        delay(backoff);
        backoff *= 2; // Exponential backoff
      }
    }
  }

  Serial.println("[HTTP Warning] Max retries reached. Telemetry packet dropped.");
  return false;
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - lastSampleTime >= SAMPLING_INTERVAL_MS || lastSampleTime == 0) {
    lastSampleTime = currentMillis;
    sequenceNumber++;

    float temperature = NAN;
    float humidity = NAN;
    float pressure = NAN;
    String sensorStatus = "OK";

#ifdef USE_BME280
    temperature = bme.readTemperature();
    humidity = bme.readHumidity();
    pressure = bme.readPressure() / 100.0F; // Convert Pa to hPa

    if (isnan(temperature) || isnan(humidity) || isnan(pressure)) {
      sensorStatus = "SENSOR_READ_FAILED";
      Serial.println("[Sensor Error] Failed to read from BME280!");
    }
#endif

#ifdef USE_DHT22
    temperature = dht.readTemperature();
    humidity = dht.readHumidity();

    if (isnan(temperature) || isnan(humidity)) {
      sensorStatus = "SENSOR_READ_FAILED";
      Serial.println("[Sensor Error] Failed to read from DHT22!");
    }
#endif

    // Generate ISO timestamp string (Mock or via NTP / RTC)
    // Here we use formatted uptime sequence or ISO placeholder
    char timestampBuffer[32];
    snprintf(timestampBuffer, sizeof(timestampBuffer), "2026-10-13T12:%02lu:%02luZ",
             (currentMillis / 60000) % 60, (currentMillis / 1000) % 60);

    // Build JSON payload adhering to the common EDAS-IoT schema
    String jsonPayload = "{";
    jsonPayload += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
    jsonPayload += "\"timestamp\":\"" + String(timestampBuffer) + "\",";
    jsonPayload += "\"sequence_number\":" + String(sequenceNumber) + ",";

    if (!isnan(temperature)) {
      jsonPayload += "\"temperature_c\":" + String(temperature, 2) + ",";
    } else {
      jsonPayload += "\"temperature_c\":null,";
    }

    if (!isnan(humidity)) {
      jsonPayload += "\"humidity_pct\":" + String(humidity, 2) + ",";
    } else {
      jsonPayload += "\"humidity_pct\":null,";
    }

    if (!isnan(pressure)) {
      jsonPayload += "\"pressure_hpa\":" + String(pressure, 2) + ",";
    } else {
      jsonPayload += "\"pressure_hpa\":null,";
    }

    jsonPayload += "\"sensor_status\":\"" + sensorStatus + "\"";
    jsonPayload += "}";

    Serial.println("\n[Telemetry] Prepared Payload:");
    Serial.println(jsonPayload);

    transmitPayload(jsonPayload);
  }

  delay(50); // Yield CPU
}
