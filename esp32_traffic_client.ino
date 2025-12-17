

#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

#if (ESP_ARDUINO_VERSION) > ESP_ARDUINO_VERSION_VAL(3,0,0)
#error "Please use ESP32 core version lower than V 3.0.0, 2.0.17 is recommended"
#endif

// ================= POWER (AXP2101) =================
#define XPOWERS_CHIP_AXP2101
#include "XPowersLib.h"
#include "utilities.h"
XPowersPMU PMU;

// ================= WiFi CONFIG =================
const char* WIFI_SSID = "YOUR_WIFI_SSID";      // <-- Change this
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";  // <-- Change this

// ================= SERVER CONFIG =================
// For local testing, use your PC's IP address
// For cloud, use your Azure/AWS URL
const char* SERVER_URL = "http://192.168.1.100:5000/analyze";  // <-- Change this

// ================= LED PINS =================
#define RED_LED    15
#define GREEN_LED  16

// ================= TIMING =================
#define CAPTURE_INTERVAL_MS 3000   // Send image every 3 seconds
#define WIFI_TIMEOUT_MS     10000  // WiFi connection timeout
#define HTTP_TIMEOUT_MS     15000  // HTTP request timeout

// ================= GLOBALS =================
unsigned long lastCapture = 0;
int consecutiveFailures = 0;
#define MAX_FAILURES 5  // After 5 failures, default to GREEN

// ================= SETUP =================
void setup() {
    Serial.begin(115200);
    
    // Wait for Serial
    while (!Serial);
    delay(3000);
    
    Serial.println("\n========================================");
    Serial.println("  TRAFFIC DETECTION - ESP32 CLIENT");
    Serial.println("========================================\n");

    // Initialize LEDs
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    
    // Start with both off
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, LOW);

    // Initialize PMU
    if (!initPMU()) {
        Serial.println("❌ PMU initialization failed!");
        blinkError();
    }

    // Initialize Camera
    if (!initCamera()) {
        Serial.println("❌ Camera initialization failed!");
        blinkError();
    }

    // Connect to WiFi
    if (!connectWiFi()) {
        Serial.println("❌ WiFi connection failed!");
        blinkError();
    }

    Serial.println("\n✅ System ready! Starting traffic detection...\n");
    
    // Flash green to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(GREEN_LED, HIGH);
        delay(200);
        digitalWrite(GREEN_LED, LOW);
        delay(200);
    }
}

// ================= MAIN LOOP =================
void loop() {
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ WiFi disconnected, reconnecting...");
        connectWiFi();
    }

    // Capture and analyze at intervals
    if (millis() - lastCapture >= CAPTURE_INTERVAL_MS) {
        lastCapture = millis();
        
        // Capture image
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("❌ Camera capture failed");
            handleFailure();
            return;
        }

        Serial.printf("📷 Captured: %d bytes\n", fb->len);

        // Send to server and get decision
        String decision = sendImageToServer(fb->buf, fb->len);
        
        // Return frame buffer
        esp_camera_fb_return(fb);

        // Control LEDs based on decision
        if (decision == "RED") {
            Serial.println("🔴 Decision: RED - High traffic");
            digitalWrite(RED_LED, HIGH);
            digitalWrite(GREEN_LED, LOW);
            consecutiveFailures = 0;
        } 
        else if (decision == "GREEN") {
            Serial.println("🟢 Decision: GREEN - Low traffic");
            digitalWrite(RED_LED, LOW);
            digitalWrite(GREEN_LED, HIGH);
            consecutiveFailures = 0;
        } 
        else {
            // Error or no response
            handleFailure();
        }
        
        Serial.println();
    }
}

// ================= FUNCTIONS =================

bool initPMU() {
    Serial.print("Initializing PMU... ");
    
    if (!PMU.begin(Wire, AXP2101_SLAVE_ADDRESS, I2C_SDA, I2C_SCL)) {
        return false;
    }
    
    // Camera power
    PMU.setALDO1Voltage(1800);
    PMU.enableALDO1();
    PMU.setALDO2Voltage(2800);
    PMU.enableALDO2();
    PMU.setALDO4Voltage(3000);
    PMU.enableALDO4();
    PMU.disableTSPinMeasure();
    
    Serial.println("OK");
    return true;
}

bool initCamera() {
    Serial.print("Initializing camera... ");
    
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;  // JPEG for network transfer
    config.frame_size   = FRAMESIZE_UXGA;  // Start with high resolution
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count     = 1;

    // Check for PSRAM and adjust settings
    if (psramFound()) {
        Serial.print("(PSRAM found) ");
        config.jpeg_quality = 10;
        config.fb_count = 2;
        config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
        // Limit frame size when PSRAM is not available
        Serial.print("(No PSRAM) ");
        config.frame_size = FRAMESIZE_SVGA;
        config.fb_location = CAMERA_FB_IN_DRAM;
    }

    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("FAILED (error 0x%x)\n", err);
        return false;
    }

    // Camera settings
    sensor_t *s = esp_camera_sensor_get();
    
    // Adjust based on sensor type
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    
    // Use QVGA for faster transfer (320x240)
    s->set_framesize(s, FRAMESIZE_QVGA);
    
    // LilyGo camera orientation fix
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);

    Serial.println("OK");
    return true;
}

bool connectWiFi() {
    Serial.printf("Connecting to WiFi: %s ", WIFI_SSID);
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - start > WIFI_TIMEOUT_MS) {
            Serial.println("TIMEOUT");
            return false;
        }
        delay(500);
        Serial.print(".");
    }
    
    Serial.printf("\n✅ Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    return true;
}

String sendImageToServer(uint8_t *imageData, size_t imageLen) {
    Serial.printf("📤 Sending to server... ");
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.setTimeout(HTTP_TIMEOUT_MS);
    
    // Create multipart form data
    String boundary = "----ESP32Boundary";
    String head = "--" + boundary + "\r\n";
    head += "Content-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n";
    head += "Content-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--" + boundary + "--\r\n";
    
    size_t totalLen = head.length() + imageLen + tail.length();
    
    // Create buffer for request body
    uint8_t *body = (uint8_t*)malloc(totalLen);
    if (!body) {
        Serial.println("❌ Memory allocation failed");
        return "ERROR";
    }
    
    // Build request body
    memcpy(body, head.c_str(), head.length());
    memcpy(body + head.length(), imageData, imageLen);
    memcpy(body + head.length() + imageLen, tail.c_str(), tail.length());
    
    // Set headers
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    
    // Send request
    int httpCode = http.POST(body, totalLen);
    free(body);
    
    if (httpCode == HTTP_CODE_OK) {
        String response = http.getString();
        http.end();
        
        // Parse JSON response
        DynamicJsonDocument doc(512);
        DeserializationError error = deserializeJson(doc, response);
        
        if (error) {
            Serial.printf("❌ JSON parse error: %s\n", error.c_str());
            return "ERROR";
        }
        
        int cars = doc["cars"];
        const char* decision = doc["decision"];
        
        Serial.printf("✅ Cars: %d, Decision: %s\n", cars, decision);
        return String(decision);
    } 
    else {
        Serial.printf("❌ HTTP error: %d\n", httpCode);
        http.end();
        return "ERROR";
    }
}

void handleFailure() {
    consecutiveFailures++;
    Serial.printf("⚠️ Failure count: %d/%d\n", consecutiveFailures, MAX_FAILURES);
    
    if (consecutiveFailures >= MAX_FAILURES) {
        // Fail-safe: default to GREEN
        Serial.println("⚠️ Too many failures, defaulting to GREEN");
        digitalWrite(RED_LED, LOW);
        digitalWrite(GREEN_LED, HIGH);
    } else {
        // Blink both LEDs to indicate error
        digitalWrite(RED_LED, HIGH);
        digitalWrite(GREEN_LED, HIGH);
        delay(100);
        digitalWrite(RED_LED, LOW);
        digitalWrite(GREEN_LED, LOW);
    }
}

void blinkError() {
    // Continuous red blink indicates fatal error
    while (1) {
        digitalWrite(RED_LED, HIGH);
        delay(200);
        digitalWrite(RED_LED, LOW);
        delay(200);
    }
}
