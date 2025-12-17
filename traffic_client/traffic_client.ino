/*
 * ESP32-S3 Traffic Detection Client
 * LilyGo Camera S3
 * 
 * Captures JPEG images and sends to cloud server for car detection
 * Controls LEDs based on cloud response (RED/GREEN)
 * 
 * LEDs: GPIO 15 (RED), GPIO 16 (GREEN)
 */

#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include <HTTPClient.h>
#include <ArduinoJson.h>

#if (ESP_ARDUINO_VERSION) > ESP_ARDUINO_VERSION_VAL(3,0,0)
#error "Use ESP32 core <= 2.0.17"
#endif

// ================= POWER (AXP2101) =================
#define XPOWERS_CHIP_AXP2101
#include "XPowersLib.h"
#include "utilities.h"
XPowersPMU PMU;

// ================= WiFi CONFIG =================
const char* WIFI_SSID = "raj";
const char* WIFI_PASS = "12121212";

// ================= SERVER CONFIG =================
const char* SERVER_URL = "http://172.16.248.179:5000/analyze";

// ================= LED PINS =================
#define RED_LED    15
#define GREEN_LED  16

// ================= TIMING =================
#define CAPTURE_INTERVAL_MS 3000   // Send image every 3 seconds
#define HTTP_TIMEOUT_MS     15000  // HTTP request timeout

// ================= GLOBALS =================
unsigned long lastCapture = 0;
int consecutiveFailures = 0;
#define MAX_FAILURES 5

// ================= SETUP =================
void setup()
{
    Serial.begin(115200);
    delay(3000);
    Serial.println("\nTraffic Detection - LilyGO T-Camera S3");

    // Initialize LEDs
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, LOW);

    // ----- PMU INIT -----
    if (!PMU.begin(Wire, AXP2101_SLAVE_ADDRESS, I2C_SDA, I2C_SCL)) {
        Serial.println("PMU init failed");
        while (1);
    }

    PMU.setALDO1Voltage(1800); PMU.enableALDO1();
    PMU.setALDO2Voltage(2800); PMU.enableALDO2();
    PMU.setALDO4Voltage(3000); PMU.enableALDO4();
    PMU.disableTSPinMeasure();

    Serial.println("PMU OK");

    // ----- CAMERA CONFIG -----
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
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = FRAMESIZE_QVGA;   // 320x240
    config.jpeg_quality = 12;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        while (1);
    }

    // Sensor orientation fix
    sensor_t *s = esp_camera_sensor_get();
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);

    Serial.println("Camera OK");

    // ----- WIFI STATION MODE -----
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());

    Serial.println("\nSystem ready! Starting traffic detection...\n");
    
    // Flash green to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(GREEN_LED, HIGH);
        delay(200);
        digitalWrite(GREEN_LED, LOW);
        delay(200);
    }
}

// ================= LOOP =================
void loop()
{
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected, reconnecting...");
        WiFi.begin(WIFI_SSID, WIFI_PASS);
        delay(5000);
        return;
    }

    // Capture and analyze at intervals
    if (millis() - lastCapture >= CAPTURE_INTERVAL_MS) {
        lastCapture = millis();
        
        // Capture image
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            handleFailure();
            return;
        }

        Serial.printf("Captured: %d bytes\n", fb->len);

        // Send to server and get decision
        String decision = sendImageToServer(fb->buf, fb->len);
        
        // Return frame buffer
        esp_camera_fb_return(fb);

        // Control LEDs based on decision
        if (decision == "RED") {
            Serial.println("Decision: RED - High traffic");
            digitalWrite(RED_LED, HIGH);
            digitalWrite(GREEN_LED, LOW);
            consecutiveFailures = 0;
        } 
        else if (decision == "GREEN") {
            Serial.println("Decision: GREEN - Low traffic");
            digitalWrite(RED_LED, LOW);
            digitalWrite(GREEN_LED, HIGH);
            consecutiveFailures = 0;
        } 
        else {
            handleFailure();
        }
        
        Serial.println();
    }
}

// ================= SEND IMAGE TO SERVER =================
String sendImageToServer(uint8_t *imageData, size_t imageLen) {
    Serial.print("Sending to server... ");
    
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
        Serial.println("Memory allocation failed");
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
        StaticJsonDocument<512> doc;
        DeserializationError error = deserializeJson(doc, response);
        
        if (error) {
            Serial.printf("JSON parse error: %s\n", error.c_str());
            return "ERROR";
        }
        
        int cars = doc["cars"];
        const char* decision = doc["decision"];
        
        Serial.printf("Cars: %d, Decision: %s\n", cars, decision);
        return String(decision);
    } 
    else {
        Serial.printf("HTTP error: %d\n", httpCode);
        http.end();
        return "ERROR";
    }
}

// ================= FAILURE HANDLER =================
void handleFailure() {
    consecutiveFailures++;
    Serial.printf("Failure count: %d/%d\n", consecutiveFailures, MAX_FAILURES);
    
    if (consecutiveFailures >= MAX_FAILURES) {
        // Fail-safe: default to GREEN
        Serial.println("Too many failures, defaulting to GREEN");
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
