// ============================================================
// ESP32-S3 Sense — Standalone Camera Stream Server
// Board  : Seeed XIAO ESP32-S3 Sense
// Camera : OV2640 (built-in)
//
// Endpoints
//   http://<IP>/           → redirect to Python server UI
//   http://<IP>/stream     → MJPEG live stream  (port 80)
//   http://<IP>/capture    → single JPEG snapshot
//   http://<IP>/status     → JSON health {"ok":true,"heap":N}
//
// MJPEG is served on the same port-80 server (no separate port needed
// for the standalone use-case; the Python server embeds the /stream URL).
//
// Pin map matches Maddy_Flight_Controller.ino exactly (XIAO ESP32-S3 Sense).
// ============================================================

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// Rename esp_camera's sensor_t to avoid any future Adafruit conflicts
#define sensor_t esp_cam_sensor_t
#include "esp_camera.h"
#undef sensor_t

// ── WiFi ────────────────────────────────────────────────────
const char* WIFI_SSID     = "Madhan";
const char* WIFI_PASSWORD = "Pmkmk9495!!";

// ── OV2640 pin map (XIAO ESP32-S3 Sense) ───────────────────
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39
#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15
#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13

// ── Config ──────────────────────────────────────────────────
#define STREAM_FPS      15          // target ~15 fps
#define FRAME_DELAY_MS  (1000 / STREAM_FPS)
#define PYTHON_SERVER   "192.168.1.100:5050"  // update to PC's IP

// ── Globals ─────────────────────────────────────────────────
WebServer server(80);
bool      cameraReady = false;
String    myIP        = "";

// ── Camera init ─────────────────────────────────────────────
void setupCamera() {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_4;
  cfg.ledc_timer   = LEDC_TIMER_1;
  cfg.pin_d0    = CAM_PIN_D0;
  cfg.pin_d1    = CAM_PIN_D1;
  cfg.pin_d2    = CAM_PIN_D2;
  cfg.pin_d3    = CAM_PIN_D3;
  cfg.pin_d4    = CAM_PIN_D4;
  cfg.pin_d5    = CAM_PIN_D5;
  cfg.pin_d6    = CAM_PIN_D6;
  cfg.pin_d7    = CAM_PIN_D7;
  cfg.pin_xclk  = CAM_PIN_XCLK;
  cfg.pin_pclk  = CAM_PIN_PCLK;
  cfg.pin_vsync = CAM_PIN_VSYNC;
  cfg.pin_href  = CAM_PIN_HREF;
  cfg.pin_sccb_sda = CAM_PIN_SIOD;
  cfg.pin_sccb_scl = CAM_PIN_SIOC;
  cfg.pin_pwdn  = CAM_PIN_PWDN;
  cfg.pin_reset = CAM_PIN_RESET;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;  // always newest frame

  // PSRAM present on XIAO ESP32-S3 Sense — use higher res
  if (psramFound()) {
    cfg.frame_size   = FRAMESIZE_VGA;   // 640×480
    cfg.jpeg_quality = 12;              // 0-63, lower=better
    cfg.fb_count     = 2;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    cfg.frame_size   = FRAMESIZE_QVGA;  // 320×240
    cfg.jpeg_quality = 15;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init failed 0x%x\n", err);
    return;
  }

  // Fine-tune OV2640 settings
  esp_cam_sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gainceiling(s, (gainceiling_t)4);
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
  }

  cameraReady = true;
  Serial.printf("[CAM] OV2640 ready  PSRAM=%s  framesize=%s\n",
    psramFound() ? "yes" : "no",
    psramFound() ? "VGA 640x480" : "QVGA 320x240");
}

// ── HTTP handlers ────────────────────────────────────────────

// GET / → redirect to Python server UI
void handleRoot() {
  String url = "http://" + String(PYTHON_SERVER) + "/";
  server.sendHeader("Location", url, true);
  server.send(302, "text/plain", "");
}

// GET /capture → JPEG snapshot
void handleCapture() {
  if (!cameraReady) {
    server.send(503, "text/plain", "Camera not ready");
    return;
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Frame capture failed");
    return;
  }
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Cache-Control", "no-cache, no-store");
  server.sendHeader("X-Frame-Width",  String(fb->width));
  server.sendHeader("X-Frame-Height", String(fb->height));
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// GET /stream → multipart MJPEG
void handleStream() {
  if (!cameraReady) {
    server.send(503, "text/plain", "Camera not ready");
    return;
  }

  WiFiClient client = server.client();
  client.print(
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n\r\n"
  );

  unsigned long tLast = 0;
  while (client.connected()) {
    unsigned long now = millis();
    if (now - tLast < FRAME_DELAY_MS) {
      delay(1);
      continue;
    }
    tLast = now;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) continue;

    client.printf(
      "--frame\r\n"
      "Content-Type: image/jpeg\r\n"
      "Content-Length: %u\r\n\r\n",
      fb->len
    );
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);
  }
  client.stop();
}

// GET /status → JSON
void handleStatus() {
  StaticJsonDocument<128> doc;
  doc["ok"]       = cameraReady;
  doc["ip"]       = myIP;
  doc["heap"]     = (int)esp_get_free_heap_size();
  doc["psram"]    = psramFound();
  doc["fps_cfg"]  = STREAM_FPS;
  String out;
  serializeJson(doc, out);
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", out);
}

// ── WiFi setup ───────────────────────────────────────────────
void setupWiFi() {
  Serial.printf("[WIFI] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    myIP = WiFi.localIP().toString();
    Serial.printf("\n[WIFI] Connected — IP: %s\n", myIP.c_str());
    Serial.printf("[WIFI] Stream  : http://%s/stream\n",  myIP.c_str());
    Serial.printf("[WIFI] Capture : http://%s/capture\n", myIP.c_str());
    Serial.printf("[WIFI] Status  : http://%s/status\n",  myIP.c_str());
  } else {
    Serial.println("\n[WIFI] Failed — starting AP mode: ESP32-CAM-AP");
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32-CAM-AP", "maddy1234");
    myIP = WiFi.softAPIP().toString();
    Serial.printf("[WIFI] AP IP: %s\n", myIP.c_str());
  }
}

// ── setup / loop ─────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n[BOOT] ESP32-S3 Sense Camera Server");

  setupCamera();
  setupWiFi();

  server.on("/",        HTTP_GET, handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/stream",  HTTP_GET, handleStream);
  server.on("/status",  HTTP_GET, handleStatus);
  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });

  server.begin();
  Serial.println("[HTTP] Server started on port 80");
}

void loop() {
  server.handleClient();
}
