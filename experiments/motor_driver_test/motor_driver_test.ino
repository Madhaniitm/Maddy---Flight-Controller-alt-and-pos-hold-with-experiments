/*
 * Motor Driver Test — N-MOSFET speed control
 * ============================================
 * Hardware: ESP32-S3-Sense, N-type MOSFET gate on D0
 * Control:  Web page slider (connect phone/laptop to WiFi AP)
 *
 * Steps:
 *   1. Flash this sketch
 *   2. Connect your phone/laptop WiFi to "MotorTest" (password: 12345678)
 *   3. Open browser → http://192.168.4.1
 *   4. Slide UP = faster, slide DOWN = slower
 *   5. Watch Serial Monitor for speed % and PWM values
 *
 * Wiring:
 *   ESP32-S3 D0 → MOSFET gate (through 100Ω resistor is good practice)
 *   MOSFET drain → motor -ve
 *   Motor +ve → battery +
 *   Battery - → ESP32 GND → MOSFET source
 */

#include <WiFi.h>
#include <WebServer.h>

#define MOTOR_PIN   D0
#define PWM_CH      0
#define PWM_FREQ    20000   // 20kHz — silent motor operation
#define PWM_BITS    8       // 0–255

const char* AP_SSID = "MotorTest";
const char* AP_PASS = "12345678";

WebServer server(80);
int motorSpeed = 0;   // 0–100 %

// ── Web page ─────────────────────────────────────────────────────────────────
const char HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Motor Speed</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: Arial, sans-serif;
      background: #1a1a2e;
      color: white;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 20px;
      min-height: 100vh;
    }
    h2  { color: #e94560; margin-bottom: 10px; font-size: 22px; }
    .speed-display {
      font-size: 64px;
      font-weight: bold;
      color: #e94560;
      margin: 20px 0;
      min-width: 160px;
      text-align: center;
    }
    .slider-section {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      margin: 20px 0;
    }
    .label { font-size: 13px; color: #aaa; letter-spacing: 1px; }
    input[type=range] {
      -webkit-appearance: slider-vertical;
      appearance: slider-vertical;
      width: 60px;
      height: 300px;
      cursor: pointer;
      accent-color: #e94560;
    }
    .buttons { display: flex; gap: 12px; margin-top: 24px; flex-wrap: wrap; justify-content: center; }
    .btn {
      background: #16213e;
      color: white;
      border: 2px solid #e94560;
      padding: 12px 28px;
      font-size: 15px;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn:hover  { background: #e94560; }
    .btn.danger { background: #e94560; border-color: #e94560; }
    .bar-wrap { width: 200px; background: #16213e; border-radius: 8px; height: 12px; margin-top: 16px; }
    .bar      { height: 12px; background: #e94560; border-radius: 8px; width: 0%; transition: width 0.1s; }
  </style>
</head>
<body>
  <h2>Motor Speed Control</h2>

  <div class="speed-display" id="pct">0%</div>

  <div class="bar-wrap"><div class="bar" id="bar"></div></div>

  <div class="slider-section">
    <span class="label">&#9650; FAST</span>
    <input type="range" id="spd" min="0" max="100" value="0"
           oninput="setSpeed(this.value)">
    <span class="label">&#9660; STOP</span>
  </div>

  <div class="buttons">
    <button class="btn danger" onclick="setSpeed(0)">STOP</button>
    <button class="btn" onclick="setSpeed(25)">25%</button>
    <button class="btn" onclick="setSpeed(50)">50%</button>
    <button class="btn" onclick="setSpeed(75)">75%</button>
    <button class="btn" onclick="setSpeed(100)">FULL</button>
  </div>

  <script>
    function setSpeed(v) {
      v = parseInt(v);
      document.getElementById('spd').value = v;
      document.getElementById('pct').innerText = v + '%';
      document.getElementById('bar').style.width = v + '%';
      fetch('/set?speed=' + v).catch(() => {});
    }
  </script>
</body>
</html>
)rawhtml";

// ── HTTP handlers ─────────────────────────────────────────────────────────────
void handleRoot() {
  server.send_P(200, "text/html", HTML);
}

void handleSet() {
  if (!server.hasArg("speed")) {
    server.send(400, "text/plain", "missing speed");
    return;
  }
  motorSpeed = constrain(server.arg("speed").toInt(), 0, 100);
  int duty   = map(motorSpeed, 0, 100, 0, 255);
  ledcWrite(PWM_CH, duty);
  Serial.printf("Speed: %3d%%   PWM duty: %3d / 255\n", motorSpeed, duty);
  server.send(200, "text/plain", "OK");
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);

  // PWM on MOSFET gate
  ledcSetup(PWM_CH, PWM_FREQ, PWM_BITS);
  ledcAttachPin(MOTOR_PIN, PWM_CH);
  ledcWrite(PWM_CH, 0);   // motor OFF at startup
  Serial.println("PWM on D0 initialised — motor OFF.");

  // WiFi Access Point (no router needed)
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.printf("WiFi AP: %s  password: %s\n", AP_SSID, AP_PASS);
  Serial.printf("Open browser → http://%s\n", WiFi.softAPIP().toString().c_str());

  server.on("/",    handleRoot);
  server.on("/set", handleSet);
  server.begin();
  Serial.println("Web server ready.");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
}
