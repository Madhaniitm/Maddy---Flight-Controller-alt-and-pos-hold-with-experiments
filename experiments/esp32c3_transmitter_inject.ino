/*
 * ESP32-C3 Mini — Transmitter Signal Injector
 * ============================================
 * Replaces gimbal pot signals for Pitch, Roll, Yaw, Throttle.
 * Each channel: GPIO → 1kΩ → signal wire  (10µF cap to GND at junction)
 * Computer connects via USB Serial at 115200 baud.
 *
 * Wiring:
 *   GPIO 3 → 1kΩ → Pitch signal wire   (10µF to GND)
 *   GPIO 4 → 1kΩ → Roll signal wire    (10µF to GND)
 *   GPIO 5 → 1kΩ → Yaw signal wire     (10µF to GND)
 *   GPIO 6 → 1kΩ → Throttle signal wire (10µF to GND)
 *   3.3V pin → Transmitter battery +
 *   GND      → Transmitter battery -
 *
 * Serial commands (send from Python over USB):
 *   HOVER       → pitch + roll neutral, yaw neutral
 *   FORWARD     → pitch forward
 *   BACK        → pitch back
 *   LEFT        → roll left
 *   RIGHT       → roll right
 *   P:700       → set pitch to raw value 0-1023
 *   R:400       → set roll to raw value 0-1023
 *   Y:512       → set yaw to raw value 0-1023
 *   T:600       → set throttle to raw value 0-1023
 *   STATUS      → print current values
 */

// ── Pin assignments ───────────────────────────────────────────────────────────
#define PIN_PITCH     3
#define PIN_ROLL      4
#define PIN_YAW       5
#define PIN_THROTTLE  6

// ── PWM config ────────────────────────────────────────────────────────────────
#define PWM_FREQ      20000   // 20 kHz — well above audible, clean after RC filter
#define PWM_BITS      10      // 10-bit resolution: 0–1023

// ── Calibration (tune these after measuring your transmitter) ─────────────────
// Transmitter on 3.3V supply → neutral = 1.65V = 51.5% = ~528/1023
// Adjust NEUTRAL, FULL_HI, FULL_LO by measuring actual drone response
#define NEUTRAL       512     // mid-point (~1.65V)
#define FULL_HI       820     // stick pushed fully one way (~2.65V)
#define FULL_LO       200     // stick pushed fully other way (~0.65V)
#define THROTTLE_MIN  100     // throttle at rest (safety — drone off)
#define THROTTLE_HOVER 500    // throttle to hover (tune this per drone)

// ── LEDC channel assignments ──────────────────────────────────────────────────
#define CH_PITCH      0
#define CH_ROLL       1
#define CH_YAW        2
#define CH_THROTTLE   3

// ── Current values (for STATUS command) ──────────────────────────────────────
int cur_pitch    = NEUTRAL;
int cur_roll     = NEUTRAL;
int cur_yaw      = NEUTRAL;
int cur_throttle = THROTTLE_MIN;


void setChannel(int ch, int& cur, int val) {
  val = constrain(val, 0, 1023);
  ledcWrite(ch, val);
  cur = val;
}

void hover() {
  setChannel(CH_PITCH, cur_pitch, NEUTRAL);
  setChannel(CH_ROLL,  cur_roll,  NEUTRAL);
  setChannel(CH_YAW,   cur_yaw,   NEUTRAL);
  // throttle unchanged — operator controls throttle separately
}


void setup() {
  Serial.begin(115200);
  delay(500);

  // Set up PWM channels (Arduino ESP32 core 2.x API)
  ledcSetup(CH_PITCH,    PWM_FREQ, PWM_BITS);
  ledcSetup(CH_ROLL,     PWM_FREQ, PWM_BITS);
  ledcSetup(CH_YAW,      PWM_FREQ, PWM_BITS);
  ledcSetup(CH_THROTTLE, PWM_FREQ, PWM_BITS);

  ledcAttachPin(PIN_PITCH,    CH_PITCH);
  ledcAttachPin(PIN_ROLL,     CH_ROLL);
  ledcAttachPin(PIN_YAW,      CH_YAW);
  ledcAttachPin(PIN_THROTTLE, CH_THROTTLE);

  // Safe startup values
  ledcWrite(CH_PITCH,    NEUTRAL);
  ledcWrite(CH_ROLL,     NEUTRAL);
  ledcWrite(CH_YAW,      NEUTRAL);
  ledcWrite(CH_THROTTLE, THROTTLE_MIN);

  Serial.println("ESP32-C3 Transmitter Injector ready.");
  Serial.println("Commands: HOVER FORWARD BACK LEFT RIGHT");
  Serial.println("          P:val R:val Y:val T:val  (0-1023)");
  Serial.println("          STATUS");
}


void loop() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;

  // ── Preset commands ──────────────────────────────────────────────
  if (cmd == "HOVER") {
    hover();
    Serial.println("OK HOVER");
  }
  else if (cmd == "FORWARD") {
    setChannel(CH_PITCH, cur_pitch, FULL_HI);
    setChannel(CH_ROLL,  cur_roll,  NEUTRAL);
    Serial.println("OK FORWARD");
  }
  else if (cmd == "BACK") {
    setChannel(CH_PITCH, cur_pitch, FULL_LO);
    setChannel(CH_ROLL,  cur_roll,  NEUTRAL);
    Serial.println("OK BACK");
  }
  else if (cmd == "LEFT") {
    setChannel(CH_PITCH, cur_pitch, NEUTRAL);
    setChannel(CH_ROLL,  cur_roll,  FULL_LO);
    Serial.println("OK LEFT");
  }
  else if (cmd == "RIGHT") {
    setChannel(CH_PITCH, cur_pitch, NEUTRAL);
    setChannel(CH_ROLL,  cur_roll,  FULL_HI);
    Serial.println("OK RIGHT");
  }

  // ── Raw value commands ───────────────────────────────────────────
  else if (cmd.startsWith("P:")) {
    int v = cmd.substring(2).toInt();
    setChannel(CH_PITCH, cur_pitch, v);
    Serial.println("OK P:" + String(cur_pitch));
  }
  else if (cmd.startsWith("R:")) {
    int v = cmd.substring(2).toInt();
    setChannel(CH_ROLL, cur_roll, v);
    Serial.println("OK R:" + String(cur_roll));
  }
  else if (cmd.startsWith("Y:")) {
    int v = cmd.substring(2).toInt();
    setChannel(CH_YAW, cur_yaw, v);
    Serial.println("OK Y:" + String(cur_yaw));
  }
  else if (cmd.startsWith("T:")) {
    int v = cmd.substring(2).toInt();
    setChannel(CH_THROTTLE, cur_throttle, v);
    Serial.println("OK T:" + String(cur_throttle));
  }

  // ── Status ───────────────────────────────────────────────────────
  else if (cmd == "STATUS") {
    Serial.printf("P:%d R:%d Y:%d T:%d\n",
                  cur_pitch, cur_roll, cur_yaw, cur_throttle);
  }
  else {
    Serial.println("ERR unknown command: " + cmd);
  }
}
