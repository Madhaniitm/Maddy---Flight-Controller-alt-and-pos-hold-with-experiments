# dRehmFlight ESP32-S3 Beta 1.0

**Seeed XIAO ESP32-S3 Sense Flight Controller** - Complete port of dRehmFlight for Seeed XIAO ESP32-S3 Sense

## What's New in This Version

This is a **complete port** of the Teensy-based dRehmFlight to run entirely on the **Seeed XIAO ESP32-S3 Sense** board. No external Teensy board needed!

### Key Changes from Teensy Version:

1. **Integrated WiFi** - WiFi server runs directly on ESP32-S3 (no separate ESP32 client needed)
2. **ESP32 LEDC PWM** - Replaced Teensy's `analogWrite()` and `PWMServo` library with ESP32's LEDC peripheral
3. **Seeed XIAO Pin Assignments** - Optimized for XIAO ESP32-S3 Sense compact form factor
4. **All Flight Control Logic Preserved** - IMU, PID, Madgwick filter, mixer unchanged
5. **MTF-01 Optical Flow Support** - Position hold and altitude hold using optical flow + rangefinder (UART)

## Hardware Requirements

### Required:
- **Seeed XIAO ESP32-S3 Sense** board
- **MPU6050** IMU (I2C)
- **4x Brushed Motors** (8520 or similar)
- **4x Motor Drivers** (MOSFET-based for brushed motors)
- **Battery** (1S LiPo, 3.7V recommended)

### Optional (for Optical Flow features):
- **MTF-01** Optical Flow + Rangefinder sensor (UART)
  - Provides altitude hold (5cm-3m range)
  - Provides position hold (requires textured surface)

## Pin Connections

### Seeed XIAO ESP32-S3 Sense Pinout:

```
I2C (MPU6050):
- SDA: GPIO 5 (D4 on XIAO)
- SCL: GPIO 6 (D5 on XIAO)

Motors (PWM via LEDC) - Quadcopter X Configuration:
- Motor 1 (Front Left):  GPIO 1 (D0 on XIAO)
- Motor 2 (Front Right): GPIO 2 (D1 on XIAO)
- Motor 3 (Back Right):  GPIO 3 (D2 on XIAO)
- Motor 4 (Back Left):   GPIO 4 (D3 on XIAO)

LED:
- Built-in LED: GPIO 21

MTF-01 Optical Flow (UART via Serial1):
- MTF-01 TX -> GPIO 44 (D7/RX on XIAO)
- MTF-01 RX -> GPIO 43 (D6/TX on XIAO)
- MTF-01 VCC -> 5V (check your sensor voltage requirement!)
- MTF-01 GND -> GND
```

### MPU6050 Connections:
```
MPU6050 VCC -> 3.3V
MPU6050 GND -> GND
MPU6050 SDA -> GPIO 5 (D4 on XIAO)
MPU6050 SCL -> GPIO 6 (D5 on XIAO)
```

### ToF Sensor Connections (if using):
```
TCA9548A VCC -> 3.3V
TCA9548A GND -> GND
TCA9548A SDA -> GPIO 8
TCA9548A SCL -> GPIO 9

VL53L0X sensors connect to TCA9548A channels:
- Channel 0 (SD0): Right sensor
- Channel 1 (SD1): Front sensor
- Channel 4 (SD4): Top sensor
- Channel 5 (SD5): Bottom sensor (altitude)

Note: Left (SD2) and Back (SD3) sensors disabled in code
```

## Required Arduino Libraries

Install these libraries via Arduino IDE Library Manager (Sketch → Include Library → Manage Libraries):

1. **Adafruit MPU6050** by Adafruit (this will also install Adafruit Unified Sensor library)
2. **WiFi** (built-in with ESP32 board support)
3. **WebServer** (built-in with ESP32 board support)
4. **WebSocketsServer** by Markus Sattler
5. **ArduinoJson** by Benoit Blanchon
6. **VL53L0X** by Pololu (if using ToF sensors)
7. **Wire** (built-in)

### How to Install Adafruit MPU6050:
1. Open Arduino IDE
2. Go to **Sketch → Include Library → Manage Libraries**
3. Search for "**Adafruit MPU6050**"
4. Click **Install** (it will also install dependencies automatically)
5. Wait for installation to complete

### Installing ESP32 Board Support

1. Open Arduino IDE
2. Go to **File → Preferences**
3. Add this URL to "Additional Boards Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**
5. Search for "ESP32" and install "esp32" by Espressif Systems
6. Select your board: **Tools → Board → ESP32 Arduino → ESP32S3 Dev Module** (or Xiao ESP32S3)

## WiFi Configuration

Edit these lines in the `.ino` file (near the top):

```cpp
const char* WIFI_SSID = "YourWiFiName";       // Change this
const char* WIFI_PASSWORD = "YourPassword";   // Change this
```

## How to Upload

1. Connect ESP32-S3 via USB
2. Open `dRehmFlight_ESP32S3_BETA_1.0.ino` in Arduino IDE
3. Select the correct board and port:
   - **Board**: ESP32S3 Dev Module (or Xiao ESP32S3)
   - **Port**: Your USB serial port
4. Click **Upload**
5. Open **Serial Monitor** (115200 baud) to see:
   - WiFi connection status
   - IP address
   - Sensor initialization status

## Connecting and Flying

### Step 1: Power Up
1. Power the ESP32-S3 via battery or USB
2. Wait for WiFi connection (LED will blink)
3. Note the IP address from Serial Monitor

### Step 2: Open Web Controller
1. Open the **ESP32S3_DroneController.html** file in your browser
2. Enter the ESP32-S3's IP address
3. Click "Connect"

**Note**: Use `ESP32S3_DroneController.html` for full MTF-01 support (position hold + landing)

### Step 3: Arming and Flight
1. **Throttle**: Stick at minimum (bottom)
2. **ARM Switch** (CH5): Toggle to ARM position (<1500 PWM)
3. Drone is armed when throttle is low and ARM is enabled
4. Increase throttle slowly to take off

### Manual Controls:
- **CH1**: Throttle (1000-2000)
- **CH2**: Roll (1000-2000, 1500=center)
- **CH3**: Pitch (1000-2000, 1500=center)
- **CH4**: Yaw (1000-2000, 1500=center)
- **CH5**: ARM switch (>1500=DISARMED, <1500=ARMED)

### Web Controller Features:
- **Virtual Joysticks**: Left (Throttle/Yaw), Right (Pitch/Roll)
- **ARM/DISARM Switch**: Enable/disable motors
- **Altitude Hold Switch**: Toggle altitude hold (MTF-01)
- **Position Hold Switch**: Toggle position hold (MTF-01)
- **Auto Takeoff**: Smooth throttle ramp to set percentage
- **Auto Landing**: MTF-01 guided descent to ground
- **Altitude Setpoint**: Adjust target altitude (0.2-3.0m)

## MTF-01 Optical Flow Features

### Altitude Hold
- Automatically maintains set altitude using MTF-01 rangefinder
- Toggle via WebSocket: `{"althold": 1}` (enable) or `{"althold": 0}` (disable)
- Set target altitude: `{"altitude": 1.0}` (in meters, 0.2m to 3.0m)
- Only activates when > 15cm off ground
- Valid range: 5cm to 3m

### Position Hold
- Holds X/Y position using optical flow sensor
- Toggle via WebSocket: `{"poshold": 1}` (enable) or `{"poshold": 0}` (disable)
- Toggle via Web Controller: "POSITION HOLD" switch
- Automatically locks position at current location when enabled
- Reset position to origin: `{"resetpos": 1}`
- Only works with good flow quality (>100/255)
- Requires valid altitude reading

### Auto Landing
- Smooth controlled descent using MTF-01 rangefinder
- Trigger via WebSocket: `{"land": 1}`
- Trigger via Web Controller: "LANDING PROTOCOL" button
- Descent rate: 30cm/second (adjustable in code)
- Automatically stops at 10cm above ground
- Centers all controls during descent

### How It Works
- **Optical Flow**: Measures pixel movement (velocity estimation)
- **Rangefinder**: Measures altitude (5cm-3m range)
- Position = Integrated (optical flow × altitude × scale factor)
- PID controllers adjust roll/pitch to maintain position
- PID controller adjusts throttle to maintain altitude

## IMU Calibration

The IMU (MPU6050) needs calibration to eliminate drift and ensure accurate attitude estimation. The flight controller includes a built-in calibration function.

### How to Calibrate IMU

1. **Place drone on flat, level surface** (very important!)
2. **Edit the code**: Open `dRehmFlight_ESP32S3_BETA_1.0.ino`
3. **Uncomment line 380**:
   ```cpp
   //calculate_IMU_error(); //UNCOMMENT THIS LINE
   ```
   Should become:
   ```cpp
   calculate_IMU_error(); //UNCOMMENTED
   ```
4. **Upload to ESP32-S3** and open Serial Monitor (115200 baud)
5. **Keep drone still** for ~10 seconds while it collects 12000 samples
6. **Copy the calibration values** printed in Serial Monitor:
   ```cpp
   float AccErrorX = 0.0234;
   float AccErrorY = -0.0156;
   float AccErrorZ = 0.0891;
   float GyroErrorX = -1.8432;
   float GyroErrorY = 1.6734;
   float GyroErrorZ = -1.2701;
   ```
7. **Paste these values** at lines 105-110 in the code (replace existing values)
8. **Comment out the calibration call** again:
   ```cpp
   //calculate_IMU_error(); //Commented out after calibration
   ```
9. **Upload again** with your custom calibration values

### When to Recalibrate

- After mounting IMU in different orientation
- If drone drifts significantly during hover
- After IMU replacement or hardware changes
- Every few months for best accuracy

**Note**: Current calibration values (lines 105-110) were measured on the reference hardware. Your IMU will have slightly different offsets.

## Debugging and Monitoring

### Serial Monitor Output (115200 baud)
The flight controller provides real-time debugging via Serial Monitor. To enable debug output, call these functions in the main loop:

**Available Print Functions:**
- `printRadioData()` - Radio PWM values (CH1-CH6)
- `printDesiredState()` - Desired throttle, roll, pitch, yaw
- `printGyroData()` - Gyroscope readings (deg/s)
- `printAccelData()` - Accelerometer readings (G's)
- `printRollPitchYaw()` - Current attitude angles (degrees)
- `printPIDoutput()` - PID controller outputs
- `printMotorCommands()` - Motor PWM values (0-255)
- `printLoopRate()` - Main loop execution time (microseconds)

**MTF-01 Specific Functions:**
- `printOpticalFlowData()` - Flow X/Y, quality, range, position
- `printAltitudeData()` - Current altitude, target, error, correction
- `printPositionData()` - Position hold status and coordinates

**ToF Sensor Functions (if enabled):**
- `printToFReadings()` - All ToF sensor distances

### How to Use Debug Functions

Add the desired print function to the `loop()` before `loopRate(2000)`:

```cpp
void loop() {
  // ... existing code ...

  // Add debugging output (uncomment as needed)
  // printRadioData();        // Show radio inputs
  // printRollPitchYaw();     // Show attitude
  // printMotorCommands();    // Show motor outputs
  // printOpticalFlowData();  // Show MTF-01 data
  // printAltitudeData();     // Show altitude hold

  loopRate(2000);
}
```

**Note**: Each function prints at ~100Hz (every 10ms). Only enable functions you need to avoid Serial Monitor overflow.

## Troubleshooting

### WiFi Won't Connect
- Check SSID and password
- Ensure 2.4GHz WiFi (ESP32 doesn't support 5GHz)
- Check serial monitor for error messages

### Motors Not Spinning
- Check ARM switch (CH5 must be < 1500)
- Verify throttle is > 1050 when armed
- Check motor driver connections
- Verify PWM pins are correct

### IMU Errors
- Check MPU6050 wiring (SDA/SCL correct?)
- Verify I2C pull-up resistors (usually built-in)
- Check serial monitor for WHO_AM_I response

### Drone Unstable/Oscillates
- Check PID gains (reduce Kp if oscillating)
- Ensure IMU is mounted flat and secure
- Calibrate IMU by keeping drone level during power-up
- Check motor directions match mixer configuration

### MTF-01 Optical Flow Not Working
- **Check UART Connections**:
  - MTF-01 TX → ESP32 GPIO 17 (RX2)
  - MTF-01 RX → ESP32 GPIO 18 (TX2)
  - Don't cross TX/RX incorrectly!
- **Check Power**: MTF-01 typically needs 5V (check your sensor datasheet)
- **Check Baud Rate**: Default 115200 (defined in code)
- **Serial Monitor**: Look for "MTF-01 initialized" message
- **Debug Mode**: Uncomment debug line in `readMTF01Data()` to see packets
- **Packet Format**: Check your sensor datasheet - format may vary by manufacturer
- **No Data**: Ensure sensor has clear view of ground (6cm-3m range)

### Position Hold Issues
- **Flow Quality Low**: Surface needs texture/patterns (won't work on plain white/black)
- **Altitude Too High**: Optical flow works best < 2m
- **Drifting**: Calibrate `pixel_to_cm_scale` factor (line 296 in code)
- **Unstable**: Reduce position PID gains (`Kp_position`, `Ki_position`, `Kd_position`)

## PID Tuning

Current gains are conservative for stability. Adjust in code if needed:

```cpp
// Angle Mode PID
float Kp_roll_angle = 0.3;    // Increase for faster response
float Ki_roll_angle = 0.01;   // Keep low to prevent wind-up
float Kd_roll_angle = 0.00;   // Usually keep at 0

// Rate Mode PID
float Kp_roll_rate = 0.08;    // Fine-tune for responsiveness
float Ki_roll_rate = 0.0;     // Usually disabled
float Kd_roll_rate = 0.01;    // Small value for damping

// Same for pitch and yaw
```

## Flight Control Modes

The flight controller includes three PID control modes. The default mode is **Cascaded Angle+Rate (controlANGLE2)**, which provides the best stability.

### Available Control Modes:

**1. Cascaded Angle+Rate Mode (controlANGLE2)** - **DEFAULT**
- Two-loop cascaded PID controller
- Outer loop: PID on angle error
- Inner loop: PID on rate error
- Best stability and performance
- Currently active in main loop

**2. Simple Angle Mode (controlANGLE)**
- Single-loop PID on angle error
- Yaw stabilized on rate from GyroZ
- Simpler but less stable than cascaded mode
- Available as alternative

**3. Rate/Acro Mode (controlRATE)**
- PID on gyro rate error only
- Direct control of rotation rates
- For experienced pilots and aerobatic flying
- No self-leveling
- Available as alternative

### How to Switch Control Modes:

Edit the main loop and replace `controlANGLE2()` with desired mode:

```cpp
void loop() {
  // ... existing code ...

  //PID Controller - choose one:
  controlANGLE2();  // Default - cascaded mode (best stability)
  // controlANGLE();   // Simple angle mode
  // controlRATE();    // Rate/acro mode

  // ... rest of code ...
}
```

**Note**: Only use one control function at a time. Comment out the others.

## Startup Calibration

The flight controller automatically runs two calibration routines on startup:

### 1. Attitude Calibration (Madgwick Filter Warmup)

**What it does:**
- Runs 10000 iterations of IMU loop at 2kHz (~5 seconds)
- Warms up Madgwick filter for accurate attitude estimation
- Eliminates the ~30 second convergence delay on takeoff
- Displays progress in Serial Monitor

**Requirements:**
- Keep drone flat and still during warmup
- Runs automatically on every power-up
- Cannot be skipped (critical for accurate flight)

**Serial Output:**
```
Calibrating attitude... (warming up Madgwick filter)
Progress: 2000 / 10000
Progress: 4000 / 10000
...
Attitude calibration complete!
Initial attitude - Roll: 0.15 Pitch: -0.23 Yaw: 0.00
```

### 2. IMU Error Calibration (Optional, One-Time)

See **IMU Calibration** section above for details on obtaining AccError and GyroError values.

## Advanced Features

### Tailsitter Configuration Support

The `switchRollYaw()` function is available for tailsitter-type aircraft configurations:

```cpp
//Switches roll and yaw axes for vertical takeoff/landing
switchRollYaw(reverseRoll, reverseYaw);
```

- `reverseRoll`: 1 (normal) or -1 (reversed)
- `reverseYaw`: 1 (normal) or -1 (reversed)
- Useful when transitioning between vertical and horizontal flight modes
- Call before PID controller to adjust control axes

### Parameter Fading Functions

The flight controller includes helper functions for smoothly transitioning parameters between values. Useful for dynamic flight mode switching or gain scheduling.

**floatFaderLinear() - Simple Linear Fade**

Fades a parameter between min and max values based on a binary state:

```cpp
float floatFaderLinear(float param, float param_min, float param_max, float fadeTime, int state, int loopFreq)
```

**Parameters:**
- `param`: Current value of parameter
- `param_min`: Minimum bound
- `param_max`: Maximum bound
- `fadeTime`: Time in seconds to complete fade
- `state`: 0 (fade to min) or 1 (fade to max)
- `loopFreq`: Loop frequency in Hz (typically 2000)

**Example - Fade PID gain based on channel 6:**
```cpp
void controlMixer() {
  if (channel_6_pwm > 1500) {
    // Fade to aggressive gains over 2 seconds
    Kp_roll_angle = floatFaderLinear(Kp_roll_angle, 0.2, 0.5, 2.0, 1, 2000);
  } else {
    // Fade to smooth gains over 2 seconds
    Kp_roll_angle = floatFaderLinear(Kp_roll_angle, 0.2, 0.5, 2.0, 0, 2000);
  }
  // ... rest of mixer code
}
```

**floatFaderLinear2() - Asymmetric Fade**

Fades a parameter to a desired value with different fade times for increasing vs decreasing:

```cpp
float floatFaderLinear2(float param, float param_des, float param_lower, float param_upper, float fadeTime_up, float fadeTime_down, int loopFreq)
```

**Parameters:**
- `param`: Current value of parameter
- `param_des`: Desired target value
- `param_lower`: Lower bound
- `param_upper`: Upper bound
- `fadeTime_up`: Time in seconds to fade upward
- `fadeTime_down`: Time in seconds to fade downward
- `loopFreq`: Loop frequency in Hz (typically 2000)

**Example - Fade gain to target with fast descent, slow climb:**
```cpp
void controlMixer() {
  float target_gain = (channel_6_pwm > 1500) ? 0.5 : 0.2;
  // Fast down (1s), slow up (2s)
  Kp_roll_angle = floatFaderLinear2(Kp_roll_angle, target_gain, 0.2, 0.5, 2.0, 1.0, 2000);
  // ... rest of mixer code
}
```

**Use Cases:**
- **Mode transitions**: Smooth transition between hover and forward flight
- **Gain scheduling**: Adjust PID gains based on speed or throttle
- **Dynamic configuration**: Switch between stable/aggressive flight characteristics
- **Smooth arming**: Gradually ramp up control authority

**Note**: Call these functions inside `controlMixer()` and update the parameter on every loop iteration for smooth fading.

## Technical Specifications

- **Loop Rate**: 2000 Hz (2kHz)
- **IMU Update**: 2000 Hz
- **ToF Update**: 20 Hz
- **WiFi Update**: ~50 Hz
- **Control Mode**: Cascaded angle+rate PID
- **Motor PWM**: 20 kHz, 8-bit (0-255)

## Safety Features

1. **Failsafe**: Motors cut if WiFi disconnects > 1 second
2. **Throttle Cut**: ARM switch must be enabled to fly
3. **Command Filtering**: Low-pass filter on all radio inputs
4. **Integrator Anti-Windup**: Prevents PID integral buildup
5. **Accelerometer Validation**: Rejects corrupted IMU data

## Performance Notes for Your Drone

### Current Configuration:
- **Weight**: 108 grams
- **Motors**: 8520 brushed
- **Battery**: 600mAh 1S LiPo
- **Props**: 65mm

### Expected Performance:
- **Thrust-to-Weight**: ~0.81:1 (marginal with all sensors)
- **Flight Time**: ~3-4 minutes
- **Max Throttle Needed**: 100% for stable hover

### Recommendations:
1. **Reduce weight** to 80-90g for better performance
2. **Remove unused sensors** (left, back ToF already disabled)
3. Consider **lighter battery** (400mAh) to save ~15g
4. Or **upgrade to more powerful motors** (0720 or 0820)

## Credits

- **Original dRehmFlight**: Nicholas Rehm
- **ESP32-S3 Port**: Claude (2026-01-26)
- **Madgwick Filter**: Sebastian Madgwick
- **MPU6050 Library**: Various contributors

## License

Same as original dRehmFlight (check original project for license terms)

## Support

For issues specific to this ESP32-S3 port, check:
1. Serial Monitor output for debug info
2. Verify all library versions are compatible
3. Test motors individually before flight
4. Always fly in a safe, open area

**Happy Flying! 🚁**
