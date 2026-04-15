//========================================================================================================================//
//                                                 USER-SPECIFIED DEFINES                                                 //                                                                 
//========================================================================================================================//

// WiFi Configuration
#define USE_WIFI_RX  //ESP32-S3 integrated WiFi receiver
const char* WIFI_SSID = "Madhan";  // WiFi network name
const char* WIFI_PASSWORD = "Pmkmk9495!!";  // WiFi password

// Camera (XIAO ESP32-S3 Sense built-in OV2640)
#define USE_CAMERA   // Comment out to disable

//ToF Sensors for obstacle avoidance and altitude hold (DISABLED)
//#define USE_TOF_SENSORS  //VL53L0X sensors via TCA9548A multiplexer

//Optical Flow Sensor for position and altitude hold
//#define USE_MTF01_OPTICAL_FLOW  //MTF-01 optical flow + rangefinder via I2C

//LiteWing Drone Positioning Module by Circuit Digest
//  - Altitude hold: VL53L1X ToF sensor via I2C + 1D Kalman [z,vz] in firmware
//  - Position hold: PMW3901 optical flow via SPI + 2D Kalman [x,y,vx,vy] in firmware
//  Libraries required: "VL53L1X" by Pololu (install via Arduino Library Manager)
//  NOTE: Only enable ONE sensor define at a time (mutually exclusive)
#define USE_LITEWING_MODULE  //Enable LiteWing altitude + position hold

//IMU Selection
#define USE_MPU6050_I2C //MPU6050 via I2C (default for ESP32)

//Gyro full scale range (deg/sec)
#define GYRO_250DPS //Default

//Accelerometer full scale range (G's)
#define ACCEL_2G //Default

//========================================================================================================================//

//REQUIRED LIBRARIES

#include <Wire.h>      //I2C communication
#include <WiFi.h>      //ESP32 WiFi
#include <WebServer.h> //HTTP server
#include <WebSocketsServer.h> //WebSocket for real-time control
#include <ArduinoJson.h> //JSON parsing

// IMU Library first — must come before esp_camera.h to avoid sensor_t conflict
// (both Adafruit_Sensor.h and esp32-camera/sensor.h define 'sensor_t')
#if defined USE_MPU6050_I2C
  #include <Adafruit_MPU6050.h>
  #include <Adafruit_Sensor.h>
  Adafruit_MPU6050 mpu;
#else
  #error No MPU defined...
#endif

// Camera library (installed "esp32-camera" via Arduino board package)
// Rename esp_camera's sensor_t to avoid conflict with Adafruit's sensor_t above.
// camera code never uses sensor_t directly, so this is safe.
#if defined USE_CAMERA
  #define sensor_t esp_cam_sensor_t
  #include "esp_camera.h"
  #undef sensor_t
  // XIAO ESP32-S3 Sense OV2640 pin map
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
  bool       cameraReady = false;
  WiFiServer streamServer(82);  // MJPEG live stream on port 82
#endif

// (Adafruit MPU6050 included above, before esp_camera, to avoid sensor_t conflict)

//========================================================================================================================//
//NOTE: Gyro and accel scale ranges are configured in IMUinit() using Adafruit library functions

//========================================================================================================================//
//                                               USER-SPECIFIED VARIABLES                                                 //                           
//========================================================================================================================//

//Radio failsafe values for every channel in the event that bad receiver data is detected
unsigned long channel_1_fs = 1000; //throttle
unsigned long channel_2_fs = 1500; //roll
unsigned long channel_3_fs = 1500; //pitch
unsigned long channel_4_fs = 1500; //yaw
unsigned long channel_5_fs = 2000; //ARM switch (>1500 = disarmed)
unsigned long channel_6_fs = 2000; //aux1

//Filter parameters - Defaults tuned for 2kHz loop rate
float B_madgwick = 0.03;  //Madgwick filter parameter
float B_accel = 0.14;     //Accelerometer LP filter parameter
float B_gyro = 0.14;      //Gyro LP filter parameter
float B_mag = 1.0;        //Magnetometer LP filter parameter

//IMU calibration parameters - calibrate IMU using calculate_IMU_error() in the void setup() to get these values, then comment out calculate_IMU_error()
float AccErrorX = 0.0870;
float AccErrorY = -0.0070;
float AccErrorZ = -0.0586;
float GyroErrorX = 1.1473;
float GyroErrorY = 1.4808;
float GyroErrorZ = 0.2825;

//Controller parameters
float i_limit = 20.0;     //Integrator saturation level
float maxRoll = 10.0;     //Max roll angle (degrees)
float maxPitch = 10.0;    //Max pitch angle (degrees)
float maxYaw = 90.0;      //Max yaw rate (deg/s)

// PID GAINS - TUNED FOR STABLE FLIGHT
float Kp_roll_angle = 0.3;    
float Ki_roll_angle = 0.01;   
float Kd_roll_angle = 0.00;  
float B_loop_roll = 0.95;      

float Kp_pitch_angle = 0.3;   
float Ki_pitch_angle = 0.01;  
float Kd_pitch_angle = 0.00; 
float B_loop_pitch = 0.95;     

float Kp_roll_rate = 0.08;    
float Ki_roll_rate = 0.0;     
float Kd_roll_rate = 0.01;     

float Kp_pitch_rate = 0.08;   
float Ki_pitch_rate = 0.0;    
float Kd_pitch_rate = 0.01;    

float Kp_yaw = 0.06;   // Increased from 0.035 for stronger immediate response
float Ki_yaw = 0.0;  // Reduced from 0.008 to prevent yaw windup
float Kd_yaw = 0.008;  // Increased from 0.004 for better damping

// Motor compensation and yaw scale (can be tuned via web UI)
float motor_comp_m1 = 1.00;  //Front-Right CCW
float motor_comp_m2 = 1.00;  //Back-Right CW
float motor_comp_m3 = 1.00;  //Back-Left CCW
float motor_comp_m4 = 1.00;  //Front-Left CW
float yaw_scale = 1.0;  //Yaw mixer scaling (adjustable via UI)
int DUTY_IDLE = 45;     //Minimum PWM baseline when armed (adjustable via UI)
int loop_rate_hz = 2000; //Main loop rate in Hz (adjustable via UI, 100-2000)           

//========================================================================================================================//
//                                SEEED XIAO ESP32-S3 SENSE PIN ASSIGNMENTS                                              //
//========================================================================================================================//

// Seeed XIAO ESP32-S3 Sense I2C pins
#define SDA_PIN 5   // I2C SDA (D4 on XIAO)
#define SCL_PIN 6   // I2C SCL (D5 on XIAO)

//Motor PWM pins using ESP32 LEDC (4 motors for quadcopter)
const int m1Pin = 1;   //Front Right motor (D0 on XIAO) - CCW
const int m2Pin = 2;   //Back Right motor  (D1 on XIAO) - CW
const int m3Pin = 3;   //Back Left motor   (D2 on XIAO) - CCW
const int m4Pin = 4;   //Front Left motor  (D3 on XIAO) - CW

//LED pin
const int LED_PIN = 21;  //Built-in LED on XIAO ESP32-S3

//MTF-01 Optical Flow UART pins (defined in MTF-01 section if enabled)
// Uses D6 (GPIO43 TX) and D7 (GPIO44 RX) on XIAO ESP32-S3

//========================================================================================================================//

// WiFi Server Objects
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81);

// Control channel values (PWM: 1000-2000)
unsigned long channel_1_pwm = 1000; // Throttle
unsigned long channel_2_pwm = 1500; // Roll
unsigned long channel_3_pwm = 1500; // Pitch
unsigned long channel_4_pwm = 1500; // Yaw
unsigned long channel_5_pwm = 2000; // ARM (default disarmed)
unsigned long channel_6_pwm = 1000; // Aux1
unsigned long channel_1_pwm_prev, channel_2_pwm_prev, channel_3_pwm_prev, channel_4_pwm_prev;

// WiFi status
bool clientConnected = false;
unsigned long lastCommandTime = 0;
const unsigned long FAILSAFE_TIMEOUT = 1000; // 1 second


//ToF Sensor variables and flags (if enabled)
#if defined USE_TOF_SENSORS
  #include <VL53L0X.h>
  
  // TCA9548A I2C Multiplexer address
  #define TCA9548A_ADDRESS 0x70
  
  // VL53L0X sensor objects
  VL53L0X tof_right, tof_front, tof_left, tof_back, tof_top, tof_bottom;
  
  // Sensor readings in millimeters
  uint16_t distance_right = 8190;
  uint16_t distance_front = 8190;
  uint16_t distance_left = 8190;
  uint16_t distance_back = 8190;
  uint16_t distance_top = 8190;
  uint16_t distance_bottom = 8190;
  
  // Obstacle avoidance threshold (mm)
  const uint16_t OBSTACLE_THRESHOLD = 200; // 20cm
  
  // Altitude hold variables
  float altitude_setpoint = 1000.0; // Target altitude in mm (default 1m)
  float altitude_current = 0.0;     // Current altitude in mm
  float altitude_error = 0.0;
  float altitude_error_integral = 0.0;
  float altitude_error_previous = 0.0;
  
  // Altitude PID gains
  float Kp_altitude = 0.5;
  float Ki_altitude = 0.1;
  float Kd_altitude = 0.2;
  
  // Altitude hold output
  float altitude_correction = 0.0;
  
  // Obstacle flags
  bool obstacle_right = false;
  bool obstacle_front = false;
  bool obstacle_left = false;
  bool obstacle_back = false;
  bool obstacle_top = false;
  
  // Feature enable flags
  bool altitude_hold_enabled = true;
  bool obstacle_avoid_enabled = true;
  
  // Timing for sensor updates
  unsigned long tof_update_time = 0;
  const unsigned long TOF_UPDATE_INTERVAL = 50; // 50ms (20Hz)
#endif

//MTF-01 Optical Flow Sensor variables and flags (if enabled)
#if defined USE_MTF01_OPTICAL_FLOW
  // MTF-01 UART pins (using Serial1 on XIAO ESP32-S3 Sense)
  #define MTF01_RX_PIN 44  // D7 on XIAO (GPIO44, Serial1 RX)
  #define MTF01_TX_PIN 43  // D6 on XIAO (GPIO43, Serial1 TX)
  #define MTF01_BAUD_RATE 115200  // Standard baud rate for MTF-01

  // Serial buffer for parsing (increased for Micolink protocol, max packet ~71 bytes)
  uint8_t mtf01_buffer[80];
  uint8_t mtf01_buffer_index = 0;

  // Optical flow readings (pixels/second)
  int16_t flow_x = 0;  // Changed to int16 for raw sensor data
  int16_t flow_y = 0;  // Changed to int16 for raw sensor data
  uint8_t flow_quality = 0;  // 0-255, higher is better

  // Rangefinder reading (millimeters)
  uint32_t range_mm = 0;  // uint32 supports 0-8m range

  // Position hold variables
  float position_x = 0.0;      // Integrated position in cm
  float position_y = 0.0;      // Integrated position in cm
  float position_x_setpoint = 0.0;  // Target X position
  float position_y_setpoint = 0.0;  // Target Y position

  // Position PID errors
  float pos_x_error = 0.0;
  float pos_x_error_integral = 0.0;
  float pos_x_error_previous = 0.0;
  float pos_y_error = 0.0;
  float pos_y_error_integral = 0.0;
  float pos_y_error_previous = 0.0;

  // Position PID gains
  float Kp_position = 0.3;
  float Ki_position = 0.05;
  float Kd_position = 0.1;

  // Position hold outputs (added to roll/pitch commands)
  float position_x_correction = 0.0;
  float position_y_correction = 0.0;

  // Altitude hold variables (using rangefinder)
  float altitude_setpoint = 1000.0; // Target altitude in mm (default 1m)
  float altitude_current = 0.0;     // Current altitude in mm
  float altitude_error = 0.0;
  float altitude_error_integral = 0.0;
  float altitude_error_previous = 0.0;

  // Altitude PID gains
  float Kp_altitude = 0.5;
  float Ki_altitude = 0.1;
  float Kd_altitude = 0.2;

  // Altitude hold output
  float altitude_correction = 0.0;

  // Feature enable flags
  bool altitude_hold_enabled = false;   // Altitude hold OFF by default
  bool position_hold_enabled = false;   // Position hold OFF by default

  // Landing control variables
  bool landing_active = false;
  unsigned long landing_start_time = 0;
  float landing_initial_altitude = 0.0;
  const float LANDING_DESCENT_RATE = 300.0; // mm/s (30cm/s)
  const float LANDING_GROUND_THRESHOLD = 100.0; // mm (10cm - consider landed)

  // Timing for sensor updates
  unsigned long optical_flow_update_time = 0;
  const unsigned long OPTICAL_FLOW_UPDATE_INTERVAL = 20; // 20ms (50Hz)

  // Conversion factor: pixels to cm (depends on altitude and sensor FOV)
  // This needs calibration! Typical value for MTF-01: ~0.1 cm per pixel at 1m altitude
  float pixel_to_cm_scale = 0.1;
#endif

//LiteWing Drone Positioning Module - Kalman altitude estimator + 2-stage PID
//  Mirrors: kalman_core.c (1-D Kalman [z,vz]) + position_controller_pid.c (pidZ + pidVZ)
#if defined USE_LITEWING_MODULE
  #include <VL53L1X.h>   // Pololu VL53L1X library - install via Arduino Library Manager

  VL53L1X lw_tof;
  bool lw_tof_ready = false;

  // =========================================================================
  // 9-STATE EKF  (exact port of kalman_core.c — Bitcraze/Crazyflie)
  // States: [X,Y,Z (world pos m), PX,PY,PZ (body vel m/s), D0,D1,D2 (att err rad)]
  // Quaternion kc_q[4]=[w,x,y,z] and rotation matrix kc_R[3][3] maintained alongside.
  // =========================================================================
  #define KC_STATE_X   0
  #define KC_STATE_Y   1
  #define KC_STATE_Z   2
  #define KC_STATE_PX  3
  #define KC_STATE_PY  4
  #define KC_STATE_PZ  5
  #define KC_STATE_D0  6
  #define KC_STATE_D1  7
  #define KC_STATE_D2  8
  #define KC_STATE_DIM 9

  float kc_S[KC_STATE_DIM] = {0};          // EKF states
  float kc_P[KC_STATE_DIM][KC_STATE_DIM];  // 9×9 error covariance
  float kc_q[4] = {1, 0, 0, 0};            // attitude quaternion [w, x, y, z]
  float kc_R[3][3];                         // rotation matrix (body→world)
  bool  kc_initialized = false;

  // Process noise (from kalman_core.c defaults)
  float kc_procNoiseAcc_xy   = 0.5f;
  float kc_procNoiseAcc_z    = 1.0f;
  float kc_procNoiseVel      = 0.0f;
  float kc_procNoisePos      = 0.0f;
  float kc_procNoiseAtt      = 0.0f;
  float kc_measNoiseGyro_rp  = 0.1f;   // rad/s
  float kc_measNoiseGyro_yaw = 0.1f;   // rad/s
  // Measurement noise
  float kc_tof_stdDev  = 0.05f;   // VL53L1X std dev (m)
  float kc_flow_stdDev = 2.0f;    // PMW3901 default pixel noise (pixels)
  bool  kc_quadIsFlying = false;  // mirrors quadIsFlying in estimator_kalman.c

  // Raw ToF reading + new-data flag
  float lw_tof_raw_m    = 0.0f;
  bool  lw_tof_new_data = false;
  unsigned long lw_tof_update_ms = 0;
  const unsigned long LW_TOF_INTERVAL_MS = 20;  // 50 Hz

  // ---- Altitude setpoint ----
  float lw_altitude_setpoint_m = 0.40f;   // metres (default 40 cm)

  // ---- Outer Z-position PID  (mirrors pidZ: kp=1.6 ki=0.5 kd=0) ----
  // Output: vertical velocity setpoint (m/s), clamped to ±zVelMax
  float lw_pidZ_kp = 1.6f, lw_pidZ_ki = 0.5f, lw_pidZ_kd = 0.0f;
  float lw_pidZ_integral = 0.0f, lw_pidZ_prev_err = 0.0f;
  float lw_vel_z_sp = 0.0f;               // output of outer loop

  // ---- Inner Z-velocity PID  (mirrors pidVZ: kp~22 ki~15 — rescaled for 0-1 thrust) ----
  float lw_pidVZ_kp = 0.70f, lw_pidVZ_ki = 0.30f, lw_pidVZ_kd = 0.0f;
  float lw_pidVZ_integral = 0.0f;
  float lw_thrust_correction = 0.0f;      // added to hover throttle → thro_des

  // ---- thrustBase equivalent  (mirrors position_controller_pid.c thrustBase) ----
  // Auto-captured from stick when alt-hold first engages; also settable via
  // WebSocket {"hover_thr": 0.48}.  Represents the throttle that just hovers.
  float lw_hover_throttle      = 0.50f;
  bool  lw_hover_thr_captured  = false;

  // ---- Feature flags ----
  bool lw_althold_enabled = false;   // {"althold":1}  via WebSocket
  bool lw_poshold_active  = false;   // {"poshold":1}  via WebSocket

  // =========================================================================
  // PMW3901 Optical Flow Sensor (SPI) — mirrors pmw3901.c + flowdeck_v1v2.c
  // XIAO ESP32-S3 hardware SPI: SCK=GPIO7(D8), MISO=GPIO8(D9), MOSI=GPIO9(D10)
  // CS pin: GPIO44 (D7) — free when USE_MTF01_OPTICAL_FLOW is disabled
  // =========================================================================
  #define PMW3901_CS_PIN 44

  // Motion burst struct (mirrors pmw3901.h motionBurst_t)
  struct motionBurst_t {
    uint8_t  motion;
    uint8_t  observation;
    int16_t  deltaX;
    int16_t  deltaY;
    uint8_t  squal;
    uint8_t  rawDataSum;
    uint8_t  maxRawData;
    uint8_t  minRawData;
    uint16_t shutter;
  };
  motionBurst_t pmw_motion;
  bool pmw3901_ready = false;
  unsigned long pmw_last_ms = 0;
  float pmw_dt = 0.01f;


  // ---- XY position setpoint (captured when poshold engages) ----
  float lw_posX_sp = 0.0f, lw_posY_sp = 0.0f;

  // ---- Outer XY-position PID  (mirrors pidX/Y: kp=1.2 ki=0.30 kd=0) ----
  // Output: velocity setpoint (m/s), clamped to ±1.0
  float lw_pidX_kp = 1.2f, lw_pidX_ki = 0.30f, lw_pidX_kd = 0.0f;
  float lw_pidX_int = 0.0f, lw_pidX_prev = 0.0f;
  float lw_pidY_kp = 1.2f, lw_pidY_ki = 0.30f, lw_pidY_kd = 0.0f;
  float lw_pidY_int = 0.0f, lw_pidY_prev = 0.0f;
  float lw_velX_sp = 0.0f, lw_velY_sp = 0.0f;

  // ---- Inner XY-velocity PID  (mirrors pidVX/Y, output normalized -1..+1) ----
  // Normalized output * maxRoll/maxPitch = degrees correction
  float lw_pidVX_kp = 0.90f, lw_pidVX_ki = 0.15f, lw_pidVX_kd = 0.05f;
  float lw_pidVX_int = 0.0f, lw_pidVX_prev = 0.0f;
  float lw_pidVY_kp = 0.90f, lw_pidVY_ki = 0.15f, lw_pidVY_kd = 0.05f;
  float lw_pidVY_int = 0.0f, lw_pidVY_prev = 0.0f;

  // Roll/pitch corrections added to roll_des/pitch_des in getDesState (normalized -1..+1)
  float lw_roll_correction  = 0.0f;
  float lw_pitch_correction = 0.0f;
#endif

//DECLARE GLOBAL VARIABLES

//General stuff
float dt;
unsigned long current_time, prev_time;
unsigned long print_counter, serial_counter;
unsigned long blink_counter, blink_delay;
bool blinkAlternate;

//IMU:
float AccX, AccY, AccZ;
float AccX_prev, AccY_prev, AccZ_prev;
float GyroX, GyroY, GyroZ;
float GyroX_prev, GyroY_prev, GyroZ_prev;
float MagX, MagY, MagZ;
float MagX_prev, MagY_prev, MagZ_prev;
float roll_IMU, pitch_IMU, yaw_IMU;
float roll_IMU_prev, pitch_IMU_prev;
float q0 = 1.0f; //Initialize quaternion for madgwick filter
float q1 = 0.0f;
float q2 = 0.0f;
float q3 = 0.0f;

//Normalized desired state:
float thro_des, roll_des, pitch_des, yaw_des;
float roll_passthru, pitch_passthru, yaw_passthru;

//Controller:
float error_roll, error_roll_prev, roll_des_prev, integral_roll, integral_roll_il, integral_roll_ol;
float integral_roll_prev, integral_roll_prev_il, integral_roll_prev_ol, derivative_roll, roll_PID = 0;
float error_pitch, error_pitch_prev, pitch_des_prev, integral_pitch, integral_pitch_il, integral_pitch_ol;
float integral_pitch_prev, integral_pitch_prev_il, integral_pitch_prev_ol, derivative_pitch, pitch_PID = 0;
float error_yaw, error_yaw_prev, integral_yaw, integral_yaw_prev, derivative_yaw, yaw_PID = 0;

//Mixer
float m1_command_scaled, m2_command_scaled, m3_command_scaled, m4_command_scaled;
int m1_command_PWM, m2_command_PWM, m3_command_PWM, m4_command_PWM;

//Flight status
bool armedFly = false;

// Telemetry broadcast timing (10 Hz)
unsigned long lastTelemetryTime = 0;

//========================================================================================================================//
//                                                      VOID SETUP                                                        //                           
//========================================================================================================================//

void setup() {
  Serial.begin(115200);
  delay(500);
  
  Serial.println("\n\n========================================");
  Serial.println("ESP32-S3 Flight Controller - dRehmFlight");
  Serial.println("========================================\n");
  
  //Initialize LED pin
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); //LED on during setup
  
  //Initialize ESP32 LEDC PWM for motors
  setupMotorPWM();
  
  //Initialize I2C for IMU and ToF sensors
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000); //400kHz I2C
  delay(100);
  
  //Initialize IMU
  IMUinit();

  //Get IMU error to zero accelerometer and gyro readings, assuming vehicle is level when powered up
  //calculate_IMU_error(); //Calibration parameters printed to serial monitor. 

  //Warm up Madgwick filter for accurate angles on startup (eliminates 30-second convergence delay)
  calibrateAttitude();

  #if defined USE_LITEWING_MODULE
  //Initialize LiteWing VL53L1X altitude sensor
  litewingSetup();
  #endif

  #if defined USE_TOF_SENSORS
  //Initialize ToF sensors
  tofSetup();
  #endif

  #if defined USE_MTF01_OPTICAL_FLOW
  //Initialize MTF-01 optical flow sensor
  mtf01Setup();
  #endif

  //Initialize WiFi
  setupWiFi();

  // Initialize camera and start MJPEG stream task on Core 0
  #if defined USE_CAMERA
  setupCamera();
  if (cameraReady) {
    xTaskCreatePinnedToCore(cameraStreamTask, "camStream", 8192, NULL, 1, NULL, 0);
  }
  #endif

  //Set radio channels to safe defaults
  channel_1_pwm = channel_1_fs;
  channel_2_pwm = channel_2_fs;
  channel_3_pwm = channel_3_fs;
  channel_4_pwm = channel_4_fs;
  channel_5_pwm = channel_5_fs;
  channel_6_pwm = channel_6_fs;
  
  //Initialize motors to stopped state (4 motors for quadcopter)
  m1_command_PWM = 0;
  m2_command_PWM = 0;
  m3_command_PWM = 0;
  m4_command_PWM = 0;
  armMotors();
  
  //Indicate entering main loop with 3 quick blinks
  setupBlink(3, 160, 70);
  
  Serial.println("\n========== SETUP COMPLETED ==========");
  Serial.println("Ready to fly!");
  Serial.println("\n*** DEBUG MODE ACTIVE ***");
  Serial.println("Watch Serial Monitor for:");
  Serial.println("1. Radio CH4 (yaw) should be ~1500");
  Serial.println("2. MTF-01: Range 50-3000mm, Quality >100");
  Serial.println("3. Motors m1-m4 should be balanced");
  Serial.println("4. Altitude hold DISABLED by default");
  Serial.println("======================================\n");
}

//========================================================================================================================//
//                                                       MAIN LOOP                                                        //                           
//========================================================================================================================//
                                                  
void loop() {
  //Keep track of time and calculate dt
  prev_time = current_time;
  current_time = micros();
  dt = (current_time - prev_time)/1000000.0;

  // NOTE: handleWiFi() moved to END of loop — after commandMotors() — so that
  // WiFi stack processing (webSocket.loop() latency, up to ~500 µs on single-core
  // ESP32) does not inject timing jitter into the IMU→Madgwick→PID→motor path.
  // getCommands() reads channel values written by the previous WiFi message,
  // so the one-loop delay (~250 µs at 4 kHz) is negligible and has zero effect
  // on attitude or Madgwick.
  getCommands();
  failSafe();

  #if defined USE_LITEWING_MODULE
  // VL53L1X: lightweight check every loop; internally gated to 50 Hz
  litewingReadAltitude();
  // EKF + PID rate-limited to 250 Hz — 9×9 matrix ops are too heavy for 4 kHz
  // with camera also streaming. 250 Hz is 4× faster than the Crazyflie EKF (1 kHz
  // on a 500 Hz loop) so accuracy is unaffected.
  static uint32_t lw_ekf_prev_us = 0;
  uint32_t lw_ekf_now_us = micros();
  if (lw_ekf_now_us - lw_ekf_prev_us >= 4000UL) {   // 250 Hz
    float lw_dt = (lw_ekf_now_us - lw_ekf_prev_us) * 1e-6f;
    lw_ekf_prev_us = lw_ekf_now_us;
    litewingAltitudeHold(lw_dt);
    litewingPositionHold(lw_dt);
  }
  #endif

  #if defined USE_TOF_SENSORS
  //Update ToF sensor readings
  updateToFSensors();
  #endif

  #if defined USE_MTF01_OPTICAL_FLOW
  //Update MTF-01 optical flow and rangefinder
  readMTF01Data();
  updatePositionEstimate(dt);
  #endif

  loopBlink(); //Indicate main loop is running

  //Get vehicle state from IMU
  getIMUdata();
  Madgwick(GyroX, -GyroY, -GyroZ, -AccX, AccY, AccZ, MagY, -MagX, MagZ, dt);

  #if defined USE_TOF_SENSORS
  //Apply altitude hold PID correction
  applyAltitudeHold(dt);
  #endif

  #if defined USE_MTF01_OPTICAL_FLOW
  //Apply altitude hold using MTF-01 rangefinder
  applyAltitudeHoldMTF01(dt);

  //Apply auto-landing if active
  applyAutoLanding(dt);
  #endif

  //Compute desired state
  getDesState();

  //PID Controller - cascaded angle mode
  controlANGLE2();

  //Actuator mixing
  controlMixer();
  scaleCommands();

  //Check arming status
  armedStatus();

  //Throttle cut check
  throttleCut();

  //Command motors
  commandMotors();

  // WiFi processing moved here (was top of loop) so it never interrupts the
  // IMU→Madgwick→PID→motor critical path. Any command received here takes effect
  // on the NEXT loop iteration via getCommands() — acceptable at 4 kHz.
  handleWiFi();

  //DEBUG MODE ENABLED - UNCOMMENT TO DIAGNOSE ISSUES
  //printRadioData();        // Check if yaw CH4 is centered at 1500
  //printDesiredState();     // Check throttle and yaw commands
  //printPIDoutput();        // Check PID outputs
  //printMotorCommands();    // Check if motors are balanced
  //printRollPitchYaw(); //check attitude

  #if defined USE_MTF01_OPTICAL_FLOW
  //printOpticalFlowData();  // CHECK MTF-01 DATA! Range and quality must be valid
  //printAltitudeData();     // Check altitude hold corrections
  //printPositionData();     // Check position hold
  #endif

  #if defined USE_TOF_SENSORS
  //printToFReadings();      //Prints ToF sensor distances and altitude hold status (expected: 0 to 8190mm)
  #endif

  #if defined USE_LITEWING_MODULE
  // ---- LiteWing debug prints — uncomment ONE at a time ----
  // STEP 1 — Verify sensors before enabling any hold:
  //   ToF should match physical height; motion=0xB0 means valid flow; shutter<200 = good lighting
  //printLitewingSensors();
  //
  // STEP 2 — Verify EKF states (hover without hold enabled):
  //   Z should track ToF reading; XY near 0; P_Z should shrink after ToF updates
  //printLitewingEKF();
  //
  // STEP 3 — Tune altitude hold (enable althold via UI):
  //   z_err should go to 0; thr_corr should settle near 0; if oscillating reduce pidVZ_kp
  //printLitewingAltHold();
  //
  // STEP 4 — Tune position hold (enable poshold via UI, after althold is solid):
  //   x_err/y_err should go to 0; if drone drifts one way check dpx/dpy axis swap
  //printLitewingPosHold();
  //
  // General overview (position + velocity + corrections):
  //printLitewingData();
  #endif

  //Regulate loop rate (set via UI, default 2000Hz)
  loopRate(loop_rate_hz);
}

//========================================================================================================================//
//                                              ESP32-SPECIFIC FUNCTIONS                                                 //                           
//========================================================================================================================//

void setupMotorPWM() {
  //DESCRIPTION: Configure ESP32 LEDC PWM for motor control (4 motors for quadcopter)
  /*
   * ESP32 uses LEDC (LED Control) peripheral for PWM generation.
   * Configured 4 channels (one for each motor) with:
   * - 20kHz frequency (suitable for brushed motors)
   * - 8-bit resolution (0-255 duty cycle range)
   */

  //Configure LEDC timer (20kHz, 8-bit resolution)
  ledcSetup(0, 20000, 8); //Channel 0: m1 (Front Right)
  ledcSetup(1, 20000, 8); //Channel 1: m2 (Back Right)
  ledcSetup(2, 20000, 8); //Channel 2: m3 (Back Left)
  ledcSetup(3, 20000, 8); //Channel 3: m4 (Front Left)

  //Attach pins to LEDC channels
  ledcAttachPin(m1Pin, 0);
  ledcAttachPin(m2Pin, 1);
  ledcAttachPin(m3Pin, 2);
  ledcAttachPin(m4Pin, 3);

  Serial.println("Motor PWM channels configured (4 motors, 20kHz, 8-bit)");
}

void setupWiFi() {
  //DESCRIPTION: Initialize WiFi and WebSocket server
  
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  Serial.printf("[WiFi] Free heap before WiFi: %d bytes\n", ESP.getFreeHeap());

  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);

  // Scan to see what networks are visible
  int n = WiFi.scanNetworks();
  Serial.printf("[WiFi] Networks found: %d\n", n);
  for (int i = 0; i < n; i++) {
    Serial.printf("  %d: \"%s\" (%d dBm) %s\n", i + 1, WiFi.SSID(i).c_str(), WiFi.RSSI(i),
                  WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "OPEN" : "SECURED");
  }
  WiFi.scanDelete();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n\nWiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm\n");
  } else {
    Serial.print("\n\nWiFi Connection Failed! Status code: ");
    Serial.println(WiFi.status());
    Serial.println("  1 = SSID not found (wrong name or 5GHz only)");
    Serial.println("  4 = Wrong password");
    Serial.println("  6 = Disconnected (other issue)");
    Serial.println("Flying without WiFi control...\n");
    return;
  }
  
  //Start WebSocket server
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  
  //Start HTTP server
  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  #if defined USE_CAMERA
  server.on("/capture", handleCapture); // Single JPEG snapshot for AI analysis
  #endif
  server.begin();
  
  Serial.println("WebSocket server started on port 81");
  Serial.println("HTTP server started on port 80\n");
}

void broadcastTelemetry() {
  // Send flight data to all connected clients at 10 Hz.
  // Runs on Core 0 (WiFi task context) — never touches the PID loop.
  // Browser buffers this for anomaly detection and AI analysis.
  if (!clientConnected) return;
  unsigned long now = millis();
  if (now - lastTelemetryTime < 100) return;   // 10 Hz cap
  lastTelemetryTime = now;

#if defined USE_LITEWING_MODULE
  char buf[400];
  snprintf(buf, sizeof(buf),
    "{\"tel\":1,\"t\":%lu"
    ",\"r\":%.2f,\"p\":%.2f,\"y\":%.2f"
    ",\"gx\":%.1f,\"gy\":%.1f,\"gz\":%.1f"
    ",\"er\":%.3f,\"ep\":%.3f,\"ey\":%.3f"
    ",\"ch1\":%lu,\"ch5\":%lu"
    ",\"m1\":%d,\"m2\":%d,\"m3\":%d,\"m4\":%d"
    ",\"alt\":%.0f,\"altsp\":%.0f,\"vz\":%.3f"
    ",\"kx\":%.3f,\"ky\":%.3f,\"kvx\":%.3f,\"kvy\":%.3f"
    ",\"althold\":%d,\"poshold\":%d}",
    now,
    roll_IMU, pitch_IMU, yaw_IMU,
    GyroX, GyroY, GyroZ,
    error_roll, error_pitch, error_yaw,
    channel_1_pwm, channel_5_pwm,
    m1_command_PWM, m2_command_PWM, m3_command_PWM, m4_command_PWM,
    kc_S[KC_STATE_Z] * 1000.0f, lw_altitude_setpoint_m * 1000.0f,
    kc_R[2][0]*kc_S[KC_STATE_PX]+kc_R[2][1]*kc_S[KC_STATE_PY]+kc_R[2][2]*kc_S[KC_STATE_PZ],
    kc_S[KC_STATE_X], kc_S[KC_STATE_Y],
    kc_R[0][0]*kc_S[KC_STATE_PX]+kc_R[0][1]*kc_S[KC_STATE_PY]+kc_R[0][2]*kc_S[KC_STATE_PZ],
    kc_R[1][0]*kc_S[KC_STATE_PX]+kc_R[1][1]*kc_S[KC_STATE_PY]+kc_R[1][2]*kc_S[KC_STATE_PZ],
    (int)lw_althold_enabled, (int)lw_poshold_active
  );
#else
  char buf[256];
  snprintf(buf, sizeof(buf),
    "{\"tel\":1,\"t\":%lu"
    ",\"r\":%.2f,\"p\":%.2f,\"y\":%.2f"
    ",\"gx\":%.1f,\"gy\":%.1f,\"gz\":%.1f"
    ",\"er\":%.3f,\"ep\":%.3f,\"ey\":%.3f"
    ",\"ch1\":%lu,\"ch5\":%lu"
    ",\"m1\":%d,\"m2\":%d,\"m3\":%d,\"m4\":%d}",
    now,
    roll_IMU, pitch_IMU, yaw_IMU,
    GyroX, GyroY, GyroZ,
    error_roll, error_pitch, error_yaw,
    channel_1_pwm, channel_5_pwm,
    m1_command_PWM, m2_command_PWM, m3_command_PWM, m4_command_PWM
  );
#endif
  webSocket.broadcastTXT(buf);
}

void handleWiFi() {
  //DESCRIPTION: Handle WiFi WebSocket and HTTP server in main loop
  webSocket.loop();
  server.handleClient();
  broadcastTelemetry();   // 10 Hz telemetry back to browser

  //Check for WiFi timeout failsafe
  if (millis() - lastCommandTime > FAILSAFE_TIMEOUT && clientConnected) {
    Serial.println("WiFi FAILSAFE - No commands received");
    clientConnected = false;
  }
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  //DESCRIPTION: Handle incoming WebSocket messages
  
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      clientConnected = false;
      break;

    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
        clientConnected = true;
        
        String msg = "{\"status\":\"connected\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
        webSocket.sendTXT(num, msg);
      }
      break;

    case WStype_TEXT:
      {
        //Parse JSON command
        StaticJsonDocument<512> doc;  // Increased size for tuning parameters
        DeserializationError error = deserializeJson(doc, payload);

        if (!error) {
          //Update channel values
          if (doc.containsKey("ch1")) channel_1_pwm = constrain((int)doc["ch1"], 1000, 2000);
          if (doc.containsKey("ch2")) channel_2_pwm = constrain((int)doc["ch2"], 1000, 2000);
          if (doc.containsKey("ch3")) channel_3_pwm = constrain((int)doc["ch3"], 1000, 2000);
          if (doc.containsKey("ch4")) channel_4_pwm = constrain((int)doc["ch4"], 1000, 2000);
          if (doc.containsKey("ch5")) channel_5_pwm = constrain((int)doc["ch5"], 1000, 2000);
          if (doc.containsKey("ch6")) channel_6_pwm = constrain((int)doc["ch6"], 1000, 2000);

          #if defined USE_LITEWING_MODULE
          // Enable/disable altitude hold
          // On enable: hover throttle auto-captured from stick on next litewingAltitudeHold() call
          if (doc.containsKey("althold")) {
            bool new_state = (doc["althold"] == 1);
            if (!new_state && lw_althold_enabled) {
              // Disabling: reset all PID state and re-arm auto-capture for next enable
              lw_thrust_correction   = 0.0f;
              lw_pidZ_integral       = 0.0f;
              lw_pidVZ_integral      = 0.0f;
              lw_pidZ_prev_err       = 0.0f;
              lw_hover_thr_captured  = false;
            }
            lw_althold_enabled = new_state;
            Serial.printf("[LiteWing] Altitude hold %s\n", lw_althold_enabled ? "ENABLED" : "DISABLED");
          }

          // Manually override altitude setpoint (metres)
          if (doc.containsKey("altset")) {
            lw_altitude_setpoint_m = constrain((float)doc["altset"], 0.05f, 3.0f);
            lw_pidZ_integral       = 0.0f; // reset outer integral on new setpoint
            Serial.printf("[LiteWing] Altitude setpoint: %.2f m\n", lw_altitude_setpoint_m);
          }

          // Manually override hover throttle (thrustBase equivalent)
          if (doc.containsKey("hover_thr")) {
            lw_hover_throttle     = constrain((float)doc["hover_thr"], 0.2f, 0.9f);
            lw_hover_thr_captured = true;
            Serial.printf("[LiteWing] Hover throttle override: %.2f\n", lw_hover_throttle);
          }

          // Position hold: capture current KF position as setpoint on enable
          if (doc.containsKey("poshold")) {
            bool new_ph = (doc["poshold"] == 1);
            if (new_ph && !lw_poshold_active) {
              // Engage: lock target to current estimated position
              lw_posX_sp = kc_S[KC_STATE_X];
              lw_posY_sp = kc_S[KC_STATE_Y];
              lw_pidX_int = lw_pidY_int = 0.0f;
              lw_pidVX_int = lw_pidVY_int = 0.0f;
              lw_pidX_prev = lw_pidY_prev = 0.0f;
              lw_pidVX_prev = lw_pidVY_prev = 0.0f;
              Serial.printf("[LiteWing] PosHold ON — target x=%.3f y=%.3f\n", lw_posX_sp, lw_posY_sp);
            } else if (!new_ph && lw_poshold_active) {
              lw_roll_correction = lw_pitch_correction = 0.0f;
              Serial.println("[LiteWing] PosHold OFF");
            }
            lw_poshold_active = new_ph;
          }
          // Live-tune XY PID gains
          if (doc.containsKey("pidX_kp"))  lw_pidX_kp  = constrain((float)doc["pidX_kp"],  0.0f, 5.0f);
          if (doc.containsKey("pidX_ki"))  lw_pidX_ki  = constrain((float)doc["pidX_ki"],  0.0f, 2.0f);
          if (doc.containsKey("pidVX_kp")) lw_pidVX_kp = constrain((float)doc["pidVX_kp"], 0.0f, 5.0f);
          if (doc.containsKey("pidVX_ki")) lw_pidVX_ki = constrain((float)doc["pidVX_ki"], 0.0f, 2.0f);
          if (doc.containsKey("pidY_kp"))  lw_pidY_kp  = constrain((float)doc["pidY_kp"],  0.0f, 5.0f);
          if (doc.containsKey("pidY_ki"))  lw_pidY_ki  = constrain((float)doc["pidY_ki"],  0.0f, 2.0f);
          if (doc.containsKey("pidVY_kp")) lw_pidVY_kp = constrain((float)doc["pidVY_kp"], 0.0f, 5.0f);
          if (doc.containsKey("pidVY_ki")) lw_pidVY_ki = constrain((float)doc["pidVY_ki"], 0.0f, 2.0f);
          // Live-tune EKF noise parameters
          if (doc.containsKey("kf_q"))    kc_procNoiseAcc_z  = constrain((float)doc["kf_q"],  0.0f, 10.0f);
          if (doc.containsKey("kf_r"))    kc_tof_stdDev      = constrain((float)doc["kf_r"],  0.001f, 1.0f);
          if (doc.containsKey("flow_std"))kc_flow_stdDev     = constrain((float)doc["flow_std"], 0.1f, 20.0f);
          if (doc.containsKey("pidZ_kp")) lw_pidZ_kp    = constrain((float)doc["pidZ_kp"], 0.0f,  5.0f);
          if (doc.containsKey("pidZ_ki")) lw_pidZ_ki    = constrain((float)doc["pidZ_ki"], 0.0f,  2.0f);
          if (doc.containsKey("pidVZ_kp"))lw_pidVZ_kp   = constrain((float)doc["pidVZ_kp"],0.0f,  5.0f);
          if (doc.containsKey("pidVZ_ki"))lw_pidVZ_ki   = constrain((float)doc["pidVZ_ki"],0.0f,  2.0f);
          #endif

          #if defined USE_TOF_SENSORS
          //Update altitude setpoint
          if (doc.containsKey("altitude")) {
            float alt = constrain((float)doc["altitude"], 0.2, 3.0);
            altitude_setpoint = alt * 1000.0; //Convert to mm
            altitude_error_integral = 0.0; //Reset integral
          }

          //Update feature flags
          if (doc.containsKey("althold")) {
            altitude_hold_enabled = (doc["althold"] == 1);
          }
          if (doc.containsKey("obstavoid")) {
            obstacle_avoid_enabled = (doc["obstavoid"] == 1);
          }
          #endif

          #if defined USE_MTF01_OPTICAL_FLOW
          //Update altitude setpoint (MTF-01)
          if (doc.containsKey("altitude")) {
            float alt = constrain((float)doc["altitude"], 0.2, 3.0);
            altitude_setpoint = alt * 1000.0; //Convert to mm
            altitude_error_integral = 0.0; //Reset integral
          }

          //Enable/disable altitude hold
          if (doc.containsKey("althold")) {
            altitude_hold_enabled = (doc["althold"] == 1);
          }

          //Enable/disable position hold
          if (doc.containsKey("poshold")) {
            position_hold_enabled = (doc["poshold"] == 1);
            if (position_hold_enabled) {
              // Reset position to current location when enabling
              position_x_setpoint = position_x;
              position_y_setpoint = position_y;
              pos_x_error_integral = 0.0;
              pos_y_error_integral = 0.0;
            }
          }

          //Reset position to origin
          if (doc.containsKey("resetpos")) {
            if (doc["resetpos"] == 1) {
              position_x = 0.0;
              position_y = 0.0;
              position_x_setpoint = 0.0;
              position_y_setpoint = 0.0;
              Serial.println("Position reset to origin");
            }
          }

          //Landing command
          if (doc.containsKey("land")) {
            if (doc["land"] == 1) {
              startAutoLanding();
            }
          }
          #endif

          //Live tuning parameters
          if (doc.containsKey("tuning")) {
            Serial.println("Receiving tuning parameters...");

            //Motor compensation values
            if (doc.containsKey("m1_comp")) {
              float m1 = constrain((float)doc["m1_comp"], 0.8, 1.05);
              motor_comp_m1 = m1;
              Serial.print("M1 comp: "); Serial.println(m1);
            }
            if (doc.containsKey("m2_comp")) {
              float m2 = constrain((float)doc["m2_comp"], 0.8, 1.05);
              motor_comp_m2 = m2;
              Serial.print("M2 comp: "); Serial.println(m2);
            }
            if (doc.containsKey("m3_comp")) {
              float m3 = constrain((float)doc["m3_comp"], 0.8, 1.05);
              motor_comp_m3 = m3;
              Serial.print("M3 comp: "); Serial.println(m3);
            }
            if (doc.containsKey("m4_comp")) {
              float m4 = constrain((float)doc["m4_comp"], 0.8, 1.05);
              motor_comp_m4 = m4;
              Serial.print("M4 comp: "); Serial.println(m4);
            }

            //Duty idle (minimum PWM baseline)
            if (doc.containsKey("duty_idle")) {
              int di = constrain((int)doc["duty_idle"], 0, 100);
              DUTY_IDLE = di;
              Serial.print("Duty idle: "); Serial.println(di);
            }

            //Yaw scale
            if (doc.containsKey("yaw_scale")) {
              float ys = constrain((float)doc["yaw_scale"], 0.1, 2.0);
              yaw_scale = ys;
              Serial.print("Yaw scale: "); Serial.println(ys);
            }

            //Loop rate (Hz)
            if (doc.containsKey("loop_rate")) {
              int lr = constrain((int)doc["loop_rate"], 100, 2000);
              loop_rate_hz = lr;
              Serial.print("Loop rate: "); Serial.print(lr); Serial.println(" Hz");
            }

            //Roll angle PID
            if (doc.containsKey("roll_angle_kp")) Kp_roll_angle = (float)doc["roll_angle_kp"];
            if (doc.containsKey("roll_angle_ki")) Ki_roll_angle = (float)doc["roll_angle_ki"];
            if (doc.containsKey("roll_angle_kd")) Kd_roll_angle = (float)doc["roll_angle_kd"];

            //Roll rate PID
            if (doc.containsKey("roll_rate_kp")) Kp_roll_rate = (float)doc["roll_rate_kp"];
            if (doc.containsKey("roll_rate_ki")) Ki_roll_rate = (float)doc["roll_rate_ki"];
            if (doc.containsKey("roll_rate_kd")) Kd_roll_rate = (float)doc["roll_rate_kd"];

            //Pitch angle PID
            if (doc.containsKey("pitch_angle_kp")) Kp_pitch_angle = (float)doc["pitch_angle_kp"];
            if (doc.containsKey("pitch_angle_ki")) Ki_pitch_angle = (float)doc["pitch_angle_ki"];
            if (doc.containsKey("pitch_angle_kd")) Kd_pitch_angle = (float)doc["pitch_angle_kd"];

            //Pitch rate PID
            if (doc.containsKey("pitch_rate_kp")) Kp_pitch_rate = (float)doc["pitch_rate_kp"];
            if (doc.containsKey("pitch_rate_ki")) Ki_pitch_rate = (float)doc["pitch_rate_ki"];
            if (doc.containsKey("pitch_rate_kd")) Kd_pitch_rate = (float)doc["pitch_rate_kd"];

            //Yaw rate PID
            if (doc.containsKey("yaw_rate_kp")) Kp_yaw = (float)doc["yaw_rate_kp"];
            if (doc.containsKey("yaw_rate_ki")) Ki_yaw = (float)doc["yaw_rate_ki"];
            if (doc.containsKey("yaw_rate_kd")) Kd_yaw = (float)doc["yaw_rate_kd"];

            Serial.println("Tuning parameters updated!");
          }

          lastCommandTime = millis(); //Reset failsafe timer
        }
      }
      break;
  }
}

//========================================================================================================================//
//                                              CAMERA FUNCTIONS                                                        //
//========================================================================================================================//

#if defined USE_CAMERA

void setupCamera() {
  camera_config_t config;
  // LEDC fields not used on ESP32-S3 (uses dedicated camera clock) — safe values only
  config.ledc_channel = LEDC_CHANNEL_4;
  config.ledc_timer   = LEDC_TIMER_1;
  config.pin_d0    = CAM_PIN_D0;
  config.pin_d1    = CAM_PIN_D1;
  config.pin_d2    = CAM_PIN_D2;
  config.pin_d3    = CAM_PIN_D3;
  config.pin_d4    = CAM_PIN_D4;
  config.pin_d5    = CAM_PIN_D5;
  config.pin_d6    = CAM_PIN_D6;
  config.pin_d7    = CAM_PIN_D7;
  config.pin_xclk  = CAM_PIN_XCLK;
  config.pin_pclk  = CAM_PIN_PCLK;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href  = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn  = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;
  config.xclk_freq_hz = 20000000;       // 20 MHz XCLK
  config.pixel_format = PIXFORMAT_JPEG; // Hardware JPEG encoder
  config.frame_size   = FRAMESIZE_QVGA; // 320×240 — good WiFi/AI balance
  config.jpeg_quality = 15;             // 0–63, lower = better quality
  config.fb_count     = 2;             // Double-buffer for smoother capture
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Init failed: 0x%x\n", err);
    cameraReady = false;
    return;
  }
  cameraReady = true;
  Serial.println("[CAM] OV2640 ready — 320×240 JPEG");
}

// /capture — single JPEG snapshot (called from port-80 HTTP server)
void handleCapture() {
  if (!cameraReady) { server.send(503, "text/plain", "Camera not ready"); return; }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)          { server.send(503, "text/plain", "Frame capture failed"); return; }
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// MJPEG stream task — pinned to Core 0, never touches the flight loop on Core 1
void cameraStreamTask(void *pvParameters) {
  streamServer.begin();
  Serial.println("[CAM] MJPEG stream server on port 82");
  for (;;) {
    WiFiClient client = streamServer.available();
    if (client) {
      client.print(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache\r\n\r\n"
      );
      while (client.connected() && cameraReady) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
          client.printf(
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
            fb->len
          );
          client.write(fb->buf, fb->len);
          client.print("\r\n");
          esp_camera_fb_return(fb);
        }
        vTaskDelay(pdMS_TO_TICKS(100)); // ~10 fps — leaves CPU for WiFi stack
      }
      client.stop();
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

#endif // USE_CAMERA

void handleRoot() {
  //DESCRIPTION: Serve status page via HTTP
  String html = "<!DOCTYPE html><html><head><title>ESP32-S3 Drone</title></head><body>";
  html += "<h1>ESP32-S3 Flight Controller</h1>";
  html += "<p>Status: " + String(clientConnected ? "Connected" : "Waiting") + "</p>";
  html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
  html += "<p>Signal: " + String(WiFi.RSSI()) + " dBm</p>";
  html += "<h3>Channels:</h3><ul>";
  html += "<li>CH1 (Throttle): " + String(channel_1_pwm) + "</li>";
  html += "<li>CH2 (Roll): " + String(channel_2_pwm) + "</li>";
  html += "<li>CH3 (Pitch): " + String(channel_3_pwm) + "</li>";
  html += "<li>CH4 (Yaw): " + String(channel_4_pwm) + "</li>";
  html += "<li>CH5 (ARM): " + String(channel_5_pwm) + " (" + String(channel_5_pwm > 1500 ? "DISARMED" : "ARMED") + ")</li>";
  html += "</ul></body></html>";
  server.send(200, "text/html", html);
}

void handleStatus() {
  //DESCRIPTION: JSON status endpoint
  String json = "{";
  json += "\"connected\":" + String(clientConnected ? "true" : "false") + ",";
  json += "\"ch1\":" + String(channel_1_pwm) + ",";
  json += "\"ch2\":" + String(channel_2_pwm) + ",";
  json += "\"ch3\":" + String(channel_3_pwm) + ",";
  json += "\"ch4\":" + String(channel_4_pwm) + ",";
  json += "\"ch5\":" + String(channel_5_pwm);
  json += "}";
  server.send(200, "application/json", json);
}


//========================================================================================================================//
//                                              TOF SENSOR FUNCTIONS                                                     //                           
//========================================================================================================================//

#if defined USE_TOF_SENSORS

void tcaSelect(uint8_t channel) {
  //DESCRIPTION: Select TCA9548A multiplexer channel
  if (channel > 7) return;
  Wire.beginTransmission(TCA9548A_ADDRESS);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void tofSetup() {
  //DESCRIPTION: Initialize all VL53L0X sensors through multiplexer
  
  Serial.println("Initializing ToF sensors...");
  delay(100);

  //Initialize each sensor (skip left and back - known to be problematic)
  
  //Channel 0 - Right sensor
  tcaSelect(0);
  if (!tof_right.init()) {
    Serial.println("Failed to init ToF Right (SD0)");
  } else {
    tof_right.setTimeout(500);
    tof_right.startContinuous(50);
    Serial.println("ToF Right initialized");
  }
  delay(50);

  //Channel 1 - Front sensor
  tcaSelect(1);
  if (!tof_front.init()) {
    Serial.println("Failed to init ToF Front (SD1)");
  } else {
    tof_front.setTimeout(500);
    tof_front.startContinuous(50);
    Serial.println("ToF Front initialized");
  }
  delay(50);

  //Channel 4 - Top sensor
  tcaSelect(4);
  if (!tof_top.init()) {
    Serial.println("Failed to init ToF Top (SD4)");
  } else {
    tof_top.setTimeout(500);
    tof_top.startContinuous(50);
    Serial.println("ToF Top initialized");
  }
  delay(50);

  //Channel 5 - Bottom sensor (altitude)
  tcaSelect(5);
  if (!tof_bottom.init()) {
    Serial.println("Failed to init ToF Bottom (SD5)");
  } else {
    tof_bottom.setTimeout(500);
    tof_bottom.startContinuous(50);
    Serial.println("ToF Bottom initialized");
  }
  delay(50);

  Serial.println("ToF sensors initialized!");
}

void updateToFSensors() {
  //DESCRIPTION: Read all ToF sensors through multiplexer
  
  if (millis() - tof_update_time < TOF_UPDATE_INTERVAL) {
    return;
  }
  tof_update_time = millis();

  //Read right sensor
  tcaSelect(0);
  distance_right = tof_right.readRangeContinuousMillimeters();
  if (tof_right.timeoutOccurred()) distance_right = 8190;

  //Read front sensor
  tcaSelect(1);
  distance_front = tof_front.readRangeContinuousMillimeters();
  if (tof_front.timeoutOccurred()) distance_front = 8190;

  //Left and back sensors disabled
  distance_left = 8190;
  distance_back = 8190;

  //Read top sensor
  tcaSelect(4);
  distance_top = tof_top.readRangeContinuousMillimeters();
  if (tof_top.timeoutOccurred()) distance_top = 8190;

  //Read bottom sensor (altitude)
  tcaSelect(5);
  distance_bottom = tof_bottom.readRangeContinuousMillimeters();
  if (tof_bottom.timeoutOccurred()) distance_bottom = 8190;

  //Update current altitude (low-pass filter)
  static float altitude_filtered = 0.0;
  if (distance_bottom < 8000) {
    altitude_filtered = 0.7 * altitude_filtered + 0.3 * distance_bottom;
    altitude_current = altitude_filtered;
  }

  //Check for obstacles
  obstacle_right = (distance_right < OBSTACLE_THRESHOLD);
  obstacle_front = (distance_front < OBSTACLE_THRESHOLD);
  obstacle_left = (distance_left < OBSTACLE_THRESHOLD);
  obstacle_back = (distance_back < OBSTACLE_THRESHOLD);
  obstacle_top = (distance_top < OBSTACLE_THRESHOLD);
}

void applyAltitudeHold(float dt) {
  //DESCRIPTION: PID controller for altitude hold using bottom ToF sensor
  
  if (!altitude_hold_enabled) {
    altitude_correction = 0.0;
    altitude_error_integral = 0.0;
    return;
  }

  //Only apply if valid reading
  if (distance_bottom > 8000 || distance_bottom < 50) {
    altitude_correction = 0.0;
    altitude_error_integral = 0.0;
    return;
  }

  //Only apply when airborne (above 15cm)
  if (altitude_current < 150.0) {
    altitude_correction = 0.0;
    altitude_error_integral = 0.0;
    return;
  }

  //Calculate error
  altitude_error = altitude_setpoint - altitude_current;

  //Integral term with anti-windup
  altitude_error_integral += altitude_error * dt;
  altitude_error_integral = constrain(altitude_error_integral, -100.0, 100.0);

  //Derivative term
  float altitude_error_derivative = (altitude_error - altitude_error_previous) / dt;
  altitude_error_previous = altitude_error;

  //PID output
  altitude_correction = Kp_altitude * altitude_error +
                        Ki_altitude * altitude_error_integral +
                        Kd_altitude * altitude_error_derivative;

  //Limit correction
  altitude_correction = constrain(altitude_correction, -0.3, 0.3);
}

void applyObstacleAvoidance(float &roll_cmd, float &pitch_cmd, float &throttle_cmd) {
  //DESCRIPTION: Apply obstacle avoidance corrections
  
  if (!obstacle_avoid_enabled) return;

  const float AVOID_STRENGTH = 0.3;

  if (obstacle_right) roll_cmd -= AVOID_STRENGTH;
  if (obstacle_left) roll_cmd += AVOID_STRENGTH;
  if (obstacle_front) pitch_cmd -= AVOID_STRENGTH;
  if (obstacle_back) pitch_cmd += AVOID_STRENGTH;
  if (obstacle_top) throttle_cmd -= AVOID_STRENGTH;

  roll_cmd = constrain(roll_cmd, -1.0, 1.0);
  pitch_cmd = constrain(pitch_cmd, -1.0, 1.0);
  throttle_cmd = constrain(throttle_cmd, 0.0, 1.0);
}

#endif //USE_TOF_SENSORS

//========================================================================================================================//
//                                         MTF-01 OPTICAL FLOW SENSOR FUNCTIONS                                          //
//========================================================================================================================//

#if defined USE_MTF01_OPTICAL_FLOW

void mtf01Setup() {
  //DESCRIPTION: Initialize MTF-01 optical flow sensor (UART)

  Serial.println("Initializing MTF-01 Optical Flow sensor (UART)...");

  // Initialize Serial1 for MTF-01 communication
  Serial1.begin(MTF01_BAUD_RATE, SERIAL_8N1, MTF01_RX_PIN, MTF01_TX_PIN);
  delay(100);

  // Clear any existing data in serial buffer
  while (Serial1.available()) {
    Serial1.read();
  }

  Serial.print("MTF-01 UART initialized on pins RX:");
  Serial.print(MTF01_RX_PIN);
  Serial.print(" TX:");
  Serial.print(MTF01_TX_PIN);
  Serial.print(" @ ");
  Serial.print(MTF01_BAUD_RATE);
  Serial.println(" baud");

  // Reset position to origin
  position_x = 0.0;
  position_y = 0.0;
  position_x_setpoint = 0.0;
  position_y_setpoint = 0.0;

  Serial.println("MTF-01 initialized! Waiting for data packets...");
}

void readMTF01Data() {
  // DESCRIPTION: Read optical flow and rangefinder data from MTF-01 via UART (Micolink protocol)
  /*
   * Micolink Packet Format:
   * Byte 0: Header 0xEF
   * Byte 1: Device ID (u8)
   * Byte 2: System ID (u8)
   * Byte 3: Message ID (u8) - 0x51 for RANGE_SENSOR
   * Byte 4: Sequence (u8)
   * Byte 5: Length (u8) - payload length
   * Bytes 6 to 6+len-1: Payload (for 0x51: time_ms u32, distance u32 mm, strength u8, precision u8, dis_status u8, reserved1 u8, flow_vel_x i16 cm/s@1m, flow_vel_y i16, flow_quality u8, flow_status u8, reserved2 u16)
   * Byte 6+len: Checksum (u8) - sum of all previous bytes
   */
  while (Serial1.available() > 0) {
    uint8_t incoming_byte = Serial1.read();

    // Shift buffer left and add new byte
    for (int i = 0; i < 79; i++) {
      mtf01_buffer[i] = mtf01_buffer[i + 1];
    }
    mtf01_buffer[79] = incoming_byte;

    // Look for header in the last few bytes (sliding window)
    for (int start = 0; start < 80; start++) {
      if (mtf01_buffer[start] == 0xEF) {  // Header found
        uint8_t msg_id = mtf01_buffer[start + 3];
        if (msg_id != 0x51) continue;  // Only handle RANGE_SENSOR

        uint8_t len = mtf01_buffer[start + 5];
        if (len != 20) continue;  // Expected payload size for 0x51 (uint32*2 + u8*4 + i16*2 + u8*2 + u16*1 = 20 bytes)

        int packet_end = start + 5 + len + 1;  // Header to checksum
        if (packet_end >= 80) continue;  // Incomplete packet

        // Verify checksum (sum of bytes from header to before checksum)
        uint8_t calc_checksum = 0;
        for (int i = start; i < start + 6 + len; i++) {  // Up to last payload byte
          calc_checksum += mtf01_buffer[i];
        }
        uint8_t rx_checksum = mtf01_buffer[start + 6 + len];
        if (calc_checksum != rx_checksum) continue;  // Invalid

        // Parse payload (little-endian assumed)
        int payload_start = start + 6;
        // Skip time_ms (u32): bytes [payload_start+0 to 3]
        range_mm = (uint32_t)mtf01_buffer[payload_start + 7] << 24 |
                   (uint32_t)mtf01_buffer[payload_start + 6] << 16 |
                   (uint32_t)mtf01_buffer[payload_start + 5] << 8  |
                   (uint32_t)mtf01_buffer[payload_start + 4];  // distance u32 mm
        // Skip strength, precision, dis_status, reserved1 (u8*4): [8-11]
        flow_x = (int16_t)((mtf01_buffer[payload_start + 13] << 8) | mtf01_buffer[payload_start + 12]);  // flow_vel_x i16
        flow_y = (int16_t)((mtf01_buffer[payload_start + 15] << 8) | mtf01_buffer[payload_start + 14]);  // flow_vel_y i16
        flow_quality = mtf01_buffer[payload_start + 16];  // u8
        // Skip flow_status (u8), reserved2 (u16): [17-19]

        // Update altitude with low-pass filter (same as before)
        static float altitude_filtered = 0.0;
        if (range_mm < 8000 && range_mm > 50) {  // Valid range: 5cm to 8m (sensor spec)
          altitude_filtered = 0.7 * altitude_filtered + 0.3 * range_mm;
          altitude_current = altitude_filtered;
        } else {
          // Invalid - reset to avoid bad data
          altitude_current = 0.0;
        }

        // Debug print (comment out after confirming it works)
        // Serial.printf("MTF01: FlowX=%d cm/s FlowY=%d cm/s Quality=%d Range=%u mm\n", flow_x, flow_y, flow_quality, range_mm);

        break;  // Processed one packet
      }
    }
  }
}

void updatePositionEstimate(float dt) {
  //DESCRIPTION: Integrate optical flow to estimate position
  /*
   * Flow is now velocity in cm/s at 1m height, so scale by actual height
   * Then integrate velocity to get position
   */

  // Only integrate if flow quality is good and altitude is valid
  if (flow_quality < 100 || altitude_current < 50 || altitude_current > 8000) {
    return;
  }

  // Velocity = reported_vel * (altitude / 1000.0)  // cm/s, scaled by height in mm to m
  float vel_x = flow_x * (altitude_current / 1000.0);
  float vel_y = flow_y * (altitude_current / 1000.0);

  // Integrate to position (cm)
  position_x += vel_x * dt;
  position_y += vel_y * dt;

  // Constrain to prevent drift
  position_x = constrain(position_x, -200.0, 200.0);
  position_y = constrain(position_y, -200.0, 200.0);
}

void applyPositionHold(float dt, float &roll_correction, float &pitch_correction) {
  //DESCRIPTION: PID controller for position hold using optical flow

  if (!position_hold_enabled) {
    position_x_correction = 0.0;
    position_y_correction = 0.0;
    pos_x_error_integral = 0.0;
    pos_y_error_integral = 0.0;
    return;
  }

  // Only apply if flow quality is good
  if (flow_quality < 100) {
    return;
  }

  // X-axis position PID
  pos_x_error = position_x_setpoint - position_x;
  pos_x_error_integral += pos_x_error * dt;
  pos_x_error_integral = constrain(pos_x_error_integral, -50.0, 50.0);
  float pos_x_error_derivative = (pos_x_error - pos_x_error_previous) / dt;
  pos_x_error_previous = pos_x_error;

  position_x_correction = Kp_position * pos_x_error +
                          Ki_position * pos_x_error_integral +
                          Kd_position * pos_x_error_derivative;

  // Y-axis position PID
  pos_y_error = position_y_setpoint - position_y;
  pos_y_error_integral += pos_y_error * dt;
  pos_y_error_integral = constrain(pos_y_error_integral, -50.0, 50.0);
  float pos_y_error_derivative = (pos_y_error - pos_y_error_previous) / dt;
  pos_y_error_previous = pos_y_error;

  position_y_correction = Kp_position * pos_y_error +
                          Ki_position * pos_y_error_integral +
                          Kd_position * pos_y_error_derivative;

  // Convert position corrections to roll/pitch corrections
  // X position -> Roll, Y position -> Pitch
  roll_correction = -position_x_correction * 0.1;    // Negative for correct direction
  pitch_correction = -position_y_correction * 0.1;   // Negative for correct direction

  // Limit corrections
  roll_correction = constrain(roll_correction, -0.3, 0.3);
  pitch_correction = constrain(pitch_correction, -0.3, 0.3);
}

void applyAltitudeHoldMTF01(float dt) {
  //DESCRIPTION: PID controller for altitude hold using MTF-01 rangefinder

  if (!altitude_hold_enabled) {
    altitude_correction = 0.0;
    altitude_error_integral = 0.0;
    return;
  }

  // Only apply if valid reading
  if (range_mm > 3000 || range_mm < 50) {
    altitude_correction = 0.0;
    altitude_error_integral = 0.0;
    return;
  }

  // Only apply when airborne (above 15cm)
  if (altitude_current < 150.0) {
    altitude_correction = 0.0;
    altitude_error_integral = 0.0;
    return;
  }

  // Calculate error
  altitude_error = altitude_setpoint - altitude_current;

  // Integral term with anti-windup
  altitude_error_integral += altitude_error * dt;
  altitude_error_integral = constrain(altitude_error_integral, -100.0, 100.0);

  // Derivative term
  float altitude_error_derivative = (altitude_error - altitude_error_previous) / dt;
  altitude_error_previous = altitude_error;

  // PID output
  altitude_correction = Kp_altitude * altitude_error +
                        Ki_altitude * altitude_error_integral +
                        Kd_altitude * altitude_error_derivative;

  // Limit correction
  altitude_correction = constrain(altitude_correction, -0.3, 0.3);
}

void startAutoLanding() {
  //DESCRIPTION: Initiate automatic landing sequence using MTF-01 rangefinder
  /*
   * Starts a controlled descent using the rangefinder
   * Descends at controlled rate until reaching ground threshold
   */

  if (range_mm > 3000 || range_mm < 50) {
    // Invalid reading - cannot land safely
    Serial.println("Landing aborted - invalid altitude reading");
    return;
  }

  landing_active = true;
  landing_start_time = millis();
  landing_initial_altitude = altitude_current;

  // Enable altitude hold for smooth descent
  altitude_hold_enabled = true;

  // Reset altitude PID integral term for smooth transition
  altitude_error_integral = 0.0;

  Serial.print("Auto-landing initiated from altitude: ");
  Serial.print(landing_initial_altitude / 1000.0, 2);
  Serial.println(" m");
}

void stopAutoLanding() {
  //DESCRIPTION: Cancel automatic landing
  landing_active = false;
  Serial.println("Auto-landing cancelled");
}

bool isLanding() {
  //DESCRIPTION: Check if auto-landing is currently active
  return landing_active;
}

bool hasLanded() {
  //DESCRIPTION: Check if drone has touched down
  /*
   * Returns true when altitude < 10cm (landing threshold)
   */
  return (altitude_current < LANDING_GROUND_THRESHOLD && altitude_current > 10.0);
}

void applyAutoLanding(float dt) {
  //DESCRIPTION: Execute smooth controlled descent during auto-landing
  /*
   * Gradually reduces altitude setpoint for smooth landing
   * Descent rate: 30cm/second (adjustable via LANDING_DESCENT_RATE)
   */

  if (!landing_active) {
    return;
  }

  // Check if we've landed
  if (hasLanded()) {
    landing_active = false;
    altitude_setpoint = altitude_current; // Hold current position (near ground)
    Serial.println("Landing complete - touchdown detected");
    return;
  }

  // Calculate time elapsed since landing started
  unsigned long elapsed_ms = millis() - landing_start_time;
  float elapsed_sec = elapsed_ms / 1000.0;

  // Calculate target altitude based on descent rate
  float target_altitude = landing_initial_altitude - (LANDING_DESCENT_RATE * elapsed_sec);

  // Don't go below ground threshold
  if (target_altitude < LANDING_GROUND_THRESHOLD) {
    target_altitude = LANDING_GROUND_THRESHOLD;
  }

  // Update altitude setpoint for smooth descent
  altitude_setpoint = target_altitude;

  // Altitude hold will handle the actual throttle control
  applyAltitudeHoldMTF01(dt);

  // Debug output (optional - comment out for production)
  // Serial.print("Landing: Target Alt = ");
  // Serial.print(target_altitude / 1000.0, 2);
  // Serial.print(" m, Current Alt = ");
  // Serial.print(altitude_current / 1000.0, 2);
  // Serial.println(" m");
}

#endif //USE_MTF01_OPTICAL_FLOW

//========================================================================================================================//
//                                    LITEWING MODULE FUNCTIONS  (Kalman + 2-stage PID)                                  //
//  Mirrors: kalman_core.c  estimator_kalman.c  range.c  position_controller_pid.c                                       //
//========================================================================================================================//

#if defined USE_LITEWING_MODULE

void litewingSetup() {
  // --- VL53L1X altitude sensor (I2C) ---
  Serial.println("[LiteWing] Initializing VL53L1X altitude sensor...");
  lw_tof.setTimeout(500);
  if (!lw_tof.init()) {
    Serial.println("[LiteWing] VL53L1X init FAILED - check I2C (SDA=5, SCL=6, addr=0x29)");
    lw_tof_ready = false;
  } else {
    lw_tof.setDistanceMode(VL53L1X::Long);
    lw_tof.setMeasurementTimingBudget(20000);
    lw_tof.startContinuous(LW_TOF_INTERVAL_MS);
    lw_tof_ready = true;
    Serial.println("[LiteWing] VL53L1X ready — Long range, 50 Hz");
  }

  // --- PMW3901 optical flow sensor (SPI) ---
  Serial.println("[LiteWing] Initializing PMW3901 optical flow...");
  SPI.begin();  // SCK=GPIO7, MISO=GPIO8, MOSI=GPIO9
  pmw3901_ready = pmw3901Init();
  if (pmw3901_ready) {
    Serial.println("[LiteWing] PMW3901 ready — optical flow active");
  } else {
    Serial.println("[LiteWing] PMW3901 init FAILED - check SPI (CS=GPIO44)");
  }

  // Initialize 9-state EKF (bootstraps quaternion from Madgwick)
  ekfInit();
}

// =========================================================================
// 9-STATE EKF  (exact port of kalman_core.c — Hamer/Richardsson, Bitcraze)
// =========================================================================

// ---- Helper: 9×9 matrix multiply C = A * B ----
// __attribute__ O3: ensure FPU pipeline is fully utilised on Xtensa LX7
__attribute__((optimize("O3")))
static void mat9Mult(float C[KC_STATE_DIM][KC_STATE_DIM],
                     const float A[KC_STATE_DIM][KC_STATE_DIM],
                     const float B[KC_STATE_DIM][KC_STATE_DIM]) {
  for (int i = 0; i < KC_STATE_DIM; i++)
    for (int j = 0; j < KC_STATE_DIM; j++) {
      float s = 0;
      for (int k = 0; k < KC_STATE_DIM; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
}

// ---- Generic scalar measurement update  (mirrors scalarUpdate in kalman_core.c) ----
// h[]   : 1×9 measurement Jacobian row vector
// error : (measured - predicted)
// std   : measurement standard deviation
static void ekfScalarUpdate(float h[KC_STATE_DIM], float error, float std) {
  // PHT[i] = (P * h^T)[i]
  float PHT[KC_STATE_DIM];
  for (int i = 0; i < KC_STATE_DIM; i++) {
    PHT[i] = 0;
    for (int j = 0; j < KC_STATE_DIM; j++) PHT[i] += kc_P[i][j] * h[j];
  }
  // Innovation covariance HPHR = H P H^T + R
  float R    = std * std;
  float HPHR = R;
  for (int i = 0; i < KC_STATE_DIM; i++) HPHR += h[i] * PHT[i];
  if (HPHR < 1e-10f) return;

  // Kalman gain K = PHT / HPHR; state update
  float K[KC_STATE_DIM];
  for (int i = 0; i < KC_STATE_DIM; i++) {
    K[i] = PHT[i] / HPHR;
    kc_S[i] += K[i] * error;
  }

  // Covariance update (Joseph form, P symmetric):
  // P_new[i][j] = P[i][j] - K[i]*PHT[j] - K[j]*PHT[i] + K[i]*K[j]*HPHR
  for (int i = 0; i < KC_STATE_DIM; i++) {
    for (int j = i; j < KC_STATE_DIM; j++) {
      float p = 0.5f*(kc_P[i][j] + kc_P[j][i])
               - K[i]*PHT[j] - K[j]*PHT[i]
               + K[i]*K[j]*HPHR;
      if (isnan(p) || p > 100.0f)       kc_P[i][j] = kc_P[j][i] = 100.0f;
      else if (i == j && p < 1e-6f)     kc_P[i][j] = kc_P[j][i] = 1e-6f;
      else                               kc_P[i][j] = kc_P[j][i] = p;
    }
  }
}

// ---- EKF init: bootstrap quaternion from Madgwick, reset states ----
// (mirrors kalmanCoreInit)
void ekfInit() {
  memset(kc_S, 0, sizeof(kc_S));
  // Bootstrap quaternion from Madgwick's converged attitude
  kc_q[0] = q0; kc_q[1] = q1; kc_q[2] = q2; kc_q[3] = q3;
  // Build rotation matrix from quaternion (body→world, same as kalman_core.c)
  kc_R[0][0] = kc_q[0]*kc_q[0]+kc_q[1]*kc_q[1]-kc_q[2]*kc_q[2]-kc_q[3]*kc_q[3];
  kc_R[0][1] = 2.0f*(kc_q[1]*kc_q[2]-kc_q[0]*kc_q[3]);
  kc_R[0][2] = 2.0f*(kc_q[1]*kc_q[3]+kc_q[0]*kc_q[2]);
  kc_R[1][0] = 2.0f*(kc_q[1]*kc_q[2]+kc_q[0]*kc_q[3]);
  kc_R[1][1] = kc_q[0]*kc_q[0]-kc_q[1]*kc_q[1]+kc_q[2]*kc_q[2]-kc_q[3]*kc_q[3];
  kc_R[1][2] = 2.0f*(kc_q[2]*kc_q[3]-kc_q[0]*kc_q[1]);
  kc_R[2][0] = 2.0f*(kc_q[1]*kc_q[3]-kc_q[0]*kc_q[2]);
  kc_R[2][1] = 2.0f*(kc_q[2]*kc_q[3]+kc_q[0]*kc_q[1]);
  kc_R[2][2] = kc_q[0]*kc_q[0]-kc_q[1]*kc_q[1]-kc_q[2]*kc_q[2]+kc_q[3]*kc_q[3];
  // Initial covariance (from kalmanCoreInit stdDev values)
  memset(kc_P, 0, sizeof(kc_P));
  kc_P[KC_STATE_X][KC_STATE_X]   = 100.0f*100.0f; // stdDevInitialPosition_xy=100
  kc_P[KC_STATE_Y][KC_STATE_Y]   = 100.0f*100.0f;
  kc_P[KC_STATE_Z][KC_STATE_Z]   = 1.0f*1.0f;     // stdDevInitialPosition_z=1
  kc_P[KC_STATE_PX][KC_STATE_PX] = 0.01f*0.01f;   // stdDevInitialVelocity=0.01
  kc_P[KC_STATE_PY][KC_STATE_PY] = 0.01f*0.01f;
  kc_P[KC_STATE_PZ][KC_STATE_PZ] = 0.01f*0.01f;
  kc_P[KC_STATE_D0][KC_STATE_D0] = 0.01f*0.01f;   // stdDevInitialAttitude=0.01
  kc_P[KC_STATE_D1][KC_STATE_D1] = 0.01f*0.01f;
  kc_P[KC_STATE_D2][KC_STATE_D2] = 0.01f*0.01f;
  kc_initialized = true;
  Serial.println("[EKF] 9-state EKF initialized from Madgwick quaternion");
}

// ---- Prediction step  (mirrors kalmanCorePredict) ----
// gyro_*_dps  : body gyro in deg/s (Maddy convention)
// zacc_ms2    : body-z acceleration in m/s² INCLUDING gravity reaction
//               (Maddy: AccZ in G's gravity-removed → zacc = (AccZ+1.0)*9.81)
// quadIsFlying: flying mode = body-z thrust only; ground mode = full 3-axis acc
//               + ROLLPITCH_ZERO_REVERSION  (mirrors estimator_kalman.c)
void ekfPredict(float dt,
                float gyroX_dps, float gyroY_dps, float gyroZ_dps,
                float zacc_ms2, bool quadIsFlying) {
  if (dt <= 0.0f || dt > 0.1f) return;

  const float DEG2RAD = (float)M_PI / 180.0f;
  const float GRAVITY = 9.81f;
  // Gyro in rad/s (kalman_core.c uses rad/s throughout)
  float gx = gyroX_dps * DEG2RAD;
  float gy = gyroY_dps * DEG2RAD;
  float gz = gyroZ_dps * DEG2RAD;
  float dt2 = dt * dt;
  // Half-angle increments for covariance rotation
  float d0 = gx*dt/2.0f, d1 = gy*dt/2.0f, d2 = gz*dt/2.0f;

  // ====== LINEARIZED DYNAMICS MATRIX A (9×9) — from kalmanCorePredict ======
  static float A[KC_STATE_DIM][KC_STATE_DIM];
  memset(A, 0, sizeof(A));
  for (int i = 0; i < KC_STATE_DIM; i++) A[i][i] = 1.0f; // identity

  // Position update from body-frame velocity (A[X:Z][PX:PZ] = R * dt)
  A[KC_STATE_X][KC_STATE_PX]=kc_R[0][0]*dt; A[KC_STATE_Y][KC_STATE_PX]=kc_R[1][0]*dt; A[KC_STATE_Z][KC_STATE_PX]=kc_R[2][0]*dt;
  A[KC_STATE_X][KC_STATE_PY]=kc_R[0][1]*dt; A[KC_STATE_Y][KC_STATE_PY]=kc_R[1][1]*dt; A[KC_STATE_Z][KC_STATE_PY]=kc_R[2][1]*dt;
  A[KC_STATE_X][KC_STATE_PZ]=kc_R[0][2]*dt; A[KC_STATE_Y][KC_STATE_PZ]=kc_R[1][2]*dt; A[KC_STATE_Z][KC_STATE_PZ]=kc_R[2][2]*dt;

  // Position update from attitude error (body-frame vel cross R columns)
  float px=kc_S[KC_STATE_PX], py=kc_S[KC_STATE_PY], pz=kc_S[KC_STATE_PZ];
  A[KC_STATE_X][KC_STATE_D0]=(py*kc_R[0][2]-pz*kc_R[0][1])*dt;
  A[KC_STATE_Y][KC_STATE_D0]=(py*kc_R[1][2]-pz*kc_R[1][1])*dt;
  A[KC_STATE_Z][KC_STATE_D0]=(py*kc_R[2][2]-pz*kc_R[2][1])*dt;
  A[KC_STATE_X][KC_STATE_D1]=(-px*kc_R[0][2]+pz*kc_R[0][0])*dt;
  A[KC_STATE_Y][KC_STATE_D1]=(-px*kc_R[1][2]+pz*kc_R[1][0])*dt;
  A[KC_STATE_Z][KC_STATE_D1]=(-px*kc_R[2][2]+pz*kc_R[2][0])*dt;
  A[KC_STATE_X][KC_STATE_D2]=(px*kc_R[0][1]-py*kc_R[0][0])*dt;
  A[KC_STATE_Y][KC_STATE_D2]=(px*kc_R[1][1]-py*kc_R[1][0])*dt;
  A[KC_STATE_Z][KC_STATE_D2]=(px*kc_R[2][1]-py*kc_R[2][0])*dt;

  // Body velocity from body velocity (gyro cross-product coupling)
  A[KC_STATE_PY][KC_STATE_PX]=-gz*dt; A[KC_STATE_PZ][KC_STATE_PX]= gy*dt;
  A[KC_STATE_PX][KC_STATE_PY]= gz*dt; A[KC_STATE_PZ][KC_STATE_PY]=-gx*dt;
  A[KC_STATE_PX][KC_STATE_PZ]=-gy*dt; A[KC_STATE_PY][KC_STATE_PZ]= gx*dt;

  // Body velocity from attitude error (gravity coupling)
  A[KC_STATE_PX][KC_STATE_D0]= 0;                      A[KC_STATE_PY][KC_STATE_D0]=-GRAVITY*kc_R[2][2]*dt; A[KC_STATE_PZ][KC_STATE_D0]= GRAVITY*kc_R[2][1]*dt;
  A[KC_STATE_PX][KC_STATE_D1]= GRAVITY*kc_R[2][2]*dt;  A[KC_STATE_PY][KC_STATE_D1]= 0;                     A[KC_STATE_PZ][KC_STATE_D1]=-GRAVITY*kc_R[2][0]*dt;
  A[KC_STATE_PX][KC_STATE_D2]=-GRAVITY*kc_R[2][1]*dt;  A[KC_STATE_PY][KC_STATE_D2]= GRAVITY*kc_R[2][0]*dt; A[KC_STATE_PZ][KC_STATE_D2]= 0;

  // Attitude-error covariance rotation (2nd-order approx — kalman_core.c §Covariance Correction)
  A[KC_STATE_D0][KC_STATE_D0]= 1-d1*d1/2-d2*d2/2; A[KC_STATE_D0][KC_STATE_D1]= d2+d0*d1/2; A[KC_STATE_D0][KC_STATE_D2]=-d1+d0*d2/2;
  A[KC_STATE_D1][KC_STATE_D0]=-d2+d0*d1/2;         A[KC_STATE_D1][KC_STATE_D1]= 1-d0*d0/2-d2*d2/2; A[KC_STATE_D1][KC_STATE_D2]= d0+d1*d2/2;
  A[KC_STATE_D2][KC_STATE_D0]= d1+d0*d2/2;         A[KC_STATE_D2][KC_STATE_D1]=-d0+d1*d2/2; A[KC_STATE_D2][KC_STATE_D2]= 1-d0*d0/2-d1*d1/2;

  // ====== COVARIANCE PROPAGATION  P = A * P * A^T ======
  // Exploit A sparsity: rows X/Y/Z use cols {self,3-8}, rows PX/PY/PZ use cols {3-8},
  // rows D0/D1/D2 use cols {6-8} only. Saves ~45% multiplications vs full mat9Mult.
  static float tmp1[KC_STATE_DIM][KC_STATE_DIM];
  // tmp1 = A * P  (sparse row-by-row)
  for (int j = 0; j < KC_STATE_DIM; j++) {
    // Rows 0-2: identity col + cols 3-8
    tmp1[0][j] = A[0][0]*kc_P[0][j] + A[0][3]*kc_P[3][j]+A[0][4]*kc_P[4][j]+A[0][5]*kc_P[5][j]+A[0][6]*kc_P[6][j]+A[0][7]*kc_P[7][j]+A[0][8]*kc_P[8][j];
    tmp1[1][j] = A[1][1]*kc_P[1][j] + A[1][3]*kc_P[3][j]+A[1][4]*kc_P[4][j]+A[1][5]*kc_P[5][j]+A[1][6]*kc_P[6][j]+A[1][7]*kc_P[7][j]+A[1][8]*kc_P[8][j];
    tmp1[2][j] = A[2][2]*kc_P[2][j] + A[2][3]*kc_P[3][j]+A[2][4]*kc_P[4][j]+A[2][5]*kc_P[5][j]+A[2][6]*kc_P[6][j]+A[2][7]*kc_P[7][j]+A[2][8]*kc_P[8][j];
    // Rows 3-5: cols 3-5 (gyro) + cols 6-8 (gravity), D0/D1/D2 diagonals already in 3-5
    tmp1[3][j] = A[3][3]*kc_P[3][j]+A[3][4]*kc_P[4][j]+A[3][5]*kc_P[5][j]                   +A[3][7]*kc_P[7][j]+A[3][8]*kc_P[8][j];  // A[3][6]=0
    tmp1[4][j] = A[4][3]*kc_P[3][j]+A[4][4]*kc_P[4][j]+A[4][5]*kc_P[5][j]+A[4][6]*kc_P[6][j]                   +A[4][8]*kc_P[8][j];  // A[4][7]=0
    tmp1[5][j] = A[5][3]*kc_P[3][j]+A[5][4]*kc_P[4][j]+A[5][5]*kc_P[5][j]+A[5][6]*kc_P[6][j]+A[5][7]*kc_P[7][j];                    // A[5][8]=0
    // Rows 6-8: only cols 6-8
    tmp1[6][j] = A[6][6]*kc_P[6][j]+A[6][7]*kc_P[7][j]+A[6][8]*kc_P[8][j];
    tmp1[7][j] = A[7][6]*kc_P[6][j]+A[7][7]*kc_P[7][j]+A[7][8]*kc_P[8][j];
    tmp1[8][j] = A[8][6]*kc_P[6][j]+A[8][7]*kc_P[7][j]+A[8][8]*kc_P[8][j];
  }
  // kc_P = tmp1 * A^T  (sparse col-by-col, same sparsity by symmetry)
  for (int i = 0; i < KC_STATE_DIM; i++) {
    kc_P[i][0] = tmp1[i][0]*A[0][0] + tmp1[i][3]*A[0][3]+tmp1[i][4]*A[0][4]+tmp1[i][5]*A[0][5]+tmp1[i][6]*A[0][6]+tmp1[i][7]*A[0][7]+tmp1[i][8]*A[0][8];
    kc_P[i][1] = tmp1[i][1]*A[1][1] + tmp1[i][3]*A[1][3]+tmp1[i][4]*A[1][4]+tmp1[i][5]*A[1][5]+tmp1[i][6]*A[1][6]+tmp1[i][7]*A[1][7]+tmp1[i][8]*A[1][8];
    kc_P[i][2] = tmp1[i][2]*A[2][2] + tmp1[i][3]*A[2][3]+tmp1[i][4]*A[2][4]+tmp1[i][5]*A[2][5]+tmp1[i][6]*A[2][6]+tmp1[i][7]*A[2][7]+tmp1[i][8]*A[2][8];
    kc_P[i][3] = tmp1[i][3]*A[3][3]+tmp1[i][4]*A[3][4]+tmp1[i][5]*A[3][5]                    +tmp1[i][7]*A[3][7]+tmp1[i][8]*A[3][8];  // A[3][6]=0
    kc_P[i][4] = tmp1[i][3]*A[4][3]+tmp1[i][4]*A[4][4]+tmp1[i][5]*A[4][5]+tmp1[i][6]*A[4][6]                    +tmp1[i][8]*A[4][8];  // A[4][7]=0
    kc_P[i][5] = tmp1[i][3]*A[5][3]+tmp1[i][4]*A[5][4]+tmp1[i][5]*A[5][5]+tmp1[i][6]*A[5][6]+tmp1[i][7]*A[5][7];                     // A[5][8]=0
    kc_P[i][6] = tmp1[i][6]*A[6][6]+tmp1[i][7]*A[6][7]+tmp1[i][8]*A[6][8];
    kc_P[i][7] = tmp1[i][6]*A[7][6]+tmp1[i][7]*A[7][7]+tmp1[i][8]*A[7][8];
    kc_P[i][8] = tmp1[i][6]*A[8][6]+tmp1[i][7]*A[8][7]+tmp1[i][8]*A[8][8];
  }

  // ====== STATE PREDICTION ======
  // Position from body-frame velocities projected to world frame (both modes)
  float dx=px*dt, dy=py*dt, dz=pz*dt + zacc_ms2*dt2/2.0f;
  kc_S[KC_STATE_X] += kc_R[0][0]*dx+kc_R[0][1]*dy+kc_R[0][2]*dz;
  kc_S[KC_STATE_Y] += kc_R[1][0]*dx+kc_R[1][1]*dy+kc_R[1][2]*dz;
  kc_S[KC_STATE_Z] += kc_R[2][0]*dx+kc_R[2][1]*dy+kc_R[2][2]*dz - GRAVITY*dt2/2.0f;

  if (quadIsFlying) {
    // Flying mode: body-z thrust only (no X/Y acc), gyro coupling in XY
    kc_S[KC_STATE_PX] += dt*(       gz*py - gy*pz - GRAVITY*kc_R[2][0]);
    kc_S[KC_STATE_PY] += dt*(-gz*px       + gx*pz - GRAVITY*kc_R[2][1]);
    kc_S[KC_STATE_PZ] += dt*(zacc_ms2+gy*px-gx*py - GRAVITY*kc_R[2][2]);
  } else {
    // Ground mode: full 3-axis acc (mirrors kalmanCorePredict ground branch)
    // Maddy AccX/AccY in G's gravity-removed → convert to m/s²
    float ax = AccX * 9.81f;
    float ay = AccY * 9.81f;
    kc_S[KC_STATE_PX] += dt*(ax      + gz*py - gy*pz - GRAVITY*kc_R[2][0]);
    kc_S[KC_STATE_PY] += dt*(ay - gz*px + gx*pz - GRAVITY*kc_R[2][1]);
    kc_S[KC_STATE_PZ] += dt*(zacc_ms2+gy*px-gx*py - GRAVITY*kc_R[2][2]);
  }

  // ====== QUATERNION INTEGRATION from gyro ======
  float dtwx=dt*gx, dtwy=dt*gy, dtwz=dt*gz;
  float angle=sqrtf(dtwx*dtwx+dtwy*dtwy+dtwz*dtwz);
  float tmpq0,tmpq1,tmpq2,tmpq3;
  if (angle > 1e-10f) {
    float ca=cosf(angle/2.0f), sa=sinf(angle/2.0f);
    float dq0=ca, dq1=sa*dtwx/angle, dq2=sa*dtwy/angle, dq3=sa*dtwz/angle;
    tmpq0=dq0*kc_q[0]-dq1*kc_q[1]-dq2*kc_q[2]-dq3*kc_q[3];
    tmpq1=dq1*kc_q[0]+dq0*kc_q[1]+dq3*kc_q[2]-dq2*kc_q[3];
    tmpq2=dq2*kc_q[0]-dq3*kc_q[1]+dq0*kc_q[2]+dq1*kc_q[3];
    tmpq3=dq3*kc_q[0]+dq2*kc_q[1]-dq1*kc_q[2]+dq0*kc_q[3];
  } else {
    tmpq0=kc_q[0]; tmpq1=kc_q[1]; tmpq2=kc_q[2]; tmpq3=kc_q[3];
  }
  // Ground mode: ROLLPITCH_ZERO_REVERSION pulls quaternion toward level
  // (mirrors kalmanCorePredict quadIsFlying==false branch in kalman_core.c)
  if (!quadIsFlying) {
    const float ROLLPITCH_ZERO_REVERSION = 0.001f;
    tmpq0 += ROLLPITCH_ZERO_REVERSION * (1.0f - tmpq0);
    tmpq1 += ROLLPITCH_ZERO_REVERSION * (0.0f - tmpq1);
    tmpq2 += ROLLPITCH_ZERO_REVERSION * (0.0f - tmpq2);
    // tmpq3 (yaw component) left untouched — only roll/pitch reverted
  }
  float norm=sqrtf(tmpq0*tmpq0+tmpq1*tmpq1+tmpq2*tmpq2+tmpq3*tmpq3);
  if (norm>1e-10f) { kc_q[0]=tmpq0/norm; kc_q[1]=tmpq1/norm; kc_q[2]=tmpq2/norm; kc_q[3]=tmpq3/norm; }
}

// ---- Process noise addition  (mirrors kalmanCoreAddProcessNoise) ----
void ekfAddProcessNoise(float dt) {
  if (dt <= 0.0f) return;
  kc_P[KC_STATE_X][KC_STATE_X]   += powf(kc_procNoiseAcc_xy*dt*dt+kc_procNoiseVel*dt+kc_procNoisePos,2);
  kc_P[KC_STATE_Y][KC_STATE_Y]   += powf(kc_procNoiseAcc_xy*dt*dt+kc_procNoiseVel*dt+kc_procNoisePos,2);
  kc_P[KC_STATE_Z][KC_STATE_Z]   += powf(kc_procNoiseAcc_z *dt*dt+kc_procNoiseVel*dt+kc_procNoisePos,2);
  kc_P[KC_STATE_PX][KC_STATE_PX] += powf(kc_procNoiseAcc_xy*dt+kc_procNoiseVel,2);
  kc_P[KC_STATE_PY][KC_STATE_PY] += powf(kc_procNoiseAcc_xy*dt+kc_procNoiseVel,2);
  kc_P[KC_STATE_PZ][KC_STATE_PZ] += powf(kc_procNoiseAcc_z *dt+kc_procNoiseVel,2);
  kc_P[KC_STATE_D0][KC_STATE_D0] += powf(kc_measNoiseGyro_rp *dt+kc_procNoiseAtt,2);
  kc_P[KC_STATE_D1][KC_STATE_D1] += powf(kc_measNoiseGyro_rp *dt+kc_procNoiseAtt,2);
  kc_P[KC_STATE_D2][KC_STATE_D2] += powf(kc_measNoiseGyro_yaw*dt+kc_procNoiseAtt,2);
  // Symmetrize and bound
  for (int i=0;i<KC_STATE_DIM;i++) for (int j=i;j<KC_STATE_DIM;j++) {
    float p=0.5f*(kc_P[i][j]+kc_P[j][i]);
    if (isnan(p)||p>100.0f) kc_P[i][j]=kc_P[j][i]=100.0f;
    else if (i==j&&p<1e-6f) kc_P[i][j]=kc_P[j][i]=1e-6f;
    else                    kc_P[i][j]=kc_P[j][i]=p;
  }
}

// ---- Finalization step  (mirrors kalmanCoreFinalize) ----
// Incorporates attitude error D0/D1/D2 into quaternion, rebuilds R, resets D states.
void ekfFinalize() {
  float v0=kc_S[KC_STATE_D0], v1=kc_S[KC_STATE_D1], v2=kc_S[KC_STATE_D2];

  if ((fabsf(v0)>0.1e-3f||fabsf(v1)>0.1e-3f||fabsf(v2)>0.1e-3f)
      && fabsf(v0)<10.0f && fabsf(v1)<10.0f && fabsf(v2)<10.0f) {
    float angle=sqrtf(v0*v0+v1*v1+v2*v2);
    float ca=cosf(angle/2.0f), sa=sinf(angle/2.0f);
    float dq0=ca, dq1=sa*v0/angle, dq2=sa*v1/angle, dq3=sa*v2/angle;
    float tmpq0=dq0*kc_q[0]-dq1*kc_q[1]-dq2*kc_q[2]-dq3*kc_q[3];
    float tmpq1=dq1*kc_q[0]+dq0*kc_q[1]+dq3*kc_q[2]-dq2*kc_q[3];
    float tmpq2=dq2*kc_q[0]-dq3*kc_q[1]+dq0*kc_q[2]+dq1*kc_q[3];
    float tmpq3=dq3*kc_q[0]+dq2*kc_q[1]-dq1*kc_q[2]+dq0*kc_q[3];
    float norm=sqrtf(tmpq0*tmpq0+tmpq1*tmpq1+tmpq2*tmpq2+tmpq3*tmpq3);
    if (norm>1e-10f) { kc_q[0]=tmpq0/norm; kc_q[1]=tmpq1/norm; kc_q[2]=tmpq2/norm; kc_q[3]=tmpq3/norm; }

    // Rotate covariance (2nd-order approx, same as kalmanCoreFinalize)
    float d0=v0/2, d1=v1/2, d2=v2/2;
    static float Af[KC_STATE_DIM][KC_STATE_DIM];
    memset(Af, 0, sizeof(Af));
    Af[KC_STATE_X][KC_STATE_X]=1; Af[KC_STATE_Y][KC_STATE_Y]=1; Af[KC_STATE_Z][KC_STATE_Z]=1;
    Af[KC_STATE_PX][KC_STATE_PX]=1; Af[KC_STATE_PY][KC_STATE_PY]=1; Af[KC_STATE_PZ][KC_STATE_PZ]=1;
    Af[KC_STATE_D0][KC_STATE_D0]= 1-d1*d1/2-d2*d2/2; Af[KC_STATE_D0][KC_STATE_D1]= d2+d0*d1/2; Af[KC_STATE_D0][KC_STATE_D2]=-d1+d0*d2/2;
    Af[KC_STATE_D1][KC_STATE_D0]=-d2+d0*d1/2;         Af[KC_STATE_D1][KC_STATE_D1]= 1-d0*d0/2-d2*d2/2; Af[KC_STATE_D1][KC_STATE_D2]= d0+d1*d2/2;
    Af[KC_STATE_D2][KC_STATE_D0]= d1+d0*d2/2;         Af[KC_STATE_D2][KC_STATE_D1]=-d0+d1*d2/2; Af[KC_STATE_D2][KC_STATE_D2]= 1-d0*d0/2-d1*d1/2;
    static float tmpf[KC_STATE_DIM][KC_STATE_DIM];
    static float AfT[KC_STATE_DIM][KC_STATE_DIM];
    mat9Mult(tmpf, Af, kc_P);
    for (int i=0;i<KC_STATE_DIM;i++) for (int j=0;j<KC_STATE_DIM;j++) AfT[i][j]=Af[j][i];
    mat9Mult(kc_P, tmpf, AfT);
  }

  // Rebuild rotation matrix from updated quaternion
  kc_R[0][0]=kc_q[0]*kc_q[0]+kc_q[1]*kc_q[1]-kc_q[2]*kc_q[2]-kc_q[3]*kc_q[3];
  kc_R[0][1]=2.0f*(kc_q[1]*kc_q[2]-kc_q[0]*kc_q[3]);
  kc_R[0][2]=2.0f*(kc_q[1]*kc_q[3]+kc_q[0]*kc_q[2]);
  kc_R[1][0]=2.0f*(kc_q[1]*kc_q[2]+kc_q[0]*kc_q[3]);
  kc_R[1][1]=kc_q[0]*kc_q[0]-kc_q[1]*kc_q[1]+kc_q[2]*kc_q[2]-kc_q[3]*kc_q[3];
  kc_R[1][2]=2.0f*(kc_q[2]*kc_q[3]-kc_q[0]*kc_q[1]);
  kc_R[2][0]=2.0f*(kc_q[1]*kc_q[3]-kc_q[0]*kc_q[2]);
  kc_R[2][1]=2.0f*(kc_q[2]*kc_q[3]+kc_q[0]*kc_q[1]);
  kc_R[2][2]=kc_q[0]*kc_q[0]-kc_q[1]*kc_q[1]-kc_q[2]*kc_q[2]+kc_q[3]*kc_q[3];

  // Reset attitude error states
  kc_S[KC_STATE_D0]=0; kc_S[KC_STATE_D1]=0; kc_S[KC_STATE_D2]=0;

  // Symmetrize and bound covariance
  for (int i=0;i<KC_STATE_DIM;i++) for (int j=i;j<KC_STATE_DIM;j++) {
    float p=0.5f*(kc_P[i][j]+kc_P[j][i]);
    if (isnan(p)||p>100.0f) kc_P[i][j]=kc_P[j][i]=100.0f;
    else if (i==j&&p<1e-6f) kc_P[i][j]=kc_P[j][i]=1e-6f;
    else                    kc_P[i][j]=kc_P[j][i]=p;
  }
}

// ---- ToF measurement update  (mirrors kalmanCoreUpdateWithTof) ----
// Tilt-corrected: h[Z] = 1/R[2][2], predictedDist = S[Z]/R[2][2]
void ekfUpdateTof(float z_meas_m) {
  if (fabsf(kc_R[2][2]) < 0.1f || kc_R[2][2] <= 0) return;
  float predictedDist = kc_S[KC_STATE_Z] / kc_R[2][2];
  float h[KC_STATE_DIM] = {0};
  h[KC_STATE_Z] = 1.0f / kc_R[2][2];
  ekfScalarUpdate(h, z_meas_m - predictedDist, kc_tof_stdDev);
  if (kc_S[KC_STATE_Z] < 0.0f) kc_S[KC_STATE_Z] = 0.0f;
}

// ---- Flow measurement update  (mirrors kalmanCoreUpdateWithFlow) ----
// dpixelx = -deltaY, dpixely = -deltaX  (axis swap from flowdeck_v1v2.c)
// flow_dt : seconds since last flow sample
// gyroX/Y_dps : body gyro deg/s
// flow_std : pixel noise std dev (from shutter or default)
void ekfUpdateFlow(float dpixelx, float dpixely, float flow_dt,
                   float gyroX_dps, float gyroY_dps, float flow_std) {
  const float DEG2RAD   = (float)M_PI / 180.0f;
  const float Npix      = 30.0f;               // [pixels], same as kalman_core.c
  const float thetapix  = 4.2f * DEG2RAD;      // [rad], camera FOV/Npix
  const float omegaFactor = 1.25f;
  float omegax_b = gyroX_dps * DEG2RAD;
  float omegay_b = gyroY_dps * DEG2RAD;
  // Modification 1 (kalman_core.c line 491): use body-frame velocity directly
  float dx_g = kc_S[KC_STATE_PX];
  float dy_g = kc_S[KC_STATE_PY];
  float z_g  = (kc_S[KC_STATE_Z] < 0.1f) ? 0.1f : kc_S[KC_STATE_Z];

  // X pixel update
  float predictedNX = (flow_dt*Npix/thetapix)*((dx_g*kc_R[2][2]/z_g) - omegaFactor*omegay_b);
  float hx[KC_STATE_DIM] = {0};
  hx[KC_STATE_Z]  = (Npix*flow_dt/thetapix)*((kc_R[2][2]*dx_g)/(-z_g*z_g));
  hx[KC_STATE_PX] = (Npix*flow_dt/thetapix)*(kc_R[2][2]/z_g);
  ekfScalarUpdate(hx, dpixelx - predictedNX, flow_std);

  // Y pixel update
  float predictedNY = (flow_dt*Npix/thetapix)*((dy_g*kc_R[2][2]/z_g) + omegaFactor*omegax_b);
  float hy[KC_STATE_DIM] = {0};
  hy[KC_STATE_Z]  = (Npix*flow_dt/thetapix)*((kc_R[2][2]*dy_g)/(-z_g*z_g));
  hy[KC_STATE_PY] = (Npix*flow_dt/thetapix)*(kc_R[2][2]/z_g);
  ekfScalarUpdate(hy, dpixely - predictedNY, flow_std);
}

// ---------- Read VL53L1X at 50 Hz, set lw_tof_new_data flag ----------
void litewingReadAltitude() {
  if (!lw_tof_ready) return;
  if (millis() - lw_tof_update_ms < LW_TOF_INTERVAL_MS) return;
  lw_tof_update_ms = millis();
  if (!lw_tof.dataReady()) return;
  uint16_t raw_mm = lw_tof.read(false);   // non-blocking read
  if (raw_mm > 50 && raw_mm < 4000) {     // valid 5 cm – 4 m
    lw_tof_raw_m    = raw_mm / 1000.0f;
    lw_tof_new_data = true;
  }
}

// ---------- 2-stage altitude hold  (mirrors position_controller_pid.c) ----------
// Runs EKF predict+process-noise+ToF-update+finalize, then 2-stage PID.
void litewingAltitudeHold(float dt) {
  if (dt <= 0.0f || dt > 0.05f) return;
  if (!kc_initialized) { ekfInit(); return; }

  // quadIsFlying: mirrors estimator_kalman.c — flying when throttle > 10%
  kc_quadIsFlying = (thro_des > 0.10f);

  // zacc: body-z acceleration in m/s² INCLUDING gravity (kalman_core.c convention)
  // AccZ already has B_accel (~0.14) applied in getIMUdata() — Madgwick uses that.
  // We apply a SECOND, heavier LP (B_zacc=0.05) only to zacc fed into the EKF
  // to reject motor vibration spikes from the Kalman predict step.
  // Madgwick is completely unaffected — it still receives the B_accel-filtered AccZ.
  static float zacc_filt = 0.0f;
  static bool  zacc_init  = false;
  const  float B_zacc     = 0.05f;   // heavier than B_accel; tune down if EKF lags
  float zacc_raw = (AccZ + 1.0f) * 9.81f;
  if (!zacc_init) { zacc_filt = zacc_raw; zacc_init = true; }
  zacc_filt = (1.0f - B_zacc) * zacc_filt + B_zacc * zacc_raw;
  float zacc = zacc_filt;
  ekfPredict(dt, GyroX, GyroY, GyroZ, zacc, kc_quadIsFlying);
  ekfAddProcessNoise(dt);

  // All measurement updates before finalize (mirrors estimator_kalman.c loop order)
  if (lw_tof_new_data) {
    ekfUpdateTof(lw_tof_raw_m);
    lw_tof_new_data = false;
  }
  litewingReadFlow();  // flow update feeds EKF before finalize (fixes A1b)
  ekfFinalize();

  // kalmanSupervisorIsStateWithinBounds: reinit if EKF diverges (fixes B1)
  // Mirrors kalman_supervisor.c: pos ±100 m, vel ±10 m/s
  bool ekf_diverged = false;
  for (int i = KC_STATE_X; i <= KC_STATE_Z; i++)  if (fabsf(kc_S[i]) > 100.0f) ekf_diverged = true;
  for (int i = KC_STATE_PX; i <= KC_STATE_PZ; i++) if (fabsf(kc_S[i]) > 10.0f)  ekf_diverged = true;
  if (ekf_diverged) {
    Serial.println("[EKF] State out of bounds — reinitializing (kalman_supervisor)");
    ekfInit();
    return;
  }

  if (!lw_althold_enabled) {
    lw_thrust_correction = 0.0f;
    lw_pidZ_integral     = 0.0f;
    lw_pidVZ_integral    = 0.0f;
    return;
  }

  // Auto-capture hover throttle (thrustBase equivalent)
  if (!lw_hover_thr_captured) {
    lw_hover_throttle      = constrain((channel_1_pwm - 1000.0f) / 1000.0f, 0.2f, 0.9f);
    lw_hover_thr_captured  = true;
    lw_altitude_setpoint_m = kc_S[KC_STATE_Z];
    Serial.printf("[LiteWing] AltHold ON  hover_thr=%.2f  setpoint=%.2fm\n",
                  lw_hover_throttle, lw_altitude_setpoint_m);
  }

  float kc_z = kc_S[KC_STATE_Z];
  // Hard ceiling: if above 2.8m clamp setpoint down so drone descends
  if (lw_altitude_setpoint_m > 2.5f) lw_altitude_setpoint_m = 2.5f;
  if (kc_z > 2.8f) lw_altitude_setpoint_m = min(lw_altitude_setpoint_m, kc_z - 0.3f);
  if (kc_z < 0.05f || kc_z > 3.5f) { lw_thrust_correction = 0.0f; return; }

  // World-frame vertical velocity: vz_world = R * [PX,PY,PZ]
  float vz_world = kc_R[2][0]*kc_S[KC_STATE_PX]
                 + kc_R[2][1]*kc_S[KC_STATE_PY]
                 + kc_R[2][2]*kc_S[KC_STATE_PZ];

  // Outer loop: Z position → Z velocity setpoint (pidZ)
  float z_err = lw_altitude_setpoint_m - kc_z;
  lw_pidZ_integral += z_err * dt;
  lw_pidZ_integral  = constrain(lw_pidZ_integral, -0.5f, 0.5f);
  float z_d = (z_err - lw_pidZ_prev_err) / dt;
  lw_pidZ_prev_err  = z_err;
  lw_vel_z_sp = lw_pidZ_kp*z_err + lw_pidZ_ki*lw_pidZ_integral + lw_pidZ_kd*z_d;
  // Anti-windup: if output is saturated, undo the integral step to prevent windup
  if (lw_vel_z_sp > 0.4f || lw_vel_z_sp < -0.4f)
    lw_pidZ_integral -= z_err * dt;
  lw_vel_z_sp = constrain(lw_vel_z_sp, -0.2f, 0.2f);  // max 0.2 m/s climb/descent (autonomous indoor)

  // Inner loop: Z velocity → thrust correction (pidVZ)
  float vz_err = lw_vel_z_sp - vz_world;
  lw_pidVZ_integral += vz_err * dt;
  lw_pidVZ_integral  = constrain(lw_pidVZ_integral, -0.5f, 0.5f);
  lw_thrust_correction = lw_pidVZ_kp*vz_err + lw_pidVZ_ki*lw_pidVZ_integral;
  lw_thrust_correction = constrain(lw_thrust_correction, -0.30f, 0.30f);
}

// ---- STEP 1: Verify sensors are reading before enabling any hold ----
// Expected: tof_m should match physical height; flow dpx/dpy should be ~0 when still
// motion=0xB0 means valid flow; shutter<200 means good lighting
void printLitewingSensors() {
  if (current_time - print_counter > 50000) {  // 20 Hz
    print_counter = micros();
    Serial.printf("[SENSOR] ToF=%.3fm(raw)  EKF_Z=%.3fm  flow_motion=0x%02X  dpx=%d dpy=%d  shutter=%d\n",
      lw_tof_raw_m, kc_S[KC_STATE_Z],
      pmw_motion.motion, (int16_t)(-pmw_motion.deltaY), (int16_t)(-pmw_motion.deltaX),
      pmw_motion.shutter);
  }
}

// ---- STEP 2: Verify EKF is running correctly (use before enabling hold) ----
// Expected: Z should track ToF; XY should be ~0 at start; velocities should be small when hovering
// P_Z/P_VZ diagonal shows EKF confidence (should shrink after ToF updates)
void printLitewingEKF() {
  if (current_time - print_counter > 50000) {  // 20 Hz
    print_counter = micros();
    float vx_w=kc_R[0][0]*kc_S[KC_STATE_PX]+kc_R[0][1]*kc_S[KC_STATE_PY]+kc_R[0][2]*kc_S[KC_STATE_PZ];
    float vy_w=kc_R[1][0]*kc_S[KC_STATE_PX]+kc_R[1][1]*kc_S[KC_STATE_PY]+kc_R[1][2]*kc_S[KC_STATE_PZ];
    float vz_w=kc_R[2][0]*kc_S[KC_STATE_PX]+kc_R[2][1]*kc_S[KC_STATE_PY]+kc_R[2][2]*kc_S[KC_STATE_PZ];
    Serial.printf("[EKF] x=%.3f y=%.3f z=%.3fm | vx=%.3f vy=%.3f vz=%.3fm/s | P_Z=%.4f P_VZ=%.4f | q=[%.3f %.3f %.3f %.3f]\n",
      kc_S[KC_STATE_X], kc_S[KC_STATE_Y], kc_S[KC_STATE_Z],
      vx_w, vy_w, vz_w,
      kc_P[KC_STATE_Z][KC_STATE_Z], kc_P[KC_STATE_PZ][KC_STATE_PZ],
      kc_q[0], kc_q[1], kc_q[2], kc_q[3]);
  }
}

// ---- STEP 3: Tune altitude hold (enable althold, check these values) ----
// Expected: z_err→0, vel_z_sp tracks smoothly, thr_corr oscillates then settles near 0
// If oscillating: reduce pidVZ_kp. If slow to settle: increase pidZ_kp.
void printLitewingAltHold() {
  if (current_time - print_counter > 50000) {  // 20 Hz
    print_counter = micros();
    float vz_w=kc_R[2][0]*kc_S[KC_STATE_PX]+kc_R[2][1]*kc_S[KC_STATE_PY]+kc_R[2][2]*kc_S[KC_STATE_PZ];
    float z_err = lw_altitude_setpoint_m - kc_S[KC_STATE_Z];
    Serial.printf("[ALTH] sp=%.3fm  z=%.3fm  err=%.3fm | vel_sp=%.3f  vz=%.3f  vz_err=%.3f | thr_corr=%.3f  hover_thr=%.2f\n",
      lw_altitude_setpoint_m, kc_S[KC_STATE_Z], z_err,
      lw_vel_z_sp, vz_w, lw_vel_z_sp - vz_w,
      lw_thrust_correction, lw_hover_throttle);
  }
}

// ---- STEP 4: Tune position hold (enable poshold, check these values) ----
// Expected: x_err/y_err→0, roll/pitch corrections small and stable
// If drifting in one direction: check axis swap (dpx/dpy signs). If oscillating: reduce pidVX_kp.
void printLitewingPosHold() {
  if (current_time - print_counter > 50000) {  // 20 Hz
    print_counter = micros();
    float vx_w=kc_R[0][0]*kc_S[KC_STATE_PX]+kc_R[0][1]*kc_S[KC_STATE_PY]+kc_R[0][2]*kc_S[KC_STATE_PZ];
    float vy_w=kc_R[1][0]*kc_S[KC_STATE_PX]+kc_R[1][1]*kc_S[KC_STATE_PY]+kc_R[1][2]*kc_S[KC_STATE_PZ];
    Serial.printf("[POSH] sp=(%.3f,%.3f)  pos=(%.3f,%.3f)  err=(%.3f,%.3f) | vx=%.3f vy=%.3f | roll_c=%.3f pitch_c=%.3f\n",
      lw_posX_sp, lw_posY_sp,
      kc_S[KC_STATE_X], kc_S[KC_STATE_Y],
      lw_posX_sp - kc_S[KC_STATE_X], lw_posY_sp - kc_S[KC_STATE_Y],
      vx_w, vy_w,
      lw_roll_correction, lw_pitch_correction);
  }
}

// ---- General overview (all-in-one) ----
void printLitewingData() {
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    float vx_w=kc_R[0][0]*kc_S[KC_STATE_PX]+kc_R[0][1]*kc_S[KC_STATE_PY]+kc_R[0][2]*kc_S[KC_STATE_PZ];
    float vy_w=kc_R[1][0]*kc_S[KC_STATE_PX]+kc_R[1][1]*kc_S[KC_STATE_PY]+kc_R[1][2]*kc_S[KC_STATE_PZ];
    float vz_w=kc_R[2][0]*kc_S[KC_STATE_PX]+kc_R[2][1]*kc_S[KC_STATE_PY]+kc_R[2][2]*kc_S[KC_STATE_PZ];
    Serial.printf("LW  x=%.3f y=%.3f z=%.3fm | vx=%.3f vy=%.3f vz=%.3fm/s | thr_corr=%.3f | AltH:%s PosH:%s\n",
      kc_S[KC_STATE_X], kc_S[KC_STATE_Y], kc_S[KC_STATE_Z],
      vx_w, vy_w, vz_w, lw_thrust_correction,
      lw_althold_enabled ? "ON" : "OFF",
      lw_poshold_active  ? "ON" : "OFF");
  }
}

// =========================================================================
// PMW3901 SPI DRIVER  (ported from pmw3901.c — Bitcraze/ESP-Drone)
// =========================================================================

static void pmw3901RegisterWrite(uint8_t reg, uint8_t value) {
  reg |= 0x80u;  // MSB=1 for write
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
  digitalWrite(PMW3901_CS_PIN, LOW);
  delayMicroseconds(50);
  SPI.transfer(reg);
  delayMicroseconds(50);
  SPI.transfer(value);
  delayMicroseconds(50);
  digitalWrite(PMW3901_CS_PIN, HIGH);
  SPI.endTransaction();
  delayMicroseconds(200);
}

static uint8_t pmw3901RegisterRead(uint8_t reg) {
  reg &= ~0x80u;  // MSB=0 for read
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
  digitalWrite(PMW3901_CS_PIN, LOW);
  delayMicroseconds(50);
  SPI.transfer(reg);
  delayMicroseconds(500);
  uint8_t data = SPI.transfer(0x00);
  delayMicroseconds(50);
  digitalWrite(PMW3901_CS_PIN, HIGH);
  SPI.endTransaction();
  delayMicroseconds(200);
  return data;
}

static void pmw3901InitRegisters() {
  // Full register initialisation sequence from pmw3901.c (Bitcraze)
  pmw3901RegisterWrite(0x7F, 0x00); pmw3901RegisterWrite(0x61, 0xAD);
  pmw3901RegisterWrite(0x7F, 0x03); pmw3901RegisterWrite(0x40, 0x00);
  pmw3901RegisterWrite(0x7F, 0x05); pmw3901RegisterWrite(0x41, 0xB3);
  pmw3901RegisterWrite(0x43, 0xF1); pmw3901RegisterWrite(0x45, 0x14);
  pmw3901RegisterWrite(0x5B, 0x32); pmw3901RegisterWrite(0x5F, 0x34);
  pmw3901RegisterWrite(0x7B, 0x08); pmw3901RegisterWrite(0x7F, 0x06);
  pmw3901RegisterWrite(0x44, 0x1B); pmw3901RegisterWrite(0x40, 0xBF);
  pmw3901RegisterWrite(0x4E, 0x3F); pmw3901RegisterWrite(0x7F, 0x08);
  pmw3901RegisterWrite(0x65, 0x20); pmw3901RegisterWrite(0x6A, 0x18);
  pmw3901RegisterWrite(0x7F, 0x09); pmw3901RegisterWrite(0x4F, 0xAF);
  pmw3901RegisterWrite(0x5F, 0x40); pmw3901RegisterWrite(0x48, 0x80);
  pmw3901RegisterWrite(0x49, 0x80); pmw3901RegisterWrite(0x57, 0x77);
  pmw3901RegisterWrite(0x60, 0x78); pmw3901RegisterWrite(0x61, 0x78);
  pmw3901RegisterWrite(0x62, 0x08); pmw3901RegisterWrite(0x63, 0x50);
  pmw3901RegisterWrite(0x7F, 0x0A); pmw3901RegisterWrite(0x45, 0x60);
  pmw3901RegisterWrite(0x7F, 0x00); pmw3901RegisterWrite(0x4D, 0x11);
  pmw3901RegisterWrite(0x55, 0x80); pmw3901RegisterWrite(0x74, 0x1F);
  pmw3901RegisterWrite(0x75, 0x1F); pmw3901RegisterWrite(0x4A, 0x78);
  pmw3901RegisterWrite(0x4B, 0x78); pmw3901RegisterWrite(0x44, 0x08);
  pmw3901RegisterWrite(0x45, 0x50); pmw3901RegisterWrite(0x64, 0xFF);
  pmw3901RegisterWrite(0x65, 0x1F); pmw3901RegisterWrite(0x7F, 0x14);
  pmw3901RegisterWrite(0x65, 0x67); pmw3901RegisterWrite(0x66, 0x08);
  pmw3901RegisterWrite(0x63, 0x70); pmw3901RegisterWrite(0x7F, 0x15);
  pmw3901RegisterWrite(0x48, 0x48); pmw3901RegisterWrite(0x7F, 0x07);
  pmw3901RegisterWrite(0x41, 0x0D); pmw3901RegisterWrite(0x43, 0x14);
  pmw3901RegisterWrite(0x4B, 0x0E); pmw3901RegisterWrite(0x45, 0x0F);
  pmw3901RegisterWrite(0x44, 0x42); pmw3901RegisterWrite(0x4C, 0x80);
  pmw3901RegisterWrite(0x7F, 0x10); pmw3901RegisterWrite(0x5B, 0x02);
  pmw3901RegisterWrite(0x7F, 0x07); pmw3901RegisterWrite(0x40, 0x41);
  pmw3901RegisterWrite(0x70, 0x00);
  delay(10);
  pmw3901RegisterWrite(0x32, 0x44); pmw3901RegisterWrite(0x7F, 0x07);
  pmw3901RegisterWrite(0x40, 0x40); pmw3901RegisterWrite(0x7F, 0x06);
  pmw3901RegisterWrite(0x62, 0xF0); pmw3901RegisterWrite(0x63, 0x00);
  pmw3901RegisterWrite(0x7F, 0x0D); pmw3901RegisterWrite(0x48, 0xC0);
  pmw3901RegisterWrite(0x6F, 0xD5); pmw3901RegisterWrite(0x7F, 0x00);
  pmw3901RegisterWrite(0x5B, 0xA0); pmw3901RegisterWrite(0x4E, 0xA8);
  pmw3901RegisterWrite(0x5A, 0x50); pmw3901RegisterWrite(0x40, 0x80);
  pmw3901RegisterWrite(0x7F, 0x00); pmw3901RegisterWrite(0x5A, 0x10);
  pmw3901RegisterWrite(0x54, 0x00);
}

bool pmw3901Init() {
  pinMode(PMW3901_CS_PIN, OUTPUT);
  digitalWrite(PMW3901_CS_PIN, HIGH);
  delay(40);
  // Toggle CS to reset (from pmw3901.c)
  digitalWrite(PMW3901_CS_PIN, HIGH); delay(2);
  digitalWrite(PMW3901_CS_PIN, LOW);  delay(2);
  digitalWrite(PMW3901_CS_PIN, HIGH); delay(2);

  uint8_t chipId    = pmw3901RegisterRead(0x00);
  uint8_t invChipId = pmw3901RegisterRead(0x5F);
  Serial.printf("[PMW3901] chip=0x%02X inv=0x%02X\n", chipId, invChipId);

  // Accept if EITHER chipId or invChipId matches (mirrors flowdeck_v1v2.c)
  if (chipId != 0x49 && invChipId != 0xB6) {
    Serial.println("[PMW3901] Bad chip ID — check wiring");
    return false;
  }
  // Power-on reset
  pmw3901RegisterWrite(0x3A, 0x5A);
  delay(5);
  // Flush motion registers
  pmw3901RegisterRead(0x02); pmw3901RegisterRead(0x03);
  pmw3901RegisterRead(0x04); pmw3901RegisterRead(0x05);
  pmw3901RegisterRead(0x06);
  delay(1);
  pmw3901InitRegisters();
  return true;
}

void pmw3901ReadMotion(motionBurst_t *m) {
  uint8_t addr = 0x16;  // burst read base register
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
  digitalWrite(PMW3901_CS_PIN, LOW);
  delayMicroseconds(50);
  SPI.transfer(addr);
  delayMicroseconds(50);
  uint8_t *buf = (uint8_t*)m;
  for (size_t i = 0; i < sizeof(motionBurst_t); i++) buf[i] = SPI.transfer(0x00);
  delayMicroseconds(50);
  digitalWrite(PMW3901_CS_PIN, HIGH);
  SPI.endTransaction();
  delayMicroseconds(50);
  // Swap shutter bytes (from pmw3901.c)
  uint16_t s = m->shutter;
  m->shutter = ((s >> 8) & 0xFF) | ((s & 0xFF) << 8);
}

// =========================================================================
// FLOW DECK READ  (mirrors flowdeck_v1v2.c → estimatorEnqueueFlow)
// Reads PMW3901 burst, validates, feeds into 9-state EKF flow update.
// =========================================================================
void litewingReadFlow() {
  if (!pmw3901_ready) return;
  pmw3901ReadMotion(&pmw_motion);
  // Motion register must be 0xB0 for valid data (flowdeck_v1v2.c)
  if (pmw_motion.motion != 0xB0) return;

  // Axis swap + negate (flowdeck_v1v2.c: accpx=-deltaY, accpy=-deltaX)
  int16_t dpx = -pmw_motion.deltaY;
  int16_t dpy = -pmw_motion.deltaX;
  // Outlier rejection (OUTLIER_LIMIT=100 from flowdeck_v1v2.c)
  if (abs(dpx) >= 100 || abs(dpy) >= 100) return;

  unsigned long now_ms = millis();
  pmw_dt = (now_ms - pmw_last_ms) / 1000.0f;
  pmw_last_ms = now_ms;
  if (pmw_dt <= 0.0f || pmw_dt > 0.2f) pmw_dt = 0.01f;

  // Compute flow std dev from shutter (mirrors flowdeck_v1v2.c)
  float shutter_f = (float)(pmw_motion.shutter);
  float flow_std = (shutter_f > 0) ? (0.0007984f*shutter_f + 0.4335f) : kc_flow_stdDev;
  flow_std = constrain(flow_std, 0.1f, 10.0f);  // floor 0.1 matches flowdeck_v1v2.c

  // Feed into 9-state EKF (proper pixel measurement model — NOT dead reckoning)
  ekfUpdateFlow((float)dpx, (float)dpy, pmw_dt, GyroX, GyroY, flow_std);
}

// =========================================================================
// XY POSITION HOLD  (mirrors position_controller_pid.c positionController +
//                   velocityController for X/Y axes)
// EKF predict+finalize+flow already ran in litewingAltitudeHold; only PID here.
// =========================================================================
void litewingPositionHold(float dt) {
  if (dt <= 0.0f || dt > 0.05f) return;

  // Note: litewingReadFlow() is now called inside litewingAltitudeHold before
  // ekfFinalize() to match estimator_kalman.c loop order (A1b fix).

  if (!lw_poshold_active) {
    lw_roll_correction = lw_pitch_correction = 0.0f;
    lw_pidX_int = lw_pidY_int = 0.0f;
    lw_pidVX_int = lw_pidVY_int = 0.0f;
    lw_pidX_prev = lw_pidY_prev = 0.0f;
    lw_pidVX_prev = lw_pidVY_prev = 0.0f;
    return;
  }

  if (kc_S[KC_STATE_Z] < 0.10f) { lw_roll_correction = lw_pitch_correction = 0.0f; return; }

  // World-frame XY velocity: vw = R * [PX, PY, PZ]
  float vx_w = kc_R[0][0]*kc_S[KC_STATE_PX]+kc_R[0][1]*kc_S[KC_STATE_PY]+kc_R[0][2]*kc_S[KC_STATE_PZ];
  float vy_w = kc_R[1][0]*kc_S[KC_STATE_PX]+kc_R[1][1]*kc_S[KC_STATE_PY]+kc_R[1][2]*kc_S[KC_STATE_PZ];

  // Outer loop: X position → X velocity setpoint (pidX)
  float ex = lw_posX_sp - kc_S[KC_STATE_X];
  lw_pidX_int += ex * dt;
  lw_pidX_int  = constrain(lw_pidX_int, -0.5f, 0.5f);
  float dx = (ex - lw_pidX_prev) / dt;
  lw_pidX_prev = ex;
  lw_velX_sp = lw_pidX_kp*ex + lw_pidX_ki*lw_pidX_int + lw_pidX_kd*dx;
  lw_velX_sp = constrain(lw_velX_sp, -1.0f, 1.0f);

  float ey = lw_posY_sp - kc_S[KC_STATE_Y];
  lw_pidY_int += ey * dt;
  lw_pidY_int  = constrain(lw_pidY_int, -0.5f, 0.5f);
  float dy = (ey - lw_pidY_prev) / dt;
  lw_pidY_prev = ey;
  lw_velY_sp = lw_pidY_kp*ey + lw_pidY_ki*lw_pidY_int + lw_pidY_kd*dy;
  lw_velY_sp = constrain(lw_velY_sp, -1.0f, 1.0f);

  // Inner loop: velocity → roll/pitch correction (pidVX/pidVY)
  float evx = lw_velX_sp - vx_w;
  lw_pidVX_int += evx * dt;
  lw_pidVX_int  = constrain(lw_pidVX_int, -1.0f, 1.0f);
  float dvx = (evx - lw_pidVX_prev) / dt;
  lw_pidVX_prev = evx;
  float roll_raw = lw_pidVX_kp*evx + lw_pidVX_ki*lw_pidVX_int + lw_pidVX_kd*dvx;

  float evy = lw_velY_sp - vy_w;
  lw_pidVY_int += evy * dt;
  lw_pidVY_int  = constrain(lw_pidVY_int, -1.0f, 1.0f);
  float dvy = (evy - lw_pidVY_prev) / dt;
  lw_pidVY_prev = evy;
  float pitch_raw = lw_pidVY_kp*evy + lw_pidVY_ki*lw_pidVY_int + lw_pidVY_kd*dvy;

  // Yaw rotation compensation (position_controller_pid.c)
  float yaw_rad = yaw_IMU * (float)M_PI / 180.0f;
  float cy = cosf(yaw_rad), sy = sinf(yaw_rad);
  float pitch_cmd = -(roll_raw*cy) - (pitch_raw*sy);
  float roll_cmd  = -(pitch_raw*cy) + (roll_raw*sy);

  lw_roll_correction  = constrain(roll_cmd,  -1.0f, 1.0f);
  lw_pitch_correction = constrain(pitch_cmd, -1.0f, 1.0f);
}

#endif // USE_LITEWING_MODULE

//========================================================================================================================//
//                                              FLIGHT CONTROL FUNCTIONS                                                 //
//========================================================================================================================//

void controlMixer() {
  //DESCRIPTION: Mix PID outputs for quadcopter (X configuration)
  //yaw_scale is now global and can be adjusted via web UI

  m1_command_scaled = thro_des - pitch_PID - roll_PID + (yaw_PID * yaw_scale); //Front Right (CCW)
  m2_command_scaled = thro_des + pitch_PID - roll_PID - (yaw_PID * yaw_scale); //Back Right  (CW)
  m3_command_scaled = thro_des + pitch_PID + roll_PID + (yaw_PID * yaw_scale); //Back Left   (CCW)
  m4_command_scaled = thro_des - pitch_PID + roll_PID - (yaw_PID * yaw_scale); //Front Left  (CW)
}

void armedStatus() {
  //DESCRIPTION: Check if ready to fly
  if ((channel_5_pwm < 1500) && (channel_1_pwm < 1050)) {
    armedFly = true;
  }
}

void IMUinit() {
  //DESCRIPTION: Initialize MPU6050 IMU

  if (!mpu.begin()) {
    Serial.println("MPU6050 initialization FAILED");
    Serial.println("Check wiring or try cycling power");
    while(1) {}
  }

  Serial.println("MPU6050 initialized successfully!");

  //Set gyro range
  #if defined GYRO_250DPS
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  #elif defined GYRO_500DPS
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  #elif defined GYRO_1000DPS
    mpu.setGyroRange(MPU6050_RANGE_1000_DEG);
  #elif defined GYRO_2000DPS
    mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
  #endif

  //Set accelerometer range
  #if defined ACCEL_2G
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  #elif defined ACCEL_4G
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  #elif defined ACCEL_8G
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  #elif defined ACCEL_16G
    mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  #endif

  //Set filter bandwidth to 21Hz (good for flight control)
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void getIMUdata() {
  //DESCRIPTION: Read and filter IMU data

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  //Accelerometer (convert m/s^2 to G's)
  AccX = a.acceleration.x / 9.81;
  AccY = a.acceleration.y / 9.81;
  AccZ = a.acceleration.z / 9.81;
  AccX = AccX - AccErrorX;
  AccY = AccY - AccErrorY;
  AccZ = AccZ - AccErrorZ;

  //LP filter accelerometer
  AccX = (1.0 - B_accel)*AccX_prev + B_accel*AccX;
  AccY = (1.0 - B_accel)*AccY_prev + B_accel*AccY;
  AccZ = (1.0 - B_accel)*AccZ_prev + B_accel*AccZ;
  AccX_prev = AccX;
  AccY_prev = AccY;
  AccZ_prev = AccZ;

  //Reject corrupted accelerometer data
  float accelMag = sqrt(AccX*AccX + AccY*AccY + AccZ*AccZ);
  if (accelMag < 0.7 || accelMag > 1.3) {
    AccX = AccX_prev;
    AccY = AccY_prev;
    AccZ = AccZ_prev;
  }

  //Gyro (convert rad/s to deg/s)
  GyroX = g.gyro.x * 57.2957795;
  GyroY = g.gyro.y * 57.2957795;
  GyroZ = g.gyro.z * 57.2957795;
  GyroX = GyroX - GyroErrorX;
  GyroY = GyroY - GyroErrorY;
  GyroZ = GyroZ - GyroErrorZ;

  //LP filter gyro
  GyroX = (1.0 - B_gyro)*GyroX_prev + B_gyro*GyroX;
  GyroY = (1.0 - B_gyro)*GyroY_prev + B_gyro*GyroY;
  GyroZ = (1.0 - B_gyro)*GyroZ_prev + B_gyro*GyroZ;
  GyroX_prev = GyroX;
  GyroY_prev = GyroY;
  GyroZ_prev = GyroZ;
}

void calculate_IMU_error() {
  //DESCRIPTION: Computes IMU accelerometer and gyro error on startup. Note: vehicle should be powered up on flat surface
  /*
   * The error values it computes are applied to the raw gyro and
   * accelerometer values AccX, AccY, AccZ, GyroX, GyroY, GyroZ in getIMUdata(). This eliminates drift in the
   * measurement. Keep vehicle steady on flat surface while this function runs.
   */

  AccErrorX = 0.0;
  AccErrorY = 0.0;
  AccErrorZ = 0.0;
  GyroErrorX = 0.0;
  GyroErrorY = 0.0;
  GyroErrorZ = 0.0;

  Serial.println("\n========================================");
  Serial.println("IMU CALIBRATION - Keep vehicle FLAT and STILL!");
  Serial.println("========================================");
  Serial.println("Collecting 12000 samples...\n");

  //Read IMU values 12000 times
  int c = 0;
  while (c < 12000) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    //Accelerometer (convert m/s^2 to G's)
    float ax = a.acceleration.x / 9.81;
    float ay = a.acceleration.y / 9.81;
    float az = a.acceleration.z / 9.81;

    //Gyro (convert rad/s to deg/s)
    float gx = g.gyro.x * 57.2957795;
    float gy = g.gyro.y * 57.2957795;
    float gz = g.gyro.z * 57.2957795;

    //Sum all readings
    AccErrorX = AccErrorX + ax;
    AccErrorY = AccErrorY + ay;
    AccErrorZ = AccErrorZ + az;
    GyroErrorX = GyroErrorX + gx;
    GyroErrorY = GyroErrorY + gy;
    GyroErrorZ = GyroErrorZ + gz;
    c++;

    //Print progress every 1000 samples
    if (c % 1000 == 0) {
      Serial.print("Progress: ");
      Serial.print(c);
      Serial.println(" / 12000");
    }
  }

  //Divide the sum by 12000 to get the error value
  AccErrorX = AccErrorX / c;
  AccErrorY = AccErrorY / c;
  AccErrorZ = AccErrorZ / c - 1.0;  //Subtract 1G from Z axis (gravity)
  GyroErrorX = GyroErrorX / c;
  GyroErrorY = GyroErrorY / c;
  GyroErrorZ = GyroErrorZ / c;

  Serial.println("\n========================================");
  Serial.println("CALIBRATION COMPLETE!");
  Serial.println("========================================");
  Serial.println("Copy these lines and paste at lines 105-110:\n");

  Serial.print("float AccErrorX = ");
  Serial.print(AccErrorX, 4);
  Serial.println(";");
  Serial.print("float AccErrorY = ");
  Serial.print(AccErrorY, 4);
  Serial.println(";");
  Serial.print("float AccErrorZ = ");
  Serial.print(AccErrorZ, 4);
  Serial.println(";");
  Serial.print("float GyroErrorX = ");
  Serial.print(GyroErrorX, 4);
  Serial.println(";");
  Serial.print("float GyroErrorY = ");
  Serial.print(GyroErrorY, 4);
  Serial.println(";");
  Serial.print("float GyroErrorZ = ");
  Serial.print(GyroErrorZ, 4);
  Serial.println(";");

  Serial.println("\nAfter pasting, comment out calculate_IMU_error() in setup()");
  Serial.println("========================================\n");

  while(1); 
}

void calibrateAttitude() {
  //DESCRIPTION: Used to warm up the main loop to allow the Madgwick filter to converge before commands can be sent to the actuators
  //Assuming vehicle is powered up on level surface!
  /*
   * This function is used on startup to warm up the attitude estimation and is what causes startup to take a few seconds
   * to boot. It eliminates the ~30 second convergence delay that would otherwise occur on takeoff.
   * Runs 10000 iterations of the main IMU loop at 2kHz to let the filter converge to accurate attitude.
   */

  Serial.println("Calibrating attitude... (warming up Madgwick filter)");

  //Warm up IMU and Madgwick filter in simulated main loop
  for (int i = 0; i <= 10000; i++) {
    prev_time = current_time;
    current_time = micros();
    dt = (current_time - prev_time)/1000000.0;
    getIMUdata();
    Madgwick(GyroX, -GyroY, -GyroZ, -AccX, AccY, AccZ, MagY, -MagX, MagZ, dt);
    loopRate(2000); //do not exceed 2000Hz

    //Print progress every 2000 iterations
    if (i % 2000 == 0 && i > 0) {
      Serial.print("Progress: ");
      Serial.print(i);
      Serial.println(" / 10000");
    }
  }

  Serial.println("Attitude calibration complete!");
  Serial.print("Initial attitude - Roll: ");
  Serial.print(roll_IMU);
  Serial.print(" Pitch: ");
  Serial.print(pitch_IMU);
  Serial.print(" Yaw: ");
  Serial.println(yaw_IMU);
}

void switchRollYaw(int reverseRoll, int reverseYaw) {
  //DESCRIPTION: Switches roll_des and yaw_des variables for tailsitter-type configurations
  /*
   * Takes in two integers (either 1 or -1) corresponding to the desired reversing of the roll axis and yaw axis, respectively.
   * Reversing of the roll or yaw axis may be needed when switching between the two for some dynamic configurations. Inputs of 1, 1 does not
   * reverse either of them, while -1, 1 will reverse the output corresponding to the new roll axis.
   * This function may be replaced in the future by a function that switches the IMU data instead (so that angle can also be estimated with the
   * IMU tilted 90 degrees from default level).
   */
  float switch_holder;

  switch_holder = yaw_des;
  yaw_des = reverseYaw*roll_des;
  roll_des = reverseRoll*switch_holder;
}

void controlANGLE() {
  //DESCRIPTION: Computes control commands based on state error (angle)
  /*
   * Basic PID control to stablize on angle setpoint based on desired states roll_des, pitch_des, and yaw_des.
   * Error is simply the desired state minus the actual state (ex. roll_des - roll_IMU). Two safety features
   * are implimented here regarding the I terms. The I terms are saturated within specified limits on startup to prevent
   * excessive buildup. This can be seen by holding the vehicle at an angle and seeing the motors ramp up on one side until
   * they've maxed out throttle...saturating I to a specified limit fixes this. The second feature defaults the I terms to 0
   * if the throttle is at the minimum setting. This means the motors will not start spooling up on the ground, and the I
   * terms will always start from 0 on takeoff. This function updates the variables roll_PID, pitch_PID, and yaw_PID which
   * can be thought of as 1-D stablized signals. They are mixed to the configuration of the vehicle in controlMixer().
   */

  //Roll
  error_roll = roll_des - roll_IMU;
  integral_roll = integral_roll_prev + error_roll*dt;
  if (channel_1_pwm < 1060) {   //Don't let integrator build if throttle is too low
    integral_roll = 0;
  }
  integral_roll = constrain(integral_roll, -i_limit, i_limit); //Saturate integrator to prevent unsafe buildup
  derivative_roll = (roll_IMU - roll_IMU_prev)/dt;
  roll_PID = .01*(Kp_roll_angle*error_roll + Ki_roll_angle*integral_roll - Kd_roll_angle*derivative_roll); //Scaled by .01 to bring within -1 to 1 range

  //Pitch
  error_pitch = pitch_des - pitch_IMU;
  integral_pitch = integral_pitch_prev + error_pitch*dt;
  if (channel_1_pwm < 1060) {   //Don't let integrator build if throttle is too low
    integral_pitch = 0;
  }
  integral_pitch = constrain(integral_pitch, -i_limit, i_limit); //Saturate integrator to prevent unsafe buildup
  derivative_pitch = (pitch_IMU - pitch_IMU_prev)/dt;
  pitch_PID = .01*(Kp_pitch_angle*error_pitch + Ki_pitch_angle*integral_pitch - Kd_pitch_angle*derivative_pitch); //Scaled by .01 to bring within -1 to 1 range

  //Yaw, stabilize on rate from GyroZ
  error_yaw = yaw_des + GyroZ; // +GyroZ: CW spin gives negative GyroZ (Z-up, RHR), corrects to negative feedback
  integral_yaw = integral_yaw_prev + error_yaw*dt;
  if (channel_1_pwm < 1060) {   //Don't let integrator build if throttle is too low
    integral_yaw = 0;
  }
  integral_yaw = constrain(integral_yaw, -i_limit, i_limit); //Saturate integrator to prevent unsafe buildup
  derivative_yaw = (error_yaw - error_yaw_prev)/dt;
  yaw_PID = .01*(Kp_yaw*error_yaw + Ki_yaw*integral_yaw + Kd_yaw*derivative_yaw); //Scaled by .01 to bring within -1 to 1 range

  //Update roll variables
  error_roll_prev = error_roll;
  integral_roll_prev = integral_roll;
  roll_IMU_prev = roll_IMU;
  //Update pitch variables
  error_pitch_prev = error_pitch;
  integral_pitch_prev = integral_pitch;
  pitch_IMU_prev = pitch_IMU;
  //Update yaw variables
  error_yaw_prev = error_yaw;
  integral_yaw_prev = integral_yaw;
}

void controlRATE() {
  //DESCRIPTION: Computes control commands based on state error (rate)
  /*
   * This is the "acro mode" or "rate mode" PID controller. It controls the drone based on gyro rates instead of angles.
   * The error is the desired rate minus the actual rate (ex. roll_des - GyroX). This provides direct control over
   * the rotation rate and is preferred for aerobatic flying and experienced pilots.
   * Everything is the same as controlANGLE() except the error is now the desired rate - raw gyro reading.
   */

  //Roll
  error_roll = roll_des - GyroX;
  integral_roll = integral_roll_prev + error_roll*dt;
  if (channel_1_pwm < 1060) {   //Don't let integrator build if throttle is too low
    integral_roll = 0;
  }
  integral_roll = constrain(integral_roll, -i_limit, i_limit); //Saturate integrator to prevent unsafe buildup
  derivative_roll = (error_roll - error_roll_prev)/dt;
  roll_PID = .01*(Kp_roll_rate*error_roll + Ki_roll_rate*integral_roll + Kd_roll_rate*derivative_roll); //Scaled by .01 to bring within -1 to 1 range

  //Pitch
  error_pitch = pitch_des - GyroY;
  integral_pitch = integral_pitch_prev + error_pitch*dt;
  if (channel_1_pwm < 1060) {   //Don't let integrator build if throttle is too low
    integral_pitch = 0;
  }
  integral_pitch = constrain(integral_pitch, -i_limit, i_limit); //Saturate integrator to prevent unsafe buildup
  derivative_pitch = (error_pitch - error_pitch_prev)/dt;
  pitch_PID = .01*(Kp_pitch_rate*error_pitch + Ki_pitch_rate*integral_pitch + Kd_pitch_rate*derivative_pitch); //Scaled by .01 to bring within -1 to 1 range

  //Yaw, stabilize on rate from GyroZ
  error_yaw = yaw_des + GyroZ; // +GyroZ: CW spin gives negative GyroZ (Z-up, RHR), corrects to negative feedback
  integral_yaw = integral_yaw_prev + error_yaw*dt;
  if (channel_1_pwm < 1060) {   //Don't let integrator build if throttle is too low
    integral_yaw = 0;
  }
  integral_yaw = constrain(integral_yaw, -i_limit, i_limit); //Saturate integrator to prevent unsafe buildup
  derivative_yaw = (error_yaw - error_yaw_prev)/dt;
  yaw_PID = .01*(Kp_yaw*error_yaw + Ki_yaw*integral_yaw + Kd_yaw*derivative_yaw); //Scaled by .01 to bring within -1 to 1 range

  //Update roll variables
  error_roll_prev = error_roll;
  integral_roll_prev = integral_roll;
  GyroX_prev = GyroX;
  //Update pitch variables
  error_pitch_prev = error_pitch;
  integral_pitch_prev = integral_pitch;
  GyroY_prev = GyroY;
  //Update yaw variables
  error_yaw_prev = error_yaw;
  integral_yaw_prev = integral_yaw;
}

float invSqrt(float x) {
  //Fast inverse square root
  float halfx = 0.5f * x;
  float y = x;
  long i = *(long*)&y;
  i = 0x5f3759df - (i>>1);
  y = *(float*)&i;
  y = y * (1.5f - (halfx * y * y));
  return y;
}

float floatFaderLinear(float param, float param_min, float param_max, float fadeTime, int state, int loopFreq){
  //DESCRIPTION: Linearly fades a float type variable between min and max bounds based on desired high or low state and time
  /*
   *  Takes in a float variable, desired minimum and maximum bounds, fade time, high or low desired state, and the loop frequency
   *  and linearly interpolates that param variable between the maximum and minimum bounds. This function can be called in controlMixer()
   *  and high/low states can be determined by monitoring the state of an auxillary radio channel. For example, if channel_6_pwm is being
   *  monitored to switch between two dynamic configurations (hover and forward flight), this function can be called within the logical
   *  statements in order to fade controller gains, for example between the two dynamic configurations. The 'state' (1 or 0) can be used
   *  to designate the two final options for that control gain based on the dynamic configuration assignment to the auxillary radio channel.
   *
   *  Example usage:
   *  if (channel_6_pwm > 1500) {
   *    Kp_roll_angle = floatFaderLinear(Kp_roll_angle, 0.2, 0.5, 2.0, 1, 2000); // Fade to max over 2 seconds
   *  } else {
   *    Kp_roll_angle = floatFaderLinear(Kp_roll_angle, 0.2, 0.5, 2.0, 0, 2000); // Fade to min over 2 seconds
   *  }
   */
  float diffParam = (param_max - param_min)/(fadeTime*loopFreq); //Difference to add or subtract from param for each loop iteration for desired fadeTime

  if (state == 1) { //Maximum param bound desired, increase param by diffParam for each loop iteration
    param = param + diffParam;
  }
  else if (state == 0) { //Minimum param bound desired, decrease param by diffParam for each loop iteration
    param = param - diffParam;
  }

  param = constrain(param, param_min, param_max); //Constrain param within max bounds

  return param;
}

float floatFaderLinear2(float param, float param_des, float param_lower, float param_upper, float fadeTime_up, float fadeTime_down, int loopFreq){
  //DESCRIPTION: Linearly fades a float type variable from its current value to the desired value, up or down
  /*
   *  Takes in a float variable to be modified, desired new position, upper value, lower value, fade time up, fade time down, and the loop frequency
   *  and linearly fades that param variable up or down to the desired value. This function can be called in controlMixer()
   *  to fade up or down between flight modes monitored by an auxillary radio channel. For example, if channel_6_pwm is being
   *  monitored to switch between two dynamic configurations (hover and forward flight), this function can be called within the logical
   *  statements in order to fade controller gains, for example between the two dynamic configurations.
   *
   *  This version allows different fade times for increasing vs decreasing the parameter.
   *
   *  Example usage:
   *  if (channel_6_pwm > 1500) {
   *    Kp_roll_angle = floatFaderLinear2(Kp_roll_angle, 0.5, 0.2, 0.5, 2.0, 1.0, 2000); // Fade to 0.5 (fast down, slow up)
   *  } else {
   *    Kp_roll_angle = floatFaderLinear2(Kp_roll_angle, 0.2, 0.2, 0.5, 2.0, 1.0, 2000); // Fade to 0.2
   *  }
   */
  if (param > param_des) { //Need to fade down to get to desired
    float diffParam = (param_upper - param_des)/(fadeTime_down*loopFreq);
    param = param - diffParam;
  }
  else if (param < param_des) { //Need to fade up to get to desired
    float diffParam = (param_des - param_lower)/(fadeTime_up*loopFreq);
    param = param + diffParam;
  }

  param = constrain(param, param_lower, param_upper); //Constrain param within max bounds

  return param;
}

void Madgwick6DOF(float gx, float gy, float gz, float ax, float ay, float az, float invSampleFreq) {
  //DESCRIPTION: 6DOF Madgwick filter for attitude estimation
  
  float recipNorm;
  float s0, s1, s2, s3;
  float qDot1, qDot2, qDot3, qDot4;
  float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

  //Convert gyroscope degrees/sec to radians/sec
  gx *= 0.0174533f;
  gy *= 0.0174533f;
  gz *= 0.0174533f;

  //Rate of change of quaternion from gyroscope
  qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
  qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
  qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
  qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

  //Compute feedback only if accelerometer measurement valid
  if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
    //Normalise accelerometer measurement
    recipNorm = invSqrt(ax * ax + ay * ay + az * az);
    ax *= recipNorm;
    ay *= recipNorm;
    az *= recipNorm;

    //Auxiliary variables to avoid repeated arithmetic
    _2q0 = 2.0f * q0;
    _2q1 = 2.0f * q1;
    _2q2 = 2.0f * q2;
    _2q3 = 2.0f * q3;
    _4q0 = 4.0f * q0;
    _4q1 = 4.0f * q1;
    _4q2 = 4.0f * q2;
    _8q1 = 8.0f * q1;
    _8q2 = 8.0f * q2;
    q0q0 = q0 * q0;
    q1q1 = q1 * q1;
    q2q2 = q2 * q2;
    q3q3 = q3 * q3;

    //Gradient descent algorithm corrective step
    s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
    s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
    s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
    s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
    recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
    s0 *= recipNorm;
    s1 *= recipNorm;
    s2 *= recipNorm;
    s3 *= recipNorm;

    //Apply feedback step
    qDot1 -= B_madgwick * s0;
    qDot2 -= B_madgwick * s1;
    qDot3 -= B_madgwick * s2;
    qDot4 -= B_madgwick * s3;
  }

  //Integrate rate of change of quaternion
  q0 += qDot1 * invSampleFreq;
  q1 += qDot2 * invSampleFreq;
  q2 += qDot3 * invSampleFreq;
  q3 += qDot4 * invSampleFreq;

  //Normalise quaternion
  recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 *= recipNorm;
  q1 *= recipNorm;
  q2 *= recipNorm;
  q3 *= recipNorm;

  //Compute angles
  roll_IMU = atan2(q0*q1 + q2*q3, 0.5f - q1*q1 - q2*q2)*57.29577951;
  pitch_IMU = -asin(constrain(-2.0f * (q1*q3 - q0*q2),-0.999999,0.999999))*57.29577951;
  yaw_IMU = -atan2(q1*q2 + q0*q3, 0.5f - q2*q2 - q3*q3)*57.29577951;
}

void Madgwick(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz, float invSampleFreq) {
  //DESCRIPTION: Madgwick filter - use 6DOF for MPU6050
  Madgwick6DOF(gx, gy, gz, ax, ay, az, invSampleFreq);
}

void getDesState() {
  //DESCRIPTION: Normalize radio commands to desired states

  #if defined USE_MTF01_OPTICAL_FLOW
  // FIXED: Simplified throttle logic for altitude hold
  // When altitude hold OFF: Use manual throttle from web controller
  // When altitude hold ON + airborne (>20cm): Use fixed hover baseline (50%)
  if (altitude_hold_enabled && altitude_current > 200.0) {
    // Altitude hold active: Use fixed hover throttle baseline
    // Altitude PID will add corrections (±0.3) on top of this
    thro_des = 0.50;  // 50% hover baseline for 50g brushed quad
  } else {
    // Manual control or below 20cm: Use joystick throttle
    thro_des = (channel_1_pwm - 1000.0)/1000.0;
  }
  #else
  // No MTF-01: Always use manual throttle
  thro_des = (channel_1_pwm - 1000.0)/1000.0;
  #endif

  roll_des = (channel_2_pwm - 1500.0)/500.0;
  pitch_des = (channel_3_pwm - 1500.0)/500.0;
  yaw_des = (channel_4_pwm - 1500.0)/500.0; // Direct yaw input (no deadband, like Flix)

  roll_passthru = roll_des/2.0;
  pitch_passthru = pitch_des/2.0;
  yaw_passthru = yaw_des/2.0;

  #if defined USE_TOF_SENSORS
  //Add altitude hold correction to throttle
  if (altitude_hold_enabled) {
    thro_des += altitude_correction;
  }

  //Apply obstacle avoidance
  if (obstacle_avoid_enabled) {
    applyObstacleAvoidance(roll_des, pitch_des, thro_des);
  }
  #endif

  #if defined USE_MTF01_OPTICAL_FLOW
  //Add altitude hold correction to throttle (using MTF-01 rangefinder)
  if (altitude_hold_enabled) {
    thro_des += altitude_correction;
  }

  //Apply position hold corrections to roll/pitch (using MTF-01 optical flow)
  if (position_hold_enabled) {
    float roll_correction = 0.0;
    float pitch_correction = 0.0;
    applyPositionHold(dt, roll_correction, pitch_correction);
    roll_des += roll_correction;
    pitch_des += pitch_correction;
  }
  #endif

  // ---- LiteWing altitude + position hold (Kalman + 2-stage PID) ----
  #if defined USE_LITEWING_MODULE
  // Altitude hold: thro_des = hover_throttle (thrustBase) + pidVZ correction
  if (lw_althold_enabled && kc_S[KC_STATE_Z] > 0.05f) {
    thro_des = lw_hover_throttle + lw_thrust_correction;
  }
  // Position hold: add normalized roll/pitch corrections (scaled to degrees at constrain below)
  if (lw_poshold_active && kc_S[KC_STATE_Z] > 0.10f) {
    roll_des  += lw_roll_correction;
    pitch_des += lw_pitch_correction;
  }
  #endif

  //Constrain within normalized bounds
  thro_des = constrain(thro_des, 0.0, 1.0);
  roll_des = constrain(roll_des, -1.0, 1.0)*maxRoll;
  pitch_des = constrain(pitch_des, -1.0, 1.0)*maxPitch;
  yaw_des = constrain(yaw_des, -1.0, 1.0)*maxYaw;
  roll_passthru = constrain(roll_passthru, -0.5, 0.5);
  pitch_passthru = constrain(pitch_passthru, -0.5, 0.5);
  yaw_passthru = constrain(yaw_passthru, -0.5, 0.5);
}

void controlANGLE2() {
  //DESCRIPTION: Cascaded PID controller (outer loop: angle, inner loop: rate)
  
  //Outer loop - PID on angle
  float roll_des_ol, pitch_des_ol;
  
  //Roll
  error_roll = roll_des - roll_IMU;
  integral_roll_ol = integral_roll_prev_ol + error_roll*dt;
  if (channel_1_pwm < 1060) {
    integral_roll_ol = 0;
  }
  integral_roll_ol = constrain(integral_roll_ol, -i_limit, i_limit);
  derivative_roll = (roll_IMU - roll_IMU_prev)/dt; 
  roll_des_ol = Kp_roll_angle*error_roll + Ki_roll_angle*integral_roll_ol;

  //Pitch
  error_pitch = pitch_des - pitch_IMU;
  integral_pitch_ol = integral_pitch_prev_ol + error_pitch*dt;
  if (channel_1_pwm < 1060) {
    integral_pitch_ol = 0;
  }
  integral_pitch_ol = constrain(integral_pitch_ol, -i_limit, i_limit);
  derivative_pitch = (pitch_IMU - pitch_IMU_prev)/dt;
  pitch_des_ol = Kp_pitch_angle*error_pitch + Ki_pitch_angle*integral_pitch_ol;

  //Apply loop gain and LP filter
  float Kl = 30.0;
  roll_des_ol = Kl*roll_des_ol;
  pitch_des_ol = Kl*pitch_des_ol;
  roll_des_ol = constrain(roll_des_ol, -240.0, 240.0);
  pitch_des_ol = constrain(pitch_des_ol, -240.0, 240.0);
  roll_des_ol = (1.0 - B_loop_roll)*roll_des_prev + B_loop_roll*roll_des_ol;
  pitch_des_ol = (1.0 - B_loop_pitch)*pitch_des_prev + B_loop_pitch*pitch_des_ol;

  //Inner loop - PID on rate
  //Roll
  error_roll = roll_des_ol - GyroX;
  integral_roll_il = integral_roll_prev_il + error_roll*dt;
  if (channel_1_pwm < 1060) {
    integral_roll_il = 0;
  }
  integral_roll_il = constrain(integral_roll_il, -i_limit, i_limit);
  derivative_roll = (error_roll - error_roll_prev)/dt; 
  roll_PID = .01*(Kp_roll_rate*error_roll + Ki_roll_rate*integral_roll_il + Kd_roll_rate*derivative_roll);

  //Pitch
  error_pitch = pitch_des_ol - GyroY;
  integral_pitch_il = integral_pitch_prev_il + error_pitch*dt;
  if (channel_1_pwm < 1060) {
    integral_pitch_il = 0;
  }
  integral_pitch_il = constrain(integral_pitch_il, -i_limit, i_limit);
  derivative_pitch = (error_pitch - error_pitch_prev)/dt; 
  pitch_PID = .01*(Kp_pitch_rate*error_pitch + Ki_pitch_rate*integral_pitch_il + Kd_pitch_rate*derivative_pitch);
  
  //Yaw
  error_yaw = yaw_des + GyroZ; // +GyroZ: CW spin gives negative GyroZ (Z-up, RHR), corrects to negative feedback
  integral_yaw = integral_yaw_prev + error_yaw*dt;

  // Reset yaw integral in multiple conditions to prevent windup
  if (channel_1_pwm < 1060 || abs(yaw_des) < 0.01) {
    // Reset when: throttle low OR yaw stick centered (within deadband)
    integral_yaw = 0;
  }

  // Tighter integral limit for yaw to prevent excessive windup
  integral_yaw = constrain(integral_yaw, -i_limit/2.0, i_limit/2.0);  // Half the limit of roll/pitch

  derivative_yaw = (error_yaw - error_yaw_prev)/dt;
  yaw_PID = .01*(Kp_yaw*error_yaw + Ki_yaw*integral_yaw + Kd_yaw*derivative_yaw);
  
  //Update variables
  integral_roll_prev_ol = integral_roll_ol;
  integral_roll_prev_il = integral_roll_il;
  error_roll_prev = error_roll;
  roll_IMU_prev = roll_IMU;
  roll_des_prev = roll_des_ol;
  integral_pitch_prev_ol = integral_pitch_ol;
  integral_pitch_prev_il = integral_pitch_il;
  error_pitch_prev = error_pitch;
  pitch_IMU_prev = pitch_IMU;
  pitch_des_prev = pitch_des_ol;
  error_yaw_prev = error_yaw;
  integral_yaw_prev = integral_yaw;
}

void scaleCommands() {
  //DESCRIPTION: Scale normalized commands to PWM values (0-255)
  //Motor compensation is applied AFTER base calculation so it affects idle speed too

  const int DUTY_MAX = 255;   // Maximum duty cycle

  // Calculate base PWM values (DUTY_IDLE is now global and adjustable via UI)
  m1_command_PWM = DUTY_IDLE + m1_command_scaled * (DUTY_MAX - DUTY_IDLE);
  m2_command_PWM = DUTY_IDLE + m2_command_scaled * (DUTY_MAX - DUTY_IDLE);
  m3_command_PWM = DUTY_IDLE + m3_command_scaled * (DUTY_MAX - DUTY_IDLE);
  m4_command_PWM = DUTY_IDLE + m4_command_scaled * (DUTY_MAX - DUTY_IDLE);

  // Constrain base PWM BEFORE applying compensation
  m1_command_PWM = constrain(m1_command_PWM, DUTY_IDLE, DUTY_MAX);
  m2_command_PWM = constrain(m2_command_PWM, DUTY_IDLE, DUTY_MAX);
  m3_command_PWM = constrain(m3_command_PWM, DUTY_IDLE, DUTY_MAX);
  m4_command_PWM = constrain(m4_command_PWM, DUTY_IDLE, DUTY_MAX);

  // Apply motor compensation for torque balance (affects idle speed too)
  // If compensation is 0.0, motor will get 0 PWM and stop completely
  m1_command_PWM *= motor_comp_m1;  //Front-Left CCW
  m2_command_PWM *= motor_comp_m2;  //Front-Right CW
  m3_command_PWM *= motor_comp_m3;  //Back-Right CCW
  m4_command_PWM *= motor_comp_m4;  //Back-Left CW

  // Final constrain to safe range (allow 0 if compensation is 0)
  m1_command_PWM = constrain(m1_command_PWM, 0, DUTY_MAX);
  m2_command_PWM = constrain(m2_command_PWM, 0, DUTY_MAX);
  m3_command_PWM = constrain(m3_command_PWM, 0, DUTY_MAX);
  m4_command_PWM = constrain(m4_command_PWM, 0, DUTY_MAX);
}

void getCommands() {
  //DESCRIPTION: Get and filter radio commands
  
  //Low-pass filter critical commands
  float b = 0.7;
  channel_1_pwm = (1.0 - b)*channel_1_pwm_prev + b*channel_1_pwm;
  channel_2_pwm = (1.0 - b)*channel_2_pwm_prev + b*channel_2_pwm;
  channel_3_pwm = (1.0 - b)*channel_3_pwm_prev + b*channel_3_pwm;
  channel_4_pwm = (1.0 - b)*channel_4_pwm_prev + b*channel_4_pwm;
  channel_1_pwm_prev = channel_1_pwm;
  channel_2_pwm_prev = channel_2_pwm;
  channel_3_pwm_prev = channel_3_pwm;
  channel_4_pwm_prev = channel_4_pwm;
}

void failSafe() {
  //DESCRIPTION: Set failsafe values if radio gives bad data
  
  unsigned minVal = 800;
  unsigned maxVal = 2200;
  int check1 = 0, check2 = 0, check3 = 0, check4 = 0, check5 = 0, check6 = 0;

  if (channel_1_pwm > maxVal || channel_1_pwm < minVal) check1 = 1;
  if (channel_2_pwm > maxVal || channel_2_pwm < minVal) check2 = 1;
  if (channel_3_pwm > maxVal || channel_3_pwm < minVal) check3 = 1;
  if (channel_4_pwm > maxVal || channel_4_pwm < minVal) check4 = 1;
  if (channel_5_pwm > maxVal || channel_5_pwm < minVal) check5 = 1;
  if (channel_6_pwm > maxVal || channel_6_pwm < minVal) check6 = 1;

  if ((check1 + check2 + check3 + check4 + check5 + check6) > 0) {
    channel_1_pwm = channel_1_fs;
    channel_2_pwm = channel_2_fs;
    channel_3_pwm = channel_3_fs;
    channel_4_pwm = channel_4_fs;
    channel_5_pwm = channel_5_fs;
    channel_6_pwm = channel_6_fs;
  }
}

void commandMotors() {
  //DESCRIPTION: Send PWM signals to motors using ESP32 LEDC (4 motors)

  ledcWrite(0, m1_command_PWM); //Motor 1 (Front Right)
  ledcWrite(1, m2_command_PWM); //Motor 2 (Back Right)
  ledcWrite(2, m3_command_PWM); //Motor 3 (Back Left)
  ledcWrite(3, m4_command_PWM); //Motor 4 (Front Left)
}

void armMotors() {
  //DESCRIPTION: Initialize motors to zero state
  
  for (int i = 0; i <= 10; i++) {
    commandMotors();
    delay(2);
  }
}

void throttleCut() {
  //DESCRIPTION: Set motors to zero if disarmed or not ready

  if ((channel_5_pwm > 1500) || (armedFly == false)) {
    armedFly = false;
    m1_command_PWM = 0;
    m2_command_PWM = 0;
    m3_command_PWM = 0;
    m4_command_PWM = 0;
  }
}

void loopRate(int freq) {
  //DESCRIPTION: Regulate loop rate to specified frequency
  
  float invFreq = 1.0/freq*1000000.0;
  unsigned long checker = micros();
  
  while (invFreq > (checker - current_time)) {
    checker = micros();
  }
}

void loopBlink() {
  //DESCRIPTION: Blink LED to indicate main loop running
  
  if (current_time - blink_counter > blink_delay) {
    blink_counter = micros();
    digitalWrite(LED_PIN, blinkAlternate);
    
    if (blinkAlternate == 1) {
      blinkAlternate = 0;
      blink_delay = 100000;
    }
    else if (blinkAlternate == 0) {
      blinkAlternate = 1;
      blink_delay = 2000000;
    }
  }
}

void setupBlink(int numBlinks, int upTime, int downTime) {
  //DESCRIPTION: Blink LED during setup
  
  for (int j = 1; j<= numBlinks; j++) {
    digitalWrite(LED_PIN, LOW);
    delay(downTime);
    digitalWrite(LED_PIN, HIGH);
    delay(upTime);
  }
}


//========================================================================================================================//
//                                              DEBUGGING PRINT FUNCTIONS                                                //
//========================================================================================================================//

void printRadioData() {
  //DESCRIPTION: Print radio PWM values
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F(" CH1:"));
    Serial.print(channel_1_pwm);
    Serial.print(F(" CH2:"));
    Serial.print(channel_2_pwm);
    Serial.print(F(" CH3:"));
    Serial.print(channel_3_pwm);
    Serial.print(F(" CH4:"));
    Serial.print(channel_4_pwm);
    Serial.print(F(" CH5:"));
    Serial.print(channel_5_pwm);
    Serial.print(F(" CH6:"));
    Serial.println(channel_6_pwm);
  }
}

void printDesiredState() {
  //DESCRIPTION: Print desired states (throttle, roll, pitch, yaw)
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("thro_des:"));
    Serial.print(thro_des);
    Serial.print(F(" roll_des:"));
    Serial.print(roll_des);
    Serial.print(F(" pitch_des:"));
    Serial.print(pitch_des);
    Serial.print(F(" yaw_des:"));
    Serial.println(yaw_des);
  }
}

void printGyroData() {
  //DESCRIPTION: Print gyroscope data (deg/s)
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("GyroX:"));
    Serial.print(GyroX);
    Serial.print(F(" GyroY:"));
    Serial.print(GyroY);
    Serial.print(F(" GyroZ:"));
    Serial.println(GyroZ);
  }
}

void printAccelData() {
  //DESCRIPTION: Print accelerometer data (G's)
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("AccX:"));
    Serial.print(AccX);
    Serial.print(F(" AccY:"));
    Serial.print(AccY);
    Serial.print(F(" AccZ:"));
    Serial.println(AccZ);
  }
}

void printRollPitchYaw() {
  //DESCRIPTION: Print attitude angles (degrees)
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("roll:"));
    Serial.print(roll_IMU);
    Serial.print(F(" pitch:"));
    Serial.print(pitch_IMU);
    Serial.print(F(" yaw:"));
    Serial.println(yaw_IMU);
  }
}

void printPIDoutput() {
  //DESCRIPTION: Print PID controller outputs
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("roll_PID:"));
    Serial.print(roll_PID);
    Serial.print(F(" pitch_PID:"));
    Serial.print(pitch_PID);
    Serial.print(F(" yaw_PID:"));
    Serial.println(yaw_PID);
  }
}

void printMotorCommands() {
  //DESCRIPTION: Print motor PWM commands (0-255)
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("m1:"));
    Serial.print(m1_command_PWM);
    Serial.print(F(" m2:"));
    Serial.print(m2_command_PWM);
    Serial.print(F(" m3:"));
    Serial.print(m3_command_PWM);
    Serial.print(F(" m4:"));
    Serial.println(m4_command_PWM);
  }
}

void printLoopRate() {
  //DESCRIPTION: Print main loop rate (microseconds)
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("dt(us):"));
    Serial.println(dt*1000000.0);
  }
}

#if defined USE_MTF01_OPTICAL_FLOW
void printOpticalFlowData() {
  //DESCRIPTION: Print MTF-01 optical flow sensor data
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("FlowX:"));
    Serial.print(flow_x);
    Serial.print(F(" FlowY:"));
    Serial.print(flow_y);
    Serial.print(F(" Quality:"));
    Serial.print(flow_quality);
    Serial.print(F(" Range(mm):"));
    Serial.print(range_mm);
    Serial.print(F(" PosX(cm):"));
    Serial.print(position_x);
    Serial.print(F(" PosY(cm):"));
    Serial.println(position_y);
  }
}

void printAltitudeData() {
  //DESCRIPTION: Print altitude hold data
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("Alt(mm):"));
    Serial.print(altitude_current);
    Serial.print(F(" Target:"));
    Serial.print(altitude_setpoint);
    Serial.print(F(" Error:"));
    Serial.print(altitude_error);
    Serial.print(F(" Corr:"));
    Serial.print(altitude_correction);
    Serial.print(F(" Enabled:"));
    Serial.println(altitude_hold_enabled ? "YES" : "NO");
  }
}

void printPositionData() {
  //DESCRIPTION: Print position hold data
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("Pos X:"));
    Serial.print(position_x);
    Serial.print(F(" Y:"));
    Serial.print(position_y);
    Serial.print(F(" Target X:"));
    Serial.print(position_x_setpoint);
    Serial.print(F(" Y:"));
    Serial.print(position_y_setpoint);
    Serial.print(F(" Enabled:"));
    Serial.println(position_hold_enabled ? "YES" : "NO");
  }
}
#endif

#if defined USE_TOF_SENSORS
void printToFReadings() {
  //DESCRIPTION: Print ToF sensor readings
  if (current_time - print_counter > 10000) {
    print_counter = micros();
    Serial.print(F("ToF(mm) - R:"));
    Serial.print(distance_right);
    Serial.print(F(" F:"));
    Serial.print(distance_front);
    Serial.print(F(" L:"));
    Serial.print(distance_left);
    Serial.print(F(" B:"));
    Serial.print(distance_back);
    Serial.print(F(" T:"));
    Serial.print(distance_top);
    Serial.print(F(" Bot:"));
    Serial.println(distance_bottom);
  }
}
#endif
