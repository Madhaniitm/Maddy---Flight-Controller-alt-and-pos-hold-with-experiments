"""
test_transmitter_usb.py
=======================
Quick test: send commands to ESP32-C3 over USB and verify response.
Run BEFORE connecting to the drone.

Usage:
    /opt/homebrew/bin/python3.11 test_transmitter_usb.py

Find your port first:
    ls /dev/cu.*          # macOS
    ls /dev/ttyUSB*       # Linux
"""

import serial, time, sys

PORT      = "/dev/cu.usbmodem101"   # change to your port
BAUD      = 115200
DELAY     = 2.0   # seconds between commands (RC filter needs ~50ms but give more for safety)


def send(ser: serial.Serial, cmd: str) -> str:
    ser.write((cmd + "\n").encode())
    time.sleep(0.1)
    resp = ser.read_all().decode(errors="ignore").strip()
    print(f"  → {cmd:12s}  ← {resp}")
    return resp


def main():
    print(f"Connecting to {PORT} at {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        print("Available ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  —  {p.description}")
        sys.exit(1)

    time.sleep(2)  # wait for ESP32 boot
    ser.read_all()  # clear boot messages
    print("Connected.\n")

    print("── Status check ──────────────────────────────")
    send(ser, "STATUS")

    print("\n── Preset command test (watch serial monitor) ─")
    for cmd in ["HOVER", "FORWARD", "HOVER", "BACK", "HOVER",
                "LEFT", "HOVER", "RIGHT", "HOVER"]:
        time.sleep(DELAY)
        send(ser, cmd)

    print("\n── Raw value test ────────────────────────────")
    for cmd in ["P:700", "P:512", "R:700", "R:512", "T:500", "T:100"]:
        time.sleep(DELAY)
        send(ser, cmd)

    send(ser, "STATUS")
    ser.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
