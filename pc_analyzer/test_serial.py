import serial, time
ser = serial.Serial("COM12", 115200, timeout=3)
time.sleep(2)
raw = ser.read(8192)
ser.close()
idx = raw.find(b"\xaa\x55")
if idx >= 0:
    frame = raw[idx:]
    dlen = frame[2] | (frame[3] << 8)
    bins = dlen // 2
    n = raw.count(b"\xaa\x55")
    print(f"SUCCESS: {n} frames, {bins} bins/frame, {len(raw)} bytes total")
else:
    print(f"No sync: {len(raw)} bytes")
    if len(raw):
        print(f"First 50: {raw[:50].hex()}")
