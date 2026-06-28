"""
串口数据接收模块

从STM32 F1精英版接收ADC采样数据，协议:
  帧头: 0xAA 0x55
  长度: 2字节 (小端, 数据区字节数)
  数据: N × uint16_t (小端, ADC原始值 0~4095)
  校验: 1字节 (数据区所有字节XOR)

用法:
    python serial_receiver.py COM3          # Windows
    python serial_receiver.py /dev/ttyUSB0  # Linux
    python serial_receiver.py --list        # 列出可用串口
"""

import argparse
import struct
import sys
import time
import threading
from collections import deque
from typing import Optional, List, Tuple
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("需要 pyserial: pip install pyserial", file=sys.stderr)
    sys.exit(1)

# ============================================================
# 协议常量
# ============================================================
SYNC1 = 0xAA
SYNC2 = 0x55
HEADER_SIZE = 4           # 0xAA + 0x55 + len_lo + len_hi
CHECKSUM_SIZE = 1
FRAME_OVERHEAD = HEADER_SIZE + CHECKSUM_SIZE  # 5 bytes


class SerialReceiver:
    """STM32串口数据接收器"""

    def __init__(self, port: str, baudrate: int = 500000, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._buf = bytearray()
        self._frame_queue: deque = deque(maxlen=256)
        self._lock = threading.Lock()
        self._frame_count = 0
        self._error_count = 0
        self._bytes_received = 0

    def connect(self) -> bool:
        """打开串口"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            print(f"串口已连接: {self.port} @ {self.baudrate} bps", file=sys.stderr)
            return True
        except serial.SerialException as e:
            print(f"串口连接失败: {e}", file=sys.stderr)
            return False

    def disconnect(self):
        """关闭串口"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭", file=sys.stderr)

    def start(self):
        """启动后台接收线程"""
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("请先 connect()")
        self.running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        print("后台接收线程已启动", file=sys.stderr)

    def _receive_loop(self):
        """后台接收循环（在独立线程中运行）"""
        while self.running:
            try:
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting)
                    self._bytes_received += len(data)
                    self._buf.extend(data)
                    self._parse_frames()
                else:
                    time.sleep(0.001)
            except serial.SerialException as e:
                print(f"串口读取错误: {e}", file=sys.stderr)
                break

    def _parse_frames(self):
        """从缓冲区解析完整帧"""
        while True:
            # 找帧头 0xAA 0x55
            sync_idx = self._buf.find(bytes([SYNC1, SYNC2]))
            if sync_idx < 0:
                # 没有帧头, 但可能最后一个是0xAA, 保留它
                if len(self._buf) > 0 and self._buf[-1] == SYNC1:
                    self._buf = self._buf[-1:]
                else:
                    self._buf.clear()
                return

            # 丢弃帧头前的垃圾
            if sync_idx > 0:
                self._buf = self._buf[sync_idx:]

            # 需要至少 HEADER_SIZE 字节来确定帧长度
            if len(self._buf) < HEADER_SIZE:
                return

            # 读取数据长度
            data_len = self._buf[2] | (self._buf[3] << 8)
            if data_len < 2 or data_len > 2048:
                # 数据长度异常，跳过这个0xAA 0x55
                self._buf = self._buf[2:]
                self._error_count += 1
                continue

            frame_total = FRAME_OVERHEAD + data_len
            if len(self._buf) < frame_total:
                return  # 数据还没收完

            # 提取帧
            frame = bytes(self._buf[:frame_total])

            # 校验
            data_start = HEADER_SIZE
            data_end = HEADER_SIZE + data_len
            expected_checksum = self._buf[data_end]
            actual_checksum = 0
            for b in self._buf[data_start:data_end]:
                actual_checksum ^= b

            if actual_checksum == expected_checksum:
                # 解析数据
                samples = []
                for i in range(0, data_len, 2):
                    lo = frame[data_start + i]
                    hi = frame[data_start + i + 1]
                    samples.append(lo | (hi << 8))

                with self._lock:
                    self._frame_queue.append(samples)
                    self._frame_count += 1
            else:
                self._error_count += 1

            # 移除已处理的帧
            self._buf = self._buf[frame_total:]

    def get_frame(self, timeout: float = 5.0) -> Optional[List[int]]:
        """
        获取一帧数据（阻塞）

        Returns:
            ADC采样值列表 [uint16, ...]，或None（超时）
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame_queue:
                    return self._frame_queue.popleft()
            time.sleep(0.005)
        return None

    def get_frame_nonblock(self) -> Optional[List[int]]:
        """获取一帧数据（非阻塞）"""
        with self._lock:
            if self._frame_queue:
                return self._frame_queue.popleft()
        return None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._frame_queue)

    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "frames_received": self.frame_count,
            "errors": self.error_count,
            "bytes_received": self._bytes_received,
            "queue_size": self.queue_size,
        }


def list_ports():
    """列出可用串口"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未发现串口设备")
        return

    print("可用串口:")
    for p in ports:
        print(f"  {p.device:20s} {p.description}")


def frame_to_samples(frame: List[int], dc_offset: int = 2048) -> np.ndarray:
    """
    将ADC原始值转为归一化浮点采样

    Args:
        frame: ADC原始值列表 [0~4095]
        dc_offset: 直流偏置 (MAX9814@3.3V → 2048)

    Returns:
        float数组, 归一化到[-1, 1]
    """
    samples = np.array(frame, dtype=np.float32)
    samples = (samples - dc_offset) / 2048.0  # -1 ~ +1
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STM32串口数据接收器")
    parser.add_argument("port", nargs="?", help="串口设备 (如 COM3, /dev/ttyUSB0)")
    parser.add_argument("--list", action="store_true", help="列出可用串口")
    parser.add_argument("--baud", type=int, default=500000, help="波特率 (默认500000)")
    parser.add_argument("--timeout", type=float, default=10.0, help="接收超时(秒)")
    parser.add_argument("--count", type=int, default=0, help="接收帧数 (0=无限)")
    args = parser.parse_args()

    if args.list:
        list_ports()
        sys.exit(0)

    if not args.port:
        parser.print_help()
        list_ports()
        sys.exit(1)

    # 连接
    rx = SerialReceiver(args.port, args.baud)
    if not rx.connect():
        sys.exit(1)

    rx.start()

    print(f"\n等待STM32数据... (按Ctrl+C停止)")
    print(f"协议: [0xAA 0x55] [len×2B] [N×uint16] [XOR]")
    print()

    try:
        received = 0
        deadline = time.time() + args.timeout

        while True:
            if args.count > 0 and received >= args.count:
                break
            if time.time() > deadline:
                break

            frame = rx.get_frame(timeout=1.0)
            if frame:
                received += 1
                deadline = time.time() + args.timeout  # 重置超时

                # 打印帧信息
                samples = np.array(frame, dtype=np.float32)
                rms = np.sqrt(np.mean((samples - 2048) ** 2))
                print(f"[{received:4d}] {len(frame)} samples | "
                      f"min={min(frame):4d} max={max(frame):4d} "
                      f"avg={np.mean(samples):6.1f} rms={rms:6.1f}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        stats = rx.stats()
        print(f"\n统计: {stats['frames_received']} 帧接收, "
              f"{stats['errors']} 错误, "
              f"{stats['bytes_received']} 字节")
        rx.disconnect()
