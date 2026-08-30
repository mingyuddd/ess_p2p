#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import asyncio
import threading
import time

# --- 协议算法区 (完全保留你的 sub.py 逻辑) ---
def make_length_4(s: str) -> str:
    return s.zfill(4)

def cacu(x, y):
    o = 3200 / 440
    a = max(-77, min(x, 363))
    e = max(-77, min(y, 363))
    n = int(((286 - a) + 77) * o + 516)
    i = int(((286 - e) + 77) * o + 516)
    d = make_length_4(hex(n)[2:])
    c = make_length_4(hex(i)[2:])
    return d, c

def ang_cal(angle, r):
    y = 143
    x = 143 + angle
    x = max(43,min(x,243))
    if(angle == 1000): x = 363
    if(angle == -1000): x = -77
    if(r == 0) : y = 143
    elif (r == 35): y = 75
    else: y = 20
    return x, y

def ackermann_to_differential_angle(v, ang):
    wheel_pos = 70 * v
    wheel_ang = ang * 2.0
    return wheel_ang, wheel_pos

def encode_control(v, ang):
    # [参数微调] 这里可能需要根据实车调整比例系数
    # 假设 Controller 输出的 ang 是弧度/秒，可能需要放大
    # target_ang = ang * 50.0 
    wheel_angle, wheel_pos = ackermann_to_differential_angle(v, ang)
    x, y = ang_cal(wheel_angle, wheel_pos)
    d, c = cacu(x, y)
    return "EB900FA2AA" + d[-2:] + d[:2] + c[-2:] + c[:2] + "0005" + "CC33C33C"
# -------------------------------------------

try:
    import bleak
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False

class HardwareBridge(Node):
    def __init__(self):
        super().__init__('hardware_bridge')
        
        # 参数声明
        self.declare_parameter('ble_address', "11:89:88:11:A1:0C")
        self.declare_parameter('ble_uuid', "0000000f-0000-1000-8000-00805f9b34fb") # 替换为你的写特征UUID
        self.declare_parameter('send_rate', 20.0) # Hz
        
        # [关键] 共享状态变量 (线程安全)
        # 我们不需要锁，因为 float 在 Python 中是原子的，且我们只做简单的读写
        self.latest_v = 0.0
        self.latest_w = 0.0
        self.last_cmd_time = time.time()
        
        # 订阅
        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        
        # 启动异步蓝牙线程
        self.ble_thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.ble_thread.start()
        
        self.get_logger().info("Hardware Bridge Ready. Mode: Latest-Sample (No Latency)")

    def cmd_cb(self, msg: Twist):
        # 只是更新变量，不做耗时操作
        self.latest_v = msg.linear.x
        self.latest_w = msg.angular.z
        self.last_cmd_time = time.time()

    def start_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_task())

    async def ble_task(self):
        if not BLE_AVAILABLE:
            self.get_logger().error("Bleak not installed!")
            return

        addr = self.get_parameter('ble_address').value
        # 如果不知道UUID，可以用 int 句柄，但 bleak 新版建议用 UUID
        char_uuid = 0x000F # self.get_parameter('ble_uuid').value 
        rate = self.get_parameter('send_rate').value
        period = 1.0 / rate

        while rclpy.ok():
            try:
                self.get_logger().info(f"Connecting to {addr}...")
                async with bleak.BleakClient(addr) as client:
                    self.get_logger().info("Connected! Sending commands...")
                    
                    while client.is_connected and rclpy.ok():
                        loop_start = asyncio.get_event_loop().time()
                        
                        # 1. 安全检查：如果超过 1秒 没收到 ROS 指令，自动停车
                        if time.time() - self.last_cmd_time > 1.0:
                            self.latest_v = 0.0
                            self.latest_w = 0.0
                        
                        # 2. 采样最新值 -> 编码
                        hex_str = encode_control(self.latest_v, self.latest_w)
                        data = bytes.fromhex(hex_str)
                        
                        # 3. 发送 (Fire and Forget)
                        # write_gatt_char 默认是 response=True，会等待设备由于 ACK
                        # 如果为了极低延迟，可以尝试 response=False (write without response)
                        # 但这需要你的蓝牙设备支持 Write Without Response
                        try:
                            # 这里的 15 对应 0x000F
                            await client.write_gatt_char(15, data, response=True)
                        except Exception as e:
                            self.get_logger().warn(f"Write failed: {e}")
                        
                        # 4. 控频
                        elapsed = asyncio.get_event_loop().time() - loop_start
                        sleep_time = max(0.0, period - elapsed)
                        await asyncio.sleep(sleep_time)

            except Exception as e:
                self.get_logger().error(f"BLE Disconnected or Error: {e}, retrying...")
                await asyncio.sleep(2.0)

def main():
    rclpy.init()
    node = HardwareBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()