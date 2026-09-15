import rtde_control
import rtde_receive


class Robot():
    def __init__(self, locator):
        # 连接到机械臂
        self.robot_ip = "192.168.101.101"
        self.rtde_c = None
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

    def getPose(self):
        pose = self.rtde_r.getActualTCPPose()
        return pose

    def movel(self, pose, speed, acceleration):
        if self.rtde_c is None:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_c.moveL(pose, speed, acceleration)

    def close(self):
        if self.rtde_c is not None:
            self.rtde_c.disconnect()
