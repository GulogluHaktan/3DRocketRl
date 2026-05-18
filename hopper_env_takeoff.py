from hopper_env import HopperRocketEnv


class HopperTakeoffEnv(HopperRocketEnv):
    def __init__(self, landing_start_z=10.0):
        super().__init__(task="takeoff", landing_start_z=landing_start_z)

