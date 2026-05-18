from hopper_env import HopperRocketEnv


class HopperLandingEnv(HopperRocketEnv):
    def __init__(self, landing_start_z=10.0):
        super().__init__(task="landing", landing_start_z=landing_start_z)

