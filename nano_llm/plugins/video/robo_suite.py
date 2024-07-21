#!/usr/bin/env python3
import time
import pprint
import logging
import traceback

import numpy as np
import robosuite as rs

from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

from nano_llm import Plugin
from nano_llm.utils import filter_keys


class RoboSuite(Plugin):
    """
    Robot simulator and image generator using robosuite:
      https://github.com/ARISE-Initiative/robosuite
    """
    def __init__(self, environment: str="Lift", robot: str="UR5e", gripper: str="RethinkGripper",
                       camera: str="frontview", camera_width: int=640, camera_height: int=480,
                       framerate: int=10, motion_select: str='random', **kwargs):
        """
        Robot simulator and image generator using robosuite from robosuite.ai
        """
        super().__init__(outputs='image', **kwargs)
        
        self.sim = None
        self.sim_config = {}
        self.keyboard = None
        
        self.add_parameters(environment=environment, robot=robot, gripper=gripper, camera=camera, 
                            camera_width=camera_width, camera_height=camera_height, framerate=framerate,
                            motion_select=motion_select)

    @classmethod
    def type_hints(cls):
        """
        Return static metadata about the plugin settings.
        """
        return dict(
            environment = dict(options=list(rs.ALL_ENVIRONMENTS)),
            robot = dict(options=list(rs.ALL_ROBOTS)),
            gripper = dict(options=list(rs.ALL_GRIPPERS)),
            camera = dict(options=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand']),
            motion_select = dict(options=['stop', 'random', 'agent', 'keyboard']),
       )
    
    def config_sim(self):
        """
        Configure the simulator with the desired settings.
        """
        config = dict(env_name=self.environment, robots=self.robot, 
                      gripper_types=self.gripper, camera_names=self.camera, 
                      camera_widths=self.camera_width, camera_heights=self.camera_height,
                      framerate=self.framerate, controller='OSC_POSE')
        
        if self.sim is not None and self.sim_config == config:
            return self.sim
            
        self.sim_config = config
        
        controller = rs.load_controller_config(default_controller='OSC_POSE')
        
        self.sim = rs.make(
            **filter_keys(config.copy(), remove=['framerate', 'controller']),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            ignore_done=True,
            control_freq=self.framerate,
            controller_configs=controller,
        )
        
        self.sim.reset()
        robot = self.sim.robots[0]
        
        if self.keyboard is None:
            self.keyboard = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
            self.keyboard.start_control()
        
        logging.info(f"{self.name} setup sim environment with configuration:\n\n{pprint.pformat(config, indent=2)}\n\nrobot_dof={robot.dof}\naction_dim={robot.action_dim}\naction_limits=\n{pprint.pformat(robot.action_limits)}\n")
        
    def render(self):
        """
        Render a frame from the simulator.
        """
        self.config_sim()
        
        dof = self.sim.robots[0].dof
        
        if self.motion_select == 'stop':
            action = np.zeros(dof)
        elif self.motion_select == 'random':
            action = np.random.randn(dof)
        elif self.motion_select == 'keyboard':
            action, gripper = input2action(device=self.keyboard, robot=self.sim.robots[0])
            logging.debug(f"{self.name} keyboard actions {action}")
        else:
            raise ValueError(f"{self.name}.motion_select had invalid value '{self.motion_select}'  (options: {self.type_hints()['motion_select']['options']})")
            
        obs, reward, done, info = self.sim.step(action)
  
        image = obs.get(f'{self.camera}_image')
        
        if 'sideview' in self.camera or 'frontview' in self.camera:
            image = np.flip(image, axis=0)

        self.output(image)

    def run(self):
        """
        Run capture continuously and attempt to handle disconnections
        """
        while not self.stop_flag:
            try:
                time_begin = time.perf_counter()
                self.process_inputs()
                self.render()
                render_time = time.perf_counter() - time_begin
                sleep_time = (1.0 / self.framerate) - render_time
                self.send_stats(summary=[f"{int(render_time * 1000)} ms"])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as error:
                logging.error(f"Exception occurred in {self.name}\n\n{traceback.format_exc()}")
                self.sim = None
                

