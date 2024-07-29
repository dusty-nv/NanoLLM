#!/usr/bin/env python3
import time
import pprint
import logging
import traceback

import numpy as np
import robosuite as rs
import mimicgen

from robosuite.devices import Keyboard, SpaceMouse
from robosuite.utils.input_utils import input2action

from nano_llm import Plugin
from nano_llm.utils import filter_keys


class MimicGen(Plugin):
    """
    Robot simulator and image generator using mimicgen and robosuite:
    
      https://github.com/NVlabs/mimicgen
      https://github.com/ARISE-Initiative/robosuite
    """
    def __init__(self, environment: str="Stack_D0", robot: str="Panda", gripper: str="PandaGripper",
                       camera: str="frontview", camera_width: int=512, camera_height: int=512,
                       framerate: int=10, **kwargs):
        """
        Robot simulator and image generator using robosuite from robosuite.ai
        """
        super().__init__(outputs='image', **kwargs)
 
        self.sim = None
        self.sim_config = {}
        
        self.keyboard = None
        self.spacenav = None

        try:
            self.keyboard = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
            self.keyboard.start_control()
        except Exception as error:
            logging.warning(f"{self.name} failed to open keyboard device:\n\n{traceback.format_exc()}")
          
        try:
            self.spacenav = SpaceMouse(pos_sensitivity=1.0, rot_sensitivity=1.0)
            self.spacenav.start_control()
        except Exception as error:
            logging.warning(f"{self.name} failed to open spacenav device:\n\n{traceback.format_exc()}")

        input_options = ['disabled', 'random', 'agent']
        
        if self.keyboard:
            input_options.append('keyboard')
            
        if self.spacenav:
            input_options.append('spacenav')
      
        self.add_parameters(environment=environment, robot=robot, gripper=gripper, camera=camera, 
                            camera_width=camera_width, camera_height=camera_height, framerate=framerate)          
                           
        self.add_parameter('motion_select', type=str, default='disabled', options=input_options)
        
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
            #motion_select = dict(options=['disabled', 'random', 'agent', 'keyboard']),
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
        
        self.next_action = None
        self.last_action = None
        
        logging.info(f"{self.name} setup sim environment with configuration:\n\n{pprint.pformat(config, indent=2)}\n\nrobot_dof={robot.dof}\naction_dim={robot.action_dim}\naction_limits=\n{pprint.pformat(robot.action_limits)}\n")
        
    def render(self):
        """
        Render a frame from the simulator.
        """
        self.config_sim()
        
        dof = self.sim.robots[0].action_dim #.dof
        
        if self.motion_select == 'disabled':
            action = np.zeros(dof)
        elif self.motion_select == 'random':
            action = np.random.randn(dof)
        elif self.motion_select == 'keyboard':
            action, gripper = input2action(device=self.keyboard, robot=self.sim.robots[0])
            logging.debug(f"{self.name} keyboard actions {action}")
        elif self.motion_select == 'spacenav':
            action, gripper = input2action(device=self.spacenav, robot=self.sim.robots[0])
        elif self.motion_select == 'agent':
            if self.next_action is not None:
                action = self.next_action
            else:
                if self.last_action is not None:
                    action = np.concatenate([np.zeros(dof-1), self.last_action[-1:]], axis=0)
                else:
                    action = np.zeros(dof)
        else:
            raise ValueError(f"{self.name}.motion_select had invalid value '{self.motion_select}'  (options: {self.parameters['motion_select']['options']})")
         
        if action is None:
            logging.debug(f"{self.name} input triggered sim reset\n")
            self.sim.reset()
            
            if self.keyboard:
                self.keyboard.start_control()
                
            if self.spacenav:
                self.spacenav.start_control()
                
            return
         
        logging.debug(f"{self.name} {self.motion_select} actions:  {action}")
              
        obs, reward, done, info = self.sim.step(action)
  
        image = obs.get(f'{self.camera}_image')
        
        if 'sideview' in self.camera or 'frontview' in self.camera:
            image = np.flip(image, axis=0).copy() # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2

        self.last_action = action
        self.next_action = None

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
        
    def process(self, action, partial=False, **kwargs):
        """
        Recieve action inputs and apply them to the simulation.
        """
        if not partial:
            action[-1] = (action[-1] * 2.0 - 1.0) * -1.0
            self.next_action = action
                

