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
from nano_llm.utils import filter_keys, convert_tensor


class MimicGen(Plugin):
    """
    Robot simulator and image generator using mimicgen and robosuite:
    
      https://github.com/NVlabs/mimicgen
      https://github.com/ARISE-Initiative/robosuite
    """
    def __init__(self, environment: str="Stack_D0", robot: str="Panda", gripper: str="PandaGripper",
                       camera: str="frontview", camera_width: int=512, camera_height: int=512,
                       framerate: int=10, genlock: bool=False, domain_randomization: str=None,
                       repeat_actions: bool=False, partial_actions: bool=False, action_scale: float=1.0, **kwargs):
        """
        Robot simulator and image generator using robosuite from robosuite.ai
        """
        super().__init__(outputs=['image', 'instruct'], **kwargs)
 
        self.sim = None
        self.sim_config = {}
        
        self.reset = False
        self.pause = False

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
                            camera_width=camera_width, camera_height=camera_height, framerate=framerate, 
                            genlock=genlock, domain_randomization=domain_randomization, 
                            repeat_actions=False, partial_actions=partial_actions, action_scale=action_scale)          
                           
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
            camera = dict(options=['agentview', 'frontview', 'sideview', 'birdview', 'robot0_robotview', 'robot0_eye_in_hand']),
            genlock = dict(display_name='GenLock')
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
            control_freq=20, #self.framerate,
            controller_configs=controller,
        )
        
        self.sim.reset()
        self.reset = False
        
        self.next_action = None
        self.last_action = None

        robot = self.sim.robots[0]
        logging.info(f"{self.name} setup sim environment with configuration:\n\n{pprint.pformat(config, indent=2)}\n\nrobot_dof={robot.dof}\naction_dim={robot.action_dim}\naction_limits={pprint.pformat(robot.action_limits)}\n")
        
    def render(self, action=None):
        """
        Render a frame from the simulator.
        """
        self.config_sim()
        
        if action is None:  # select the next action to use from different sources
            action = self.get_action() 
        
        if action is None:
            self.reset = True
            return  # either there was no action, or action was now from prior episode

        gripper = (action[-1] * 2.0 - 1.0) * -1.0   # remap from [closed=0,open+1] to [open=-1, closed=1]
        action_scaled = action * self.action_scale  # apply scaling factors, except for the gripper
        action_scaled[-1] = gripper

        obs, reward, done, info = self.sim.step(action_scaled)
  
        if reward or done:
            logging.debug(f"{self.name} reward={reward}  done={done}\n")  #   obs={list(obs.keys())}
        
        self.last_action = action
        self.next_action = None
        
        image = obs.get(f'{self.camera}_image')

        if image is None:
            return
            
        if any([x in self.camera for x in ['agentview', 'frontview', 'sideview']]):
            image = np.flip(image, axis=0).copy() # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2

        instruct = self.get_instruction()
        
        if instruct:
            self.output(instruct, channel='instruct')
            
        self.output(image)

    def get_action(self):
        """
        Selects the next action to use, either from an agent, user input, random patterns.
        These can be assembled from previous frames if ``repeat_actions=True``
        """
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
            if self.next_action is not None:    # there is a new, complete action to use
                action = self.next_action
            elif self.last_action is not None:
                if self.repeat_actions:         # reuse the last action
                    action = self.last_action
                else:                           # only reuse the gripper
                    action = np.concatenate((np.zeros(dof-1), self.last_action[-1:]))
            else:                               # stopped state (no motion)
                action = np.concatenate((np.zeros(dof-1), [1.0]))
        else:
            raise ValueError(f"{self.name}.motion_select had invalid value '{self.motion_select}'  (options: {self.parameters['motion_select']['options']})")
        
        return action

    def get_instruction(self):
        """
        Get the natural language instruction or command for the robot to follow.
        """
        env = self.environment.lower()
        
        if 'stack_three' in env:
            return "stack the red block on top of the green block, and then the blue block on top of the red block."
        elif 'stack' in env:
            return "stack the red block on top of the green block"
            
    def update(self):
        """
        Run one tick of the rendering loop
        """
        if self.reset:
            logging.info(f"{self.name} | resetting sim environment ({self.environment})\n")
            
            self.sim.reset()
            self.clear_inputs()
            self.reset = False 
            
            self.last_action = None
            self.next_action = None
    
            if self.keyboard:
                self.keyboard.start_control()
                
            if self.spacenav:
                self.spacenav.start_control()

        if self.pause:
            time.sleep(0.25)
            self.clear_inputs()
            self.last_action = None
            self.next_action = None
            return
            
        self.process_inputs()
        time_begin = time.perf_counter()
        
        if self.genlock:
            if self.next_action is not None or self.last_action is None:
                self.render() # only render on new input (or on empty pipeline)
        else:
            self.render()
            
        render_time = time.perf_counter() - time_begin
        sleep_time = (1.0 / self.framerate) - render_time
        
        self.send_stats(summary=[f"{int(render_time * 1000)} ms"])
        
        if sleep_time > 0:
            time.sleep(sleep_time)
                    
    def run(self):
        """
        Simulator rendering loop that runs forever
        """
        while not self.stop_flag:
            try:
                self.update()
            except Exception as error:
                logging.error(f"Exception occurred in {self.name}\n\n{traceback.format_exc()}")
                self.sim = None
        
    def process(self, action, partial=False, **kwargs):
        """
        Recieve action inputs and apply them to the simulation.
        """
        dof = self.sim.robots[0].action_dim
        action = convert_tensor(action, return_tensors='np')
        
        if not partial:
            self.next_action = action
        elif self.partial_actions and len(action) < dof:
            if self.repeat_actions and self.last_action is not None:
                self.next_action = np.concatenate((action, self.last_action[len(action):]))
            else:
                self.next_action = np.concatenate((action, [0] * (dof-len(action)-1), [1]))       

