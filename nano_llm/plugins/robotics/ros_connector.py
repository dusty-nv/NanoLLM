#!/usr/bin/env python3
import logging
from functools import partial
from nano_llm import Plugin
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
import rclpy.logging
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.impl.rcutils_logger import RcutilsLogger
from rosidl_runtime_py import set_message, convert
import importlib
from queue import Queue
from enum import Enum
from typing import Optional, Annotated, Any, Union
from pydantic import BaseModel, Field, ValidationError
import threading
import time


class NodeType(str, Enum):
    """
    Enum for ROS2 object types
    """
    PUBLISHER = "publisher"
    SUBSCRIBER = "subscriber"
    SERVICE_CLIENT = "service_client"
    ACTION_CLIENT = "action_client"

class ROSMessage(BaseModel):
    """
    Pydantic schema for ROS2 topic, service client request, and action client goal messages
    """
    node_type: NodeType = Field(description = "the type of ROS2 node either 'publisher', 'subscriber', 'service_client', or 'action_client'")
    msg_type: str = Field(description = "the type of ROS2 message, service, or action, e.g. 'std_msgs/msg/String'")
    name: str = Field(description = "the name of the ROS2 topic, service, or client, e.g. 'chatter'")
    timer_period: Annotated[Optional[float], Field(description = "the period of the timer, ignored if type is not 'publisher'", default=0, ge=0.0)]
    timer_duration: Annotated[Optional[float], Field(description = "the duration of time that a message will be published, ignored if type is not 'publisher'", default=0, ge=0.0)]
    msg: Annotated[Optional[Union[dict, str]], Field(description = "the message payload for the topic, service request/response, or action goal/result", default=None)]


class ROS2Connector(Plugin, Node):
    """super()
    Plugin to take in well-formed JSON messages and convert them to ROS2 messages and vice versa.
    It dynamically creates ROS2 publishers/subscribers, service clients, and action clients based 
    on the JSON message.

    Inputs: (dict) -- JSON dictionary to be converted to ROS2 message
            NOTE: - JSON message must be well-formed and follow the ROSMessage schema.
                  - A publisher, subscriber, service client, or action client can be destroyed
                    by send the case-insensitive string 'DESTROY' in the 'msg' attribute of the input.
                  - An action goal can be canceled by sending the case-insensitive string 'CANCEL' in 
                    the 'msg' attribute of the input.

    Outputs: (str) -- JSON message converted from ROS2 message
    """

    def __init__(self, **kwargs):
        """
        Take in JSON message and parse to create ROS2 publishers, subscribers,
        service clients, and action clients. Publish on and subscribe to ROS2 topics
        and send service requests and action goals. Output JSON messages constructed
        from messages and responses.

        Args:
            init_args (list): list of arguments to pass to the ROS2 node

        """
        Plugin.__init__(self, inputs=['json_in'], outputs=['json_out'],**kwargs)

        # Initialize rclpy
        rclpy.init(args=kwargs.get('init_args', []))
        self.node = rclpy.create_node('ros2_connector')
        self.exec = MultiThreadedExecutor()
        self.exec.add_node(self.node)
        self._executor_thread = None

        self.node.get_logger().info("ROS2Connector plugin started")
        self.node.get_logger().info("ROS2 node created")

        # Initialize subnode dictionaries
        self.pubs = {}
        self.subs = {}
        self.service_clients = {}
        self.action_clients = {}

        # Initialize timer, callback group, and goal handle dictionaries
        self.timers_dict = {}
        self.callback_groups = {}

    def create_publisher(self, msg: ROSMessage, msg_class) -> bool:
        """
        Create a ROS2 publisher.
        """
        assert(msg.node_type == NodeType.PUBLISHER)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        _, msg_class, _ = self.get_ros_message_type(msg)
        pub_msg = self.json_to_ros_msg(msg, msg_class)
        try:
            self.pubs[msg.name] = self.node.create_publisher(msg_class, msg.name, 10)

            if msg.timer_duration != 0:
                self.timers_dict[msg.name] = {
                    "timer": self.node.create_timer(msg.timer_period, 
                                                    partial(self.timer_callback, ros_msg=pub_msg, msg=msg, start_time=time.time()),
                                                    callback_group=self.callback_groups[msg.name]),
                    "is_publishing": True
                }
            return True

        except Exception as e:
            self.node.get_logger().error(f"Failed to create publisher: {e}")
            return False

    def timer_callback(self, ros_msg: Any, msg: ROSMessage, start_time: float) -> None:
        """
        Timer callback function for ROS2 publishers.
        """
        assert(msg.node_type == NodeType.PUBLISHER)
        elapsed_time = time.time() - start_time
        if elapsed_time < msg.timer_duration:
            self.publish_ros_message(ros_msg, msg)
            self.node.get_logger().info(f"Published message to topic: {msg.name}")
        else:
            self.timers_dict[msg.name]["is_publishing"] = False

    def create_subscriber(self, msg: ROSMessage) -> bool:
        """
        Create a ROS2 subscriber.
        """
        assert(msg.node_type == NodeType.SUBSCRIBER)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        _, msg_class, _ = self.get_ros_message_type(msg)
        try:
            self.subs[msg.name] = self.node.create_subscription(msg_class, 
                                                                    msg.name, 
                                                                    partial(self.subscriber_callback, ros_msg=msg), 
                                                                    10)

            self.node.get_logger().info(f"Successfully created subscriber")
            return True
        except Exception as e:
            self.node.get_logger().info(f"Failed to create subscriber: {e}")
            return False

    def subscriber_callback(self, msg, ros_msg) -> None:
        """
        Callback function for ROS2 subscribers. Replace payload of original message with message
        received by subscriber and JSON dump to output.
        """
        json_msg = self.ros_msg_to_json(msg)
        # replace payload of original message with received message
        ros_msg.msg = json_msg
        # convert ROS2Message to JSON dict
        out_msg = ros_msg.model_dump()
        self.output(out_msg)

    def publish_ros_message(self, ros_msg: Any, msg: ROSMessage) -> bool:
        """
        Publish a ROS2 message to a topic.
        """
        assert(msg.node_type == NodeType.PUBLISHER)
        try:
            publisher = self.pubs.get(msg.name)
            publisher.publish(ros_msg)
            self.node.get_logger().info(f"Published message to topic: {msg.name}")
            return True
        except Exception as e:
            self.node.get_logger().error(f"Failed to publish message: {e}")
            return False

    def create_service_client(self, msg: ROSMessage, msg_class) -> bool:
        """
        Create a ROS2 service client.
        """
        assert(msg.node_type == NodeType.SERVICE_CLIENT)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        self.service_clients[msg.name] = {}
        self.service_clients[msg.name]['client'] = self.node.create_client(msg_class, msg.name)
        while not self.service_clients[msg.name]['client'].wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f"service {msg.name} not available, waiting again...")
        return True

    def send_service_request_async(self, msg: ROSMessage, msg_class) -> None:
        """
        Send a service request to a ROS2 service client, await response, and output response.
        """
        assert(msg.node_type == NodeType.SERVICE_CLIENT)
        request_msg = msg_class.Request()
        if msg.msg is not None:
            set_message.set_message_fields(request_msg, msg.msg)
        future = self.service_clients[msg.name]['client'].call_async(request_msg)
        self.service_clients[msg.name]['is_processing'] = True
        future.add_done_callback(lambda f: self.handle_service_response(f, msg))

    def handle_service_response(self, future, msg: ROSMessage) -> None:
        """
        Process the response of the service request.
        """
        try: 
            response = future.result()
            result_json = self.ros_msg_to_json(response)
            msg.msg = result_json
            out_msg = msg.model_dump()
            self.output(out_msg)

            self.node.get_logger().info(f"Service call succeeded; this is the result: {out_msg}")
            self.node.service_clients[msg.name]['is_processing'] = False
        except Exception as e:
            self.node.get_logger().error(f'Service call failed: {str(e)}')

    def create_action_client(self, msg: ROSMessage, msg_class) -> bool:
        """
        Create a ROS2 action client.
        """
        assert(msg.node_type == NodeType.ACTION_CLIENT)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        self.action_clients[msg.name] = {}
        self.action_clients[msg.name]['client'] = ActionClient(self.node, msg_class, msg.name)
        self.action_clients[msg.name]['is_processing'] = False
        return True

    def send_action_goal(self, msg: ROSMessage, msg_class) -> bool:
        """
        Wrapper for sending action goals and processing the results
        """
        assert(msg.node_type == NodeType.ACTION_CLIENT)
        goal_msg = msg_class.Goal()
        if msg.msg is not None:
            set_message.set_message_fields(goal_msg, msg.msg)
        action_client = self.action_clients[msg.name]['client']

        def goal_response_callback(future):
            """
            Process the response of the goal request.
            """
            goal_handle = future.result()
            self.action_clients[msg.name]['goal_handle'] = goal_handle
            if not goal_handle.accepted:
                self.node.get_logger().error(f"Goal rejected by action server: {msg.name}")
                return
            self.node.get_logger().info(f"Goal accepted by action server: {msg.name}")
            self.action_clients[msg.name]['is_processing'] = True
            _get_result_future = goal_handle.get_result_async()
            _get_result_future.add_done_callback(get_result_callback)

        # feedback callback for action client - varies depending on action type
        # process for handling this is still under consideration
        def feedback_callback(feedback):
            """
            TODO: (Might vary across Plugin needs)
            """
            pass

        def get_result_callback(future):
            """
            Process the result of the action goal.
            """
            results = future.result()
            if results.status == GoalStatus.STATUS_SUCCEEDED:
                self.node.get_logger().info("Goal succeeded!")
            elif results.status == GoalStatus.STATUS_ABORTED:
                self.node.get_logger().error("Goal aborted.")
            elif results.status == GoalStatus.STATUS_CANCELED:
                self.node.get_logger().info("Goal cancelled.")
            else:
                self.node.get_logger().error("Unknown goal status.")

            results_json = {
                'status': results.status,
                'result': self.ros_msg_to_json(results.result)
            }

            self.node.get_logger().info("Outputting results!")
            self.action_clients[msg.name]['is_processing'] = False
            self.output(results_json)

        def send_goal():
            """
            Wait for appropriate action server to become available and send goal request.
            """
            self.node.get_logger().info(f"Sending goal request to action server: {msg.name}")
            action_client.wait_for_server()
            self.node.get_logger().info(f"Sending goal request...")
            _send_goal_future = action_client.send_goal_async(goal_msg,
                                                              feedback_callback=feedback_callback)
            _send_goal_future.add_done_callback(goal_response_callback)

        # This will return a goal handle that can be used to cancel the goal
        action_goal_handle = send_goal()
        try:
            pass
        except Exception as e:
            self.node.get_logger().error(f"Failed to send action goal: {e}")
            return
        return action_goal_handle

    def cancel_done_callback(self, msg, future):
        """
        Process the result of the goal cancellation.
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.node.get_logger().info("Goal cancellation successful.")
            self.action_clients[msg.name]['is_processing'] = False
        else:
            self.node.get_logger().error("Goal cancellation failed.")

    def process(self, input: dict, **kwargs): 
        """
        Create a ROS2 publisher, subscriber, service client, or action client.
        Publish messages to topics, subscribe to topics, send service requests, and send action goals.
        """
        json_msg = input
        msg = self.get_ros_msg_from_json(json_msg)
        if len(msg.msg_type) > 0:
            msg_type, msg_class, _ = self.get_ros_message_type(msg)
            if msg.node_type not in (NodeType.ACTION_CLIENT, NodeType.SERVICE_CLIENT):
                ros_msg = self.json_to_ros_msg(msg, msg_class)
            # Convert JSON message payload to ROS2 message for any publishing

        node_type = msg.node_type

        match node_type:

            case NodeType.PUBLISHER:
                if not self.pubs.get(msg.name) and not isinstance(msg.msg, str):
                    self.create_publisher(msg, msg_class)
                    # If the timer period is 0, publish the message immediately and only once
                    if msg.timer_period == 0:
                        self.publish_ros_message(msg)
                        self.node.get_logger().info(f"Published message to topic: {msg.name}")
                # if the publisher already exists and the message is not a string, we will publish without creating new publisher
                elif self.pubs.get(msg.name) and not isinstance(msg.msg, str):
                    # publish immediately and only once if timer_period is 0
                    if msg.timer_period == 0:
                        self.publish_ros_message(msg)
                        self.node.get_logger().info(f"Published message to topic: {msg.name}")
                    else:
                        if self.timers_dict.get(msg.name):
                            if self.timers_dict[msg.name]["is_publishing"]:
                                self.node.get_logger().warn(f"Publisher is already publishing to topic: {msg.name}. Overwriting current publisher.")
                            self.timers_dict[msg.name]["timer"].destroy()
                        self.timers_dict[msg.name] = {
                            "timer": self.node.create_timer(msg.timer_period, 
                                                            partial(self.timer_callback, ros_msg=ros_msg, 
                                                            msg=msg, start_time=time.time()),
                                                            callback_group=self.callback_groups[msg.name]),
                            "is_publishing": True
                        }

                # If the message is a string and message == 'destroy', we destroy the publisher
                elif isinstance(msg.msg, str) and msg.msg.lower() == 'destroy':
                    if self.timers_dict.get(msg.name):
                        if self.timers_dict[msg.name]["is_publishing"]:
                            self.node.get_logger().warn(f"Publisher is publishing to topic: {msg.name}. Stopping publishing and destroying publisher.")

                        # destroy the timer
                        self.timers_dict[msg.name]["timer"].destroy()
                        del self.timers_dict[msg.name]

                    # destroy the publisher
                    self.node.destroy_publisher(self.pubs[msg.name])
                    del self.pubs[msg.name]
                    del self.callback_groups[msg.name]
                    self.node.get_logger().info(f"Publisher destroyed: {msg.name}")
                    while not rclpy.ok():
                        pass

                else:
                    self.node.get_logger().error(f"Received unknown msg for publisher: {msg.msg}")

            # Multiple subscribers to the same topic will be prohibited for now
            case NodeType.SUBSCRIBER:
                if msg.msg is None:
                    if not self.subs.get(msg.name):
                        self.create_subscriber(msg)
                    else:
                        self.node.get_logger().warn(f"Subscriber to {msg.name} already exists.")

                # if msg.msg == "destroy" then we simply destroy the subscriber
                elif isinstance(msg.msg, str) and msg.msg.lower() == 'destroy':
                    self.node.destroy_subscription(self.subs[msg.name])
                    del self.subs[msg.name]
                    self.node.get_logger().info(f"Subscriber destroyed: {msg.name}")

                # msg.msg is populated with something other than "destroy"
                else:
                    self.node.get_logger().error(f"Received unknown msg for subscriber: {msg.msg}")

            case NodeType.SERVICE_CLIENT:
                # if the service client does not exist, we create it and send the service request
                if not self.service_clients.get(msg.name):
                    self.create_service_client(msg, msg_class)
                    self.send_service_request_async(msg, msg_class)

                # if the service client does exist, we send the service request without redundantly creating/destroying it
                elif self.service_clients.get(msg.name) and not isinstance(msg.msg, str):
                    if self.service_clients[msg.name]['is_processing']:
                        self.node.get_logger().warn(f"Service client {msg.name} is already processing a request. Please wait until the current process completes.")
                    else:
                        self.send_service_request_async(msg, msg_class)

                # if the message is a string and message == 'destroy', we destroy the service client and any associated overhead
                elif isinstance(msg.msg, str) and msg.msg.lower() == 'destroy':

                    if self.service_clients[msg.name]['is_processing']:
                        self.node.get_logger().warn(f"Service client {msg.name} is currently processing a request. Destroying client anyways and aborting process.")

                    self.node.destroy_client(self.service_clients[msg.name]['client'])
                    del self.service_clients[msg.name]
                    del self.callback_groups[msg.name]
                    self.node.get_logger().info(f"Service client destroyed: {msg.name}")

                else:
                    self.node.get_logger().error(f"Received unknown msg for service client: {msg.msg}")

            # If we are dealing with an action client
            case NodeType.ACTION_CLIENT:
                # we create an action client if one does not exist for specified action and send goal
                if not self.action_clients.get(msg.name) and not isinstance(msg.msg, str):
                    self.create_action_client(msg, msg_class)
                    self.action_clients[msg.name]['goal_handle'] = self.send_action_goal(msg, msg_class)

                # if client already exists for action type, don't bother creating another -- just send goal
                elif self.action_clients.get(msg.name) and not isinstance(msg.msg, str):
                    if self.action_clients[msg.name]['is_processing']:
                        self.node.get_logger().warn(f"Action client {msg.name} is already processing a goal. Please wait until the current process completes.")
                    else:
                        self.action_clients[msg.name]['goal_handle'] = self.send_action_goal(msg, msg_class)

                # cancel the current goal if there is one. If not, do nothing and communicate to the user that there is not a goal to cancel
                elif isinstance(msg.msg, str) and msg.msg.lower() == 'cancel':
                    if self.action_clients[msg.name]['is_processing']:
                        cancel_future = self.action_clients[msg.name]['goal_handle'].cancel_goal_async()
                        cancel_future.add_done_callback(partial(self.cancel_done_callback, msg))
                    else:
                        self.node.get_logger().warn(f"Action client {msg.name} is not currently processing a goal. Cannot cancel.")

                # we will destroy the action client so long as it is not processing a goal. If goal is being processed, prompt user to cancel and do nothing.
                elif isinstance(msg.msg, str) and msg.msg.lower() == 'destroy':
                    if self.action_clients[msg.name]['is_processing']:
                        self.node.get_logger().warn(f"Action client {msg.name} is currently processing a goal. You must cancel the goal first or wait for processing to complete.")
                    else:
                        self.node.destroy_client(self.action_clients[msg.name]['client'])
                        del self.action_clients[msg.name]
                        del self.callback_groups[msg.name]
                        self.node.get_logger().info(f"Action client destroyed: {msg.name}")

                else:
                    self.node.get_logger().error(f"Received unknown msg for action client: {msg.msg}")

            case _:
                self.node.get_logger().error(f"Invalid ROS2 node type: {msg.node_type}")

    # Override the Plugin.run method to spin the ROS2 node
    def run(self):
        """
        Processes the queue forever and automatically run when created with ``threaded=True``
        """
        self._executor_thread = threading.Thread(target=self.exec.spin)
        self._executor_thread.start()

        while not self.stop_flag:
            try:
                self.process_inputs(timeout=0.25)
            except Exception as error:
                logging.error(f"Exception occurred during processing of {self.name}\n\n{traceback.format_exc()}")

        logging.debug(f"{self.name} plugin stopped (thread {self.native_id})")

    # Override the Plugin.remove_plugin method to shutdown and remove the ROS2 node
    def destroy(self):
        """
        Stop a plugin thread's running, and unregister it from the global instances.
        """
        self.exec.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()

        self.stop()

        try:
            Plugin.Instances.remove(self)
        except ValueError:
            logging.warning(f"Plugin {getattr(self, 'name', '')} wasn't in global instances list when being deleted")

        plugin.destroy()
        del plugin

    def json_to_ros_msg(self, msg: ROSMessage, msg_class):
        """
        Convert JSON message to ROS2 message.
        """
        try:
            if hasattr(msg_class, 'Goal'):
                ros_msg = msg_class.Goal()
            elif hasattr(msg_class, 'Request'):
                ros_msg = msg_class.Request()
            else:
                ros_msg = msg_class()
        except Exception as e:
            self.node.get_logger().error(f"Failed to create ROS2 message: {e}")
            return False

        if msg.msg is not None:
            msg_dict = msg.msg
            set_message.set_message_fields(ros_msg, msg_dict)

        return ros_msg

    def ros_msg_to_json(self, ros_msg) -> dict:
        """
        Convert ROS2 message to JSON message.
        """
        return convert.message_to_ordereddict(ros_msg)

    def get_ros_msg_from_json(self, json_msg: dict) -> ROSMessage:
        """
        Validate JSON input and cast to ROSMessage.
        """
        try:
            ros_msg = ROSMessage.parse_obj(json_msg)
        except ValidationError as e:
            print(f"Invalid ROS2 message: {e}")
            self.node.get_logger().error(f"Invalid ROS2 message sent by agent: {e}")
            return False

        return ros_msg

    def get_ros_message_type(self, ros_msg: ROSMessage) -> tuple:
        """
        Get the ROS2 message type from the JSON message and import the class.
        """
        msg_type_str = ros_msg.msg_type
        package, msg_dir, msg_type = msg_type_str.split('/')

        # get the message class and import module and class
        msg_module = importlib.import_module(f'{package}.{msg_dir}')
        msg_class = getattr(msg_module, msg_type)
        globals()[msg_type] = msg_class
        node_type = ros_msg.node_type
        return msg_type, msg_class, node_type

    def state_dict(self, **kwargs):
        return {**super().state_dict(**kwargs)}
