#!/usr/bin/env python3
import logging

from nano_llm import Plugin


class Mux(Plugin):
    """
    Multi-input, single-output multiplexer that selects one input and sends its output.
    """
    def __init__(self, **kwargs):
        """
        Multi-input, single-output multiplexer that selects one input and sends its output.
        """ 
        super().__init__(outputs="output", **kwargs)
        
        self.active_plugin = 'All'
        self.active_channel = 0

        self.add_parameter('active_input', type=str, default='All', options=['All'], controls=['dialog', 'node'], help="Select the input channel to output (others will be dropped unless 'All' is enabled)")

    @property
    def active_input(self):
        """
        Returns the name of the plugin that's the selected input.
        For cases where channel > 0, this will be like "Plugin.1"
        """
        return self.input_to_str(self.active_plugin, self.active_channel)
        
    @active_input.setter
    def active_input(self, value):
        """
        Select the active input to send. Should be the name of a plugin
        or a Plugin instance that is connected to this plugin as input.
        """
        if isinstance(value, Plugin):
            self.active_plugin, self.active_channel = value.name, 0
        elif isinstance(value, str):
            self.active_plugin, self.active_channel = self.input_from_str(value)
        else:
            raise TypeError(f"active_input is expected to be set to a string or Plugin instance (was {type(value)})")
    
    def process(self, input, sender=None, channel=0, **kwargs):
        """
        Drop messages that are not from an active input.
        """
        if not self.active_plugin or not sender:
            logging.warning(f"{self.name} | input not selected or sender was none (dropping message)")
            return
            
        if self.active_plugin.lower() == 'all':
            return input
            
        if self.active_plugin == sender.name and self.active_channel == channel:
            return input
                    
    def connect(self, plugin, channel=0, direction='send', **kwargs):
        print('MUX CONNECT', plugin.name, channel, direction)
        super().connect(plugin, direction=direction, **kwargs)
        if direction == 'send':
            return self
        options = self.parameters['active_input']['options']
        connection_name = self.input_to_str(plugin, channel)
        print('MUX CONNECTION_NAME', connection_name, options)
        if connection_name not in options:
            options.append(connection_name)
        self.send_inputs()
        return self
    
    def disconnect(self, plugin, channel=0, direction='send', **kwargs):
        super().disconnect(plugin, direction=direction, **kwargs)
        if direction == 'send':
            return self
        options = self.parameters['active_input']['options']
        connection_name = self.input_to_str(plugin, channel)
        if connection_name in options:
            options.remove(connection_name)
        self.send_inputs()
        return self
    
    def send_inputs(self):
        print('MUX SEND INPUTS', {
            'parameters': {'active_input': self.parameters['active_input']}
        })
        self.send_state(state_dict={
            'parameters': {'active_input': self.parameters['active_input']}
        })
    
    def input_to_str(self, plugin, channel):
        if isinstance(plugin, Plugin):
            plugin = plugin.name
        elif not isinstance(plugin, str):
            raise TypeError(f"expected a Plugin instance or string containing the plugin name (was {type(plugin)}")  
        if channel > 0:
            return f"{plugin}.{channel}"
        else:
            return plugin
            
    def input_from_str(self, value):
        value = value.strip().split('.')
        return value[0], int(value[1]) if len(value) > 1 else 0
