#!/usr/bin/env python3
import sys
import time
import signal
import logging
import termcolor

from nano_llm import Agent, Pipeline

from nano_llm.plugins import UserPrompt, ChatQuery, PrintStream, Callback
from nano_llm.utils import ArgParser, print_table


class ChatAgent(Agent):
    """
    Agent for two-turn multimodal chat.
    """
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", interactive=True, **kwargs):
        """
        Args:
          model (NanoLLM|str): either the loaded model instance, or model name/path to load.
          interactive (bool): should the agent get user input from the terminal or not (default True)
        """
        super().__init__()
        
        """
        # Equivalent to:
        self.pipeline = UserPrompt(interactive=interactive, **kwargs).add(
            ChatQuery(model, **kwargs).add(
            PrintStream(relay=True).add(self.on_eos)     
        ))
        """
        
        #: input() → LLM → print() pipeline.
        self.pipeline = Pipeline([
            UserPrompt(interactive=interactive, **kwargs),
            ChatQuery(model, **kwargs),
            PrintStream(relay=True),
            self.on_eos    
        ])
        
        #: The ``ChatQuery`` session manager
        self.chat = self.pipeline[0].find(ChatQuery)
        
        #: The loaded NanoLLM model instance
        self.model = self.chat.model
        
        self.interactive = interactive
        self.last_interrupt = 0
        
        signal.signal(signal.SIGINT, self.on_interrupt)
        self.print_input_prompt()

    def on_interrupt(self, signum, frame):
        """
        Interrupts the bot output when the user presses ``Ctrl+C``.
        """
        curr_time = time.perf_counter()
        time_diff = curr_time - self.last_interrupt
        self.last_interrupt = curr_time
        
        if time_diff > 2.5:
            logging.warning("Ctrl+C:  interrupting chatbot (press again to exit)")
            self.chat.interrupt()
        else:
            while True:
                logging.warning("Ctrl+C:  exiting...")
                sys.exit(0)
                time.sleep(0.5)
            
    def on_eos(self, input):
        if input.endswith('</s>'):
            print_table(self.model.stats)
            self.print_input_prompt()

    def print_input_prompt(self):
        if self.interactive:
            termcolor.cprint('>> PROMPT: ', 'blue', end='', flush=True)
        
        
if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument("-it", "--interactive", action="store_true", help="enable interactive user input from the terminal")
    args = parser.parse_args()
    
    agent = ChatAgent(**vars(args)).run() 