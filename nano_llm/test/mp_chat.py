#!/usr/bin/env python3
from nano_llm import Agent, Pipeline

from nano_llm.plugins import UserPrompt, ChatQuery, PrintStream, ProcessProxy
from nano_llm.utils import ArgParser


class MultiprocessChat(Agent):
    """
    Test of running a LLM and chat session in a subprocess.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.pipeline = Pipeline([
            UserPrompt(interactive=False, **kwargs),  # interactive=False if kwargs.get('prompt') else True
            ProcessProxy((lambda **kwargs: ChatQuery(**kwargs)), **kwargs),
            PrintStream(color='green')  
        ])

         
if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    
    agent = MultiprocessChat(**vars(args)).run()