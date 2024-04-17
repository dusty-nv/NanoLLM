#!/usr/bin/env python3
# the terminal-based chat program was moved to chat/__main__.py
# forward deprecated 'python3 -m nano_llm' calls to nano_llm.chat
import runpy
#import logging

#logging.warning("'python3 -m nano_llm' is deprecated, please run nano_llm.chat or nano_llm.completion instead")

runpy.run_module('nano_llm.chat', run_name='__main__')
