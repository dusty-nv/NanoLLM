#!/usr/bin/env python3
from nano_llm.agents import DynamicAgent
from nano_llm.utils import ArgParser


if __name__ == "__main__":
    parser = ArgParser(extras=['web', 'log'])
    
    parser.add_argument("--load", type=str, default=None, help="load an agent from .json or .yaml")
    parser.add_argument("--agent-dir", type=str, default="/data/nano_llm/agents", help="change the agent load/save directory")
    
    parser.add_argument("--index", "--page", type=str, default="studio.html", help="the filename of the site's index html page (should be under web/templates)") 
    parser.add_argument("--root", type=str, default=None, help="the root directory for serving site files (should have static/ and template/")
    
    args = parser.parse_args()

    agent = DynamicAgent(**vars(args)).run()

    
