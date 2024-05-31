#!/usr/bin/env python3
import sys
import termcolor

from nano_llm.utils import ArgParser
from nano_llm.plugins import AudioInputDevice, AudioOutputDevice, AudioOutputFile

args = ArgParser(extras=['audio_input', 'audio_output', 'log']).parse_args()

audio_input = AudioInputDevice(**vars(args))

if args.audio_output_device is not None:
    audio_input.add(AudioOutputDevice(**vars(args)))

if args.audio_output_file is not None:
    audio_input.add(AudioOutputFile(**vars(args)))

audio_input.start().join()
