#!/usr/bin/env python3
from .audio_output import AudioOutputDevice, AudioOutputFile

from .auto_asr import AutoASR
from .auto_tts import AutoTTS

from .riva_asr import RivaASR
from .riva_tts import RivaTTS

from .fastpitch_tts import FastPitchTTS

try:
    from .xtts import XTTS
except ImportError as error:
    import logging
    logging.warning(f"failed to import XTTS plugin, disabling... ({error})")