import asyncio
import os
import sys
from loguru import logger

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.processors.logger import FrameLogger

from app.services.pipecat_wrapper import IndicWSSTTService

import argparse

async def main():
    parser = argparse.ArgumentParser(description="Pipecat Indic ASR Client")
    parser.add_argument("--language", "-l", type=str, default="hi", choices=["hi", "ne", "mai"], help="Language code (hi, ne, mai)")
    args = parser.parse_args()

    logger.info(f"Starting Pipecat Orchestration for language: {args.language}")

    # 1. Transport (Microphone) using simple VAD
    # Note: We rely on Silero VAD (using onnxruntime-gpu we have)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True, # Loopback? Or just to keep it running
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.0)), 
        )
    )

    # 2. Our Custom STT Service
    stt = IndicWSSTTService(
        uri="ws://localhost:8000/transcribe/ws",
        language=args.language 
    )
    
    # 3. Simple Logger to print transcripts
    logger_sink = FrameLogger("TRANSCRIPT") 

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            logger_sink,
            # We don't have an output transport configured to really play back, 
            # but we need to sink the frames somewhere or just end.
            # transport.output() would echo audio back.
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)
    
    print(f">>> Speak now ({args.language})... Press Ctrl+C to stop.")
    
    await runner.run(task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
