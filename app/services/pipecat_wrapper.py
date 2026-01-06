
import asyncio
import json
import websockets
from loguru import logger
from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from app.constants import log_msg

class IndicWSSTTService(STTService):
    """
    Custom Pipecat STT Service that integrates with the Indic ASR WebSocket.

    Attributes:
        _uri (str): WebSocket URI for the ASR service.
        _language (str): Target language code.
        _websocket (websockets.WebSocketClientProtocol): Active WS connection.
        _receive_task (asyncio.Task): Background task receiving transcripts.
        _transcript_queue (asyncio.Queue): Queue buffering received transcripts.
    """
    def __init__(self, uri: str, language: str = "hi", **kwargs) -> None:
        """
        Initializes the STT service.

        Args:
            uri (str): WebSocket endpoint URI.
            language (str): Language code. Defaults to "hi".
            **kwargs: Arguments passed to the parent STTService.
        """
        super().__init__(**kwargs)
        self._uri: str = uri
        self._language: str = language
        self._websocket = None
        self._receive_task = None
        self._transcript_queue: asyncio.Queue = asyncio.Queue()
        logger.info(log_msg.LOG_ENTRY.format("IndicWSSTTService.__init__", f"URI={uri}, Lang={language}"))
        
    async def start(self, frame: StartFrame) -> None:
        """
        Starts the service and initiates the WebSocket connection.
        """
        logger.info(log_msg.LOG_ENTRY.format("IndicWSSTTService.start", ""))
        try:
            await super().start(frame)
            await self._connect()
            logger.info(log_msg.LOG_EXIT.format("IndicWSSTTService.start"))
        except Exception as e:
            logger.error(log_msg.LOG_ERROR.format("IndicWSSTTService.start", str(e)))
            raise e
        return None

    async def stop(self, frame: EndFrame) -> None:
        """
        Stops the service and closes the connection.
        """
        logger.info(log_msg.LOG_ENTRY.format("IndicWSSTTService.stop", ""))
        try:
            await super().stop(frame)
            await self._disconnect()
            logger.info(log_msg.LOG_EXIT.format("IndicWSSTTService.stop"))
        except Exception as e:
            logger.error(log_msg.LOG_ERROR.format("IndicWSSTTService.stop", str(e)))
            raise e
        return None

    async def cancel(self, frame: Frame) -> None:
        """
        Cancels the service operations.
        """
        logger.info(log_msg.LOG_ENTRY.format("IndicWSSTTService.cancel", ""))
        try:
            await super().cancel(frame)
            await self._disconnect()
            logger.info(log_msg.LOG_EXIT.format("IndicWSSTTService.cancel"))
        except Exception as e:
            logger.error(log_msg.LOG_ERROR.format("IndicWSSTTService.cancel", str(e)))
            raise e
        return None
        
    async def _connect(self) -> None:
        """
        Establish WebSocket connection and start the receive loop.
        """
        try:
            logger.info(log_msg.WS_CONNECTING.format(self._uri))
            self._websocket = await websockets.connect(f"{self._uri}?language={self._language}")
            self._receive_task = asyncio.create_task(self._receive_messages())
            logger.info(log_msg.WS_CONNECTED)
        except Exception as e:
            logger.error(log_msg.LOG_ERROR.format("_connect", str(e)))
            raise e
        return None

    async def _disconnect(self) -> None:
        """
        Cleanly close WebSocket and cancel background tasks.
        """
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                logger.info(log_msg.TASK_CANCELLED.format("receive_task"))
            except Exception as e:
                logger.error(log_msg.TASK_CANCEL_ERROR.format(e))
                raise e
            self._receive_task = None
            
        if self._websocket:
            try:
                await self._websocket.close()
                logger.info(log_msg.WS_CLOSED_CLEANLY)
            except Exception as e:
                logger.error(log_msg.LOG_ERROR.format("_disconnect/close", str(e)))
                raise e
            self._websocket = None
        return None

    async def _receive_messages(self) -> None:
        """
        Background task to receive messages from the WebSocket.
        """
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "transcription":
                        text = data.get("text")
                        if text:
                            tf = TranscriptionFrame(text=text, user_id="user", timestamp=None)
                            await self._transcript_queue.put(tf)
                except Exception as e:
                    logger.error(log_msg.WS_MSG_PARSE_ERROR.format(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(log_msg.WS_CONNECTION_CLOSED)
        except Exception as e:
            logger.error(log_msg.WS_RECEIVE_ERROR.format(e))
            raise e
        return None

    async def process_frame(self, frame: Frame, direction) -> None:
        """
        Processes incoming frames from the pipeline.
        
        Args:
            frame (Frame): Input frame.
            direction: Data flow direction.
        """
        # 1. Push any pending transcripts from queue
        while not self._transcript_queue.empty():
            try:
                transcript_frame = self._transcript_queue.get_nowait()
                await self.push_frame(transcript_frame, direction)
            except Exception as e:
                logger.error(log_msg.LOG_ERROR.format("process_frame/push_pending", str(e)))
                raise e
            
        if isinstance(frame, AudioRawFrame):
            # 2. Process current frame (pass-through audio to downstream)
            if self._websocket:
                try:
                    await self._websocket.send(frame.audio)
                except Exception as e:
                    logger.error(log_msg.WS_SEND_ERROR.format(e))
                    raise e
            # 3. Push current frame
            await self.push_frame(frame, direction)
            
        else:
            # 4. Delegate other frames (Start/End/UserSpeaking) to super to handle state
            await super().process_frame(frame, direction)

        # 5. Also push any new transcripts that arrived while sending
        while not self._transcript_queue.empty():
            try:
                transcript_frame = self._transcript_queue.get_nowait()
                await self.push_frame(transcript_frame, direction)
            except Exception as e:
                logger.error(log_msg.LOG_ERROR.format("process_frame/push_subsequent", str(e)))
                raise e
        return None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Required abstract method implementation. Not used in this WebSocket logic.
        """
        yield None
