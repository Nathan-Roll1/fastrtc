"""Local, explicitly paired audio relay with English captions from Oruk Fourier."""
import asyncio
import json
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastrtc import AsyncStreamHandler, Stream, get_current_context
from fastrtc.stream import Body
from starlette.middleware.trustedhost import TrustedHostMiddleware

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fourier_fastrtc import FourierSTT

RATE = 48000
MAX_TURN = 30
ROOM_TIMEOUT = 300


@dataclass(eq=False)
class Member:
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    peer: "Member | None" = None
    handler: "Relay | None" = None
    created: float = field(default_factory=monotonic)
    offered: bool = False


class Rooms:
    def __init__(self):
        self.members = {}
        self.invites = {}

    async def close(self, member):
        if self.members.get(member.token) is not member:
            return
        participants = (member, member.peer)
        for item in participants:
            if item:
                self.members.pop(item.token, None)
                item.peer = None
                if item.handler:
                    item.handler.clear()
                    await item.handler.notify("closed")
        self.invites = {key: val for key, val in self.invites.items() if val not in participants}

    async def create(self):
        for value in list(self.members.values()):
            if (not value.handler or not value.peer or not value.peer.handler) and monotonic() - value.created > ROOM_TIMEOUT:
                await self.close(value)
        if len(self.members) >= 14:
            raise HTTPException(429, "Rooms are full. Leave a room and try again.")
        invite = secrets.token_urlsafe(20)
        member = Member()
        self.members[member.token] = self.invites[invite] = member
        return {"member": member.token, "invite": invite}

    def join(self, invite):
        host = self.invites.pop(invite, None)
        if not host or monotonic() - host.created > ROOM_TIMEOUT:
            raise HTTPException(404, "This invite has expired or was already used.")
        guest = Member(peer=host)
        host.peer = guest
        self.members[guest.token] = guest
        return {"member": guest.token}


class Relay(AsyncStreamHandler):
    def __init__(self, rooms):
        super().__init__(expected_layout="mono", input_sample_rate=RATE, output_sample_rate=RATE)
        self.rooms = rooms
        self.member = None
        self.outgoing = asyncio.Queue(maxsize=25)
        self.clear()

    def clear(self):
        self.parts = None
        self.samples = 0
        self.finishing = False
        while not self.outgoing.empty():
            self.outgoing.get_nowait()

    def copy(self):
        return Relay(self.rooms)

    async def notify(self, event, **values):
        if self.channel and self.channel.readyState == "open":
            await self.send_message(json.dumps({"event": event, **values}))

    async def start_up(self):
        member = self.rooms.members.get(get_current_context().webrtc_id)
        if member is None:
            return
        self.member = member
        member.handler = self
        if member.peer and member.peer.handler:
            await self.notify("paired")
            await member.peer.handler.notify("paired")

    async def receive(self, frame):
        member = self.member
        if not member or not member.peer or not member.peer.handler:
            return
        rate, pcm = frame
        queue = member.peer.handler.outgoing
        if queue.full():
            queue.get_nowait()
        queue.put_nowait((rate, pcm))
        if self.parts is not None and self.samples <= MAX_TURN * RATE:
            self.parts.append(pcm)
            self.samples += pcm.shape[-1]

    async def emit(self):
        if self.member and self.member.peer and self.member.peer.handler:
            # The sender's RTP frames already contain the playout clock. Inserting
            # silence on each late packet stretches words on the receiving track.
            return await self.outgoing.get()
        await asyncio.sleep(.02)
        return RATE, np.zeros((1, RATE // 50), dtype=np.int16)

    async def shutdown(self):
        if self.member:
            await self.rooms.close(self.member)


def create_app(recognizer):
    rooms = Rooms()
    stream = Stream(handler=Relay(rooms), mode="send-receive", modality="audio",
                    concurrency_limit=16, rtc_configuration={"iceServers": []},
                    server_rtc_configuration={"iceServers": []})
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    auth = HTTPBearer()

    def member(credentials: HTTPAuthorizationCredentials = Depends(auth)):
        value = rooms.members.get(credentials.credentials)
        if not value:
            raise HTTPException(403, "This room is closed.")
        return value

    @app.get("/")
    async def index():
        return FileResponse(HERE / "index.html")

    @app.post("/rooms")
    async def create(request: Request):
        if request.headers.get("X-Local-Room") != "1":
            raise HTTPException(403, "Use the local conversation page.")
        return await rooms.create()

    @app.post("/join")
    async def join(request: Request):
        return rooms.join((await request.json()).get("invite", ""))

    @app.post("/offer")
    async def offer(body: Body, who=Depends(member)):
        if body.webrtc_id != who.token or body.type != "offer" or who.offered:
            raise HTTPException(409, "This offer cannot be used.")
        who.offered = True
        return await stream.offer(body)

    def paired(who=Depends(member)):
        if not who.handler or not who.peer or not who.peer.handler:
            raise HTTPException(409, "Wait for the other participant.")
        return who

    @app.post("/turn/start")
    async def start(who=Depends(paired)):
        h = who.handler
        if h.parts is not None or h.finishing:
            raise HTTPException(409, "A turn is already open.")
        h.parts = []
        h.samples = 0
        return {"ok": True}

    @app.post("/turn/end")
    async def finish(who=Depends(paired)):
        h = who.handler
        peer = who.peer
        if h.parts is None or h.finishing:
            raise HTTPException(409, "Start a turn first.")
        h.finishing = True
        try:
            await asyncio.sleep(.42)  # Drain audio already in the browser/Opus jitter buffers.
            if who.peer is not peer:
                raise HTTPException(410, "This conversation ended.")
            parts, h.parts = h.parts, None
            if h.samples > RATE * MAX_TURN:
                raise HTTPException(413, "Please keep each turn under 30 seconds.")
            if parts:
                samples = np.concatenate(parts, axis=-1)
                caption = await asyncio.to_thread(recognizer.stt, (RATE, samples))
                if caption and who.peer is peer and peer.handler:
                    await peer.handler.notify("caption", text=caption)
            return {"ok": True}
        finally:
            h.finishing = False

    @app.post("/leave")
    async def leave(who=Depends(member)):
        await rooms.close(who)
        return {"ok": True}

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(FourierSTT()), host="127.0.0.1", port=7861, access_log=False)
