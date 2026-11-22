init_script="""(() => {
const Base = window.RTCPeerConnection;
window.__peers=[];
window.RTCPeerConnection = class extends Base {constructor(...args){super(...args);window.__peers.push(this);}};
const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia=async constraints => {
  if(!constraints || !constraints.audio) return original(constraints);
  const ctx=window.__ctx ||= new AudioContext({sampleRate:48000});
  await ctx.resume();
  if (!window.__dest) {
    const dest=ctx.createMediaStreamDestination();
    const zero=ctx.createConstantSource(); zero.offset.value=0; zero.connect(dest); zero.start();
    window.__dest=dest;window.__zero=zero;
  }
  return window.__dest.stream.clone();
};
})();"""
play_audio="""async (data) => {
const ctx=window.__ctx; await ctx.resume();
const array=Uint8Array.from(atob(data),c=>c.charCodeAt(0));
const buffer=await ctx.decodeAudioData(array.buffer);
const source=ctx.createBufferSource();source.buffer=buffer;source.connect(window.__dest);
await new Promise(done=>{source.onended=done;source.start();});
return {rate:buffer.sampleRate,seconds:buffer.duration};
}"""
import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from playwright.async_api import async_playwright
from fourier_fastrtc import FourierSTT, build_stream, _TurnWithPreRoll

tokens = lambda text: re.findall(r"[a-z]+", text.lower())
expected = {
    'question': 'What day is the meeting?',
    'answer': 'The meeting is on Friday.',
    'human': 'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.',
}
window_edges = {}
prefix_lengths = {}
segments = []
actual = _TurnWithPreRoll.determine_pause

def trace(self, audio, rate, state):
    used_to_talk = state.started_talking
    prefix = len(state.stream) if state.stream is not None else 0
    paused = actual(self, audio, rate, state)
    if state.buffer is None:
        if not state.started_talking:
            window_edges[self] = time.monotonic()
        elif not used_to_talk:
            prefix_lengths[self] = prefix
        if paused:
            segments.append((rate, state.stream.copy(), prefix_lengths.pop(self, 0)))
    return paused

_TurnWithPreRoll.determine_pause = trace

class ObservedModel(FourierSTT):
    def __init__(self):
        super().__init__()
        self.answers = []
    def stt(self, audio):
        answer = super().stt(audio)
        self.answers.append(answer)
        return answer

async def main():
    asr = ObservedModel()
    stream = build_stream(asr)
    block = stream.ui
    block.launch(server_name='127.0.0.1',server_port=8765,prevent_thread_lock=True,quiet=True)
    records = []
    try:
        async with async_playwright() as play:
            browser=await play.chromium.launch(executable_path=os.environ.get('CHROME_BIN','/usr/bin/google-chrome'),args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--autoplay-policy=no-user-gesture-required'])
            context=await browser.new_context(permissions=['microphone'])
            await context.add_init_script(script=init_script)
            page=await context.new_page()
            await page.goto('http://127.0.0.1:8765',wait_until='domcontentloaded',timeout=45000)
            start=page.get_by_role("button",name="start stream",exact=True)
            await start.wait_for(timeout=45000)
            out=page.get_by_role('textbox',name='Last completed utterance')
            await out.wait_for(state='visible')
            for i,(name, phase, reconnect) in enumerate([('question',.12,False),('answer',.72,False),('human',.66,True),('answer',.82,False),('answer',.61,False)]):
                if not i or reconnect:
                    if i:
                        await start.click()
                        await page.wait_for_function("window.__peers.every(p=>p.connectionState!=='connected')",timeout=25000)
                    await start.click()
                    await page.wait_for_function("window.__peers.some(p=>p.connectionState==='connected')",timeout=40000)
                await asyncio.sleep(2.2)
                assert len(asr.answers)==i,(i,asr.answers)
                latest=max(window_edges.values(),default=0)
                deadline=time.monotonic()+8
                while max(window_edges.values(),default=0)<=latest and time.monotonic()<deadline:
                    await asyncio.sleep(.012)
                boundary=max(window_edges.values(),default=0)
                assert boundary>latest
                await asyncio.sleep(max(0,boundary+phase-time.monotonic()))
                wave=base64.b64encode(Path(name+'.wav').read_bytes()).decode()
                await page.evaluate(play_audio,wave)
                until=time.monotonic()+95
                while len(asr.answers)<=i and time.monotonic()<until:
                    await asyncio.sleep(.09)
                assert len(asr.answers)==i+1 and len(segments)==i+1,(i,asr.answers,len(segments))
                await page.wait_for_function('(s)=>Array.from(document.querySelectorAll("textarea")).some(t=>t.value.trim()===s)',arg=asr.answers[-1],timeout=15000)
                rate,audio,pre=segments[-1];pre_audio=audio[:pre].astype('float64')/32768
                frame_len=rate//100
                energy=np.sqrt(np.mean(pre_audio[:len(pre_audio)//frame_len*frame_len].reshape(-1,frame_len)**2,axis=1)) if pre>=frame_len else np.array([])
                records.append(dict(fixture=name,phase=phase,reconnect=reconnect,expected=expected[name],fixed=asr.answers[-1],pre_roll_ms=1000*pre/rate,pre_roll_nonsilent_ms=10*int((energy>.008).sum()),total_sec=len(audio)/rate))
                print('BROWSER_TURN',json.dumps(records[-1]),flush=True)
            await asyncio.sleep(3);assert len(asr.answers)==len(records)
            await browser.close()
    finally:
        block.close()
    for row,(rate,audio,pre) in zip(records,segments):
        # Upstream 0.0.34 delivers the same complete PCM without the saved preceding window.
        row['stock_pcm_transcript']=FourierSTT.stt(asr,(rate,audio[pre:]))
        print('PAIRED_PCM',json.dumps(row),flush=True)
    Path('result.json').write_text(json.dumps(records,indent=2))
    assert all(tokens(row['fixed'])==tokens(row['expected']) for row in records[:3]),records[:3]
    assert any(tokens(row['stock_pcm_transcript'])==tokens(row['expected'])[1:] and tokens(row['fixed'])==tokens(row['expected']) for row in records),records
asyncio.run(main())
