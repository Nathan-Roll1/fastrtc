"""Two Chrome processes, distinct source voices, real remote audio and isolated captions."""
import asyncio
import base64
import io
import json
import math
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:7861"
HELPERS = """
(() => {
 const Native = window.RTCPeerConnection;
 window.RTCPeerConnection = class extends Native { constructor(...args) { super(...args); window.auditPc = this; } };
 let ctx, dest;
 navigator.mediaDevices.getUserMedia = async () => {
   ctx = new AudioContext({sampleRate:48000}); await ctx.resume(); dest = ctx.createMediaStreamDestination();
   const constant=ctx.createConstantSource(); constant.offset.value=0; constant.connect(dest); constant.start();
   return dest.stream;
 };
 window.feed = async encoded => {
   const original = Uint8Array.from(atob(encoded), x=>x.charCodeAt(0));
   const buf = await ctx.decodeAudioData(original.buffer);
   const source=ctx.createBufferSource(); source.buffer=buf; source.connect(dest);
   await new Promise(ok=>{source.onended=ok;source.start();});
 };
 window.listen = async () => {
   const rx = new AudioContext({sampleRate:16000}); await rx.resume();
   const track=document.querySelector("audio").srcObject.getAudioTracks()[0];
   const source=rx.createMediaStreamSource(new MediaStream([track]));
   const proc=rx.createScriptProcessor(1024,1,1), silent=rx.createGain(); silent.gain.value=0;
   const data=[]; proc.onaudioprocess=e=>data.push(new Float32Array(e.inputBuffer.getChannelData(0)));
   source.connect(proc);proc.connect(silent);silent.connect(rx.destination);
   window.stopListen=async()=>{
     proc.disconnect(); source.disconnect(); await rx.close();
     const n=data.reduce((s,a)=>s+a.length,0), result=new Int16Array(n); let i=0;
     for(const a of data)for(const x of a)result[i++]=Math.round(32767*Math.max(-1,Math.min(1,x)));
     const bytes = new Uint8Array(result.buffer); let binary="";
     for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
     return btoa(binary);
   };
 };
 window.auditStats=async()=>{
   const report=await window.auditPc.getStats();const res={sent:0,received:0,receivedTrack:document.querySelector("audio").srcObject.getAudioTracks()[0].readyState};
   for(const s of report.values())if(s.kind==="audio"){
      if(s.type==="inbound-rtp")res.received+=s.packetsReceived||0;
      if(s.type==="outbound-rtp")res.sent+=s.packetsSent||0;
   }return res;
 };
})();
"""


def expected_words(text):
    text = re.sub(r"\$\s*8(?:\.00)?\b", "eight dollars",text.lower())
    text = re.sub(r"\b8\b", "eight", text)
    return re.findall(r"[a-z]+", text.replace("’","'"))


def quality(encoded, original):
    captured = np.frombuffer(base64.b64decode(encoded), dtype="<i2").astype(float)/32768
    rate,x=wavfile.read(io.BytesIO(original))
    x=np.asarray(x,dtype=float)
    if x.ndim>1: x=x.mean(axis=1)
    factor=math.gcd(rate,16000)
    x=resample_poly(x,16000//factor,rate//factor)
    x=x-np.mean(x)
    assert len(captured)>=len(x),(len(captured),len(x))
    n=len(x)
    score=correlate(captured,x,mode="valid",method="fft")
    csum=np.r_[0,np.cumsum(captured*captured)]
    scaled=np.abs(score)/np.sqrt(np.maximum(csum[n:]-csum[:-n],1e-9)*np.sum(x*x))
    index=int(np.argmax(scaled))
    return {"peak_normalized_waveform_correlation":round(float(scaled[index]),4),
            "alignment_seconds":round(index/16000,3),
            "received_seconds":round(len(captured)/16000,3),
            "peak_abs_pcm":round(float(np.max(np.abs(captured))),4)}


async def main():
    for _ in range(420):
        try:
            with urllib.request.urlopen(BASE,timeout=2) as r:
                if r.status==200: break
        except OSError: pass
        await asyncio.sleep(1)
    else: raise RuntimeError("Model server did not start")
    clips=[Path(x+".wav").read_bytes() for x in ("a","b")]
    proof={"workflow":os.getenv("GITHUB_RUN_ID"), "software":{"fastrtc":"0.0.34","transcribe-cpp":"0.4.2"},
           "model_id":"oruk/oruk-fourier-2b", "model_revision":"8bcce21695e4ba13569c48de6cedffcb8a7576f5",
           "fixture":"Two separate CMU ARCTIC prompts assembled into a scripted question and reply.",
           "sources":["bdl-a0207.wav","clb-b0260.wav"],"turns":[]}
    def save(): Path("verification.json").write_text(json.dumps(proof,indent=2)+"\n")
    async with async_playwright() as p:
        browsers=[await p.chromium.launch(executable_path=shutil.which("google-chrome"),args=["--no-sandbox","--autoplay-policy=no-user-gesture-required","--disable-background-timer-throttling","--disable-renderer-backgrounding"]) for _ in range(2)]
        pages=[]
        for i in range(4):
            ctx=await browsers[i%2].new_context()
            await ctx.add_init_script(HELPERS)
            page=await ctx.new_page()
            page.on("pageerror",lambda e: print("PAGE ERROR",e))
            await page.goto(BASE)
            pages.append(page)
        a,b,c,d=pages
        async def pair(first,second):
            await first.locator("#create").click()
            await first.wait_for_function("document.querySelector('#code').textContent.startsWith('Invite: ')")
            invite=(await first.locator("#code").inner_text()).split()[-1]
            await second.locator("#invite").fill(invite)
            await second.locator("#join").click()
            for pg in (first,second):
                try: await pg.wait_for_function("document.querySelector('#status').textContent === 'Paired' && document.querySelector('audio').srcObject",timeout=45000)
                except Exception:
                    print("failed to pair",[(await item.locator('#status').inner_text()) for item in (first,second)]); raise
            denied=await first.request.post(BASE+"/join",data={"invite":invite})
            assert denied.status==404
        await pair(a,b)
        await pair(c,d)
        await asyncio.sleep(1)
        print("Four web clients connected to two explicit rooms")
        await c.evaluate("listen()")
        await asyncio.gather(a.evaluate("listen()"),b.evaluate("listen()"))
        try:
            for i,(sender,recipient,words) in enumerate([(a,b,"How much was it?"),(b,a,"It's worth eight dollars.")]):
                await sender.locator("#talk").click()
                await sender.wait_for_function("document.querySelector('#talk').textContent==='Finish turn'")
                await asyncio.sleep(.32)
                await sender.evaluate("feed",base64.b64encode(clips[i]).decode())
                await asyncio.sleep(.23)
                await sender.locator("#talk").click()
                await recipient.locator("#captions li").first.wait_for(timeout=180000)
                text=await recipient.locator("#captions li").inner_text()
                heard=await recipient.evaluate("stopListen()")
                metrics=quality(heard, clips[i])
                proof["turns"].append({"receiver":"B" if i==0 else "A","reference":words,"raw_partner_caption":text,**metrics})
                save()
                print(proof["turns"][-1])
                assert expected_words(text)==expected_words(words),text
                assert metrics["peak_normalized_waveform_correlation"]>.45,metrics
                assert await sender.locator("#captions li").count()==i
            background=np.frombuffer(base64.b64decode(await c.evaluate("stopListen()")),dtype="<i2")
            proof["other_paired_room"]={"clients":2,"captions":[await pg.locator("#captions li").count() for pg in (c,d)],
                                         "max_pcm_amplitude":int(np.abs(background.astype(int)).max())}
            assert proof["other_paired_room"]["captions"]==[0,0]
            assert proof["other_paired_room"]["max_pcm_amplitude"]<200
            proof["rtp"]={k:await pg.evaluate("auditStats()") for k,pg in zip("ABCD",pages)}
            assert all(v["sent"]>0 and v["received"]>0 and v["receivedTrack"]=="live" for v in proof["rtp"].values())
            await a.locator("#leave").click()
            await b.wait_for_function("document.querySelector('#status').textContent==='Room closed'")
            assert await a.locator("#captions li").count()==await b.locator("#captions li").count()==0
            await pair(a,b)
            assert await a.locator("#captions li").count()==await b.locator("#captions li").count()==0
            proof.update(rejoin_empty=True,invite_single_use=True,independent_browser_processes=2,isolated_sessions=4,passed=True)
            save(); print(json.dumps(proof,indent=2))
        finally:
            save()
            for browser in browsers: await browser.close()


asyncio.run(main())
