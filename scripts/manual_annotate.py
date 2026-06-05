#!/usr/bin/env python3
"""
Manual annotation tool — 브라우저 기반 좌표 클릭 도구.

python3 scripts/manual_annotate.py
→ 브라우저에서 http://localhost:8765 열기

Controls (브라우저 키보드):
  T         — target 모드 전환
  D         — dest 모드 전환
  N / Enter — 저장 후 다음 이미지
  R         — 현재 이미지 리셋
  ← / →    — 이전/다음 이동 (수정 가능)
  우클릭    — 마지막 좌표 제거

JSON 저장 위치: dataset/manual_annotations.json
  { "s1_trial_001": { "target": [[x,y], ...], "dest": [[x,y], ...] }, ... }
  target/dest가 없는 경우(INVALID): 빈 리스트 []로 저장
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]


def load_trials(manifest_path: Path) -> list[dict]:
    trials = []
    seen: set[str] = set()
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        tid = entry["id"]
        if tid not in seen:
            seen.add(tid)
            trials.append(entry)
    return trials


def get_checkpoint_img(trial: dict) -> Path:
    ck = trial.get("checkpoint", "C1")
    key = "c1_img" if ck == "C1" else "c2_img"
    p = Path(trial[key])
    if not p.is_absolute():
        p = REPO / p
    return p


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Manual Annotate</title>
<style>
  body { margin:0; background:#111; color:#ccc; font-family:monospace; font-size:13px; display:flex; flex-direction:column; height:100vh; }
  #header { background:#1a1a1a; padding:8px 12px; border-bottom:1px solid #333; display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  #progress { color:#888; }
  #trial-id { color:#fff; font-weight:bold; font-size:14px; }
  #gold-state { padding:2px 8px; border-radius:3px; font-weight:bold; }
  .gs-CLEAR { background:#1a4a1a; color:#4f4; }
  .gs-AMBIGUOUS_TARGET { background:#3a2e00; color:#fc0; }
  .gs-AMBIGUOUS_DESTINATION { background:#2a1a3a; color:#c8a0ff; }
  .gs-INVALID_TARGET { background:#3a1a1a; color:#f44; }
  .gs-INVALID_DESTINATION { background:#3a1a1a; color:#f88; }
  #mode-btn { padding:4px 12px; border:2px solid #888; border-radius:4px; cursor:pointer; font-family:monospace; font-size:13px; background:#222; color:#ccc; }
  #mode-btn.target { border-color:#f44; color:#f44; }
  #mode-btn.dest { border-color:#48f; color:#48f; }
  #labels { color:#aaa; }
  #counts { }
  #counts .t { color:#f44; font-weight:bold; }
  #counts .d { color:#48f; font-weight:bold; }
  #controls { color:#666; font-size:11px; margin-left:auto; }
  #canvas-wrap { flex:1; overflow:auto; display:flex; justify-content:center; align-items:flex-start; padding:16px; }
  canvas { cursor:crosshair; border:1px solid #333; }
  #status { background:#1a1a1a; padding:6px 12px; border-top:1px solid #333; height:22px; }
  #status.ok { color:#4f4; }
  #status.err { color:#f44; }
  #nav { display:flex; gap:8px; }
  button { padding:4px 14px; border:1px solid #555; background:#222; color:#ccc; border-radius:3px; cursor:pointer; font-family:monospace; font-size:12px; }
  button:hover { background:#333; }
  button.primary { border-color:#4a8; color:#4f4; }
  button.primary:hover { background:#1a3a2a; }
  #thumbnail-strip { background:#151515; padding:6px 12px; border-top:1px solid #2a2a2a; display:flex; gap:6px; overflow-x:auto; align-items:center; }
  .thumb { width:60px; height:45px; border:2px solid transparent; cursor:pointer; object-fit:cover; border-radius:2px; opacity:0.5; }
  .thumb.current { border-color:#fff; opacity:1; }
  .thumb.done { border-color:#4f4; opacity:0.7; }
  .thumb.done.current { border-color:#fff; }
</style>
</head>
<body>
<div id="header">
  <span id="progress">0/0</span>
  <span id="trial-id">—</span>
  <span id="gold-state">—</span>
  <span id="labels">—</span>
  <span id="counts">T: <span class="t">0</span>  D: <span class="d">0</span></span>
  <button id="mode-btn" class="target" onclick="toggleMode()">T — TARGET</button>
  <div id="nav">
    <button onclick="navigate(-1)">← Prev</button>
    <button class="primary" onclick="saveAndNext()">Save &amp; Next (N)</button>
    <button onclick="resetCurrent()">Reset (R)</button>
  </div>
  <div id="controls">T=target D=dest N=next R=reset ←→=nav  우클릭=제거</div>
</div>
<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>
<div id="status"></div>

<script>
let trials = [];
let annotations = {};
let idx = 0;
let mode = 'target';  // 'target' | 'dest'
let tgtCoords = [];
let dstCoords = [];
let img = new Image();
let imgLoaded = false;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function init() {
  const r = await fetch('/api/state');
  const data = await r.json();
  trials = data.trials;
  annotations = data.annotations;
  // find first unannotated
  idx = trials.findIndex(t => !(t.id in annotations));
  if (idx < 0) idx = 0;
  loadTrial(idx);
}

function loadTrial(i) {
  idx = i;
  const t = trials[i];
  if (!t) return;
  // load existing annotation if any
  if (t.id in annotations) {
    tgtCoords = (annotations[t.id].target || []).map(c => [...c]);
    dstCoords = (annotations[t.id].dest   || []).map(c => [...c]);
  } else {
    tgtCoords = [];
    dstCoords = [];
  }
  mode = 'target';
  updateHeader();
  imgLoaded = false;
  img = new Image();
  img.onload = () => { imgLoaded = true; render(); };
  img.src = '/api/image/' + i;
}

function render() {
  if (!imgLoaded) return;
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  // target coords (red)
  tgtCoords.forEach((c, i) => drawDot(c[0], c[1], 'rgba(255,50,50,0.85)', `T${i}`));
  // dest coords (blue)
  dstCoords.forEach((c, i) => drawDot(c[0], c[1], 'rgba(60,130,255,0.85)', `D${i}`));
  updateCounts();
}

function drawDot(x, y, color, label) {
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI*2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 12px monospace';
  ctx.fillText(label, x + 13, y + 4);
}

function updateHeader() {
  const t = trials[idx];
  document.getElementById('progress').textContent = `${idx+1}/${trials.length}`;
  document.getElementById('trial-id').textContent = t.id;
  const gs = document.getElementById('gold-state');
  gs.textContent = t.gold_state;
  gs.className = `gs-${t.gold_state}`;
  document.getElementById('labels').textContent =
    `target="${t.target_label}"  dest="${t.destination_label}"  ck=${t.checkpoint}`;
  updateModeBtn();
}

function updateModeBtn() {
  const btn = document.getElementById('mode-btn');
  const t = trials[idx];
  if (mode === 'target') {
    btn.textContent = `T — TARGET (${t.target_label})`;
    btn.className = 'target';
  } else {
    btn.textContent = `D — DEST (${t.destination_label})`;
    btn.className = 'dest';
  }
}

function updateCounts() {
  document.querySelector('#counts .t').textContent = tgtCoords.length;
  document.querySelector('#counts .d').textContent = dstCoords.length;
}

function toggleMode() {
  mode = mode === 'target' ? 'dest' : 'target';
  updateModeBtn();
}

canvas.addEventListener('click', e => {
  if (!imgLoaded) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top)  * scaleY);
  if (mode === 'target') tgtCoords.push([x, y]);
  else                   dstCoords.push([x, y]);
  render();
});

canvas.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (mode === 'target' && tgtCoords.length) tgtCoords.pop();
  else if (mode === 'dest' && dstCoords.length) dstCoords.pop();
  render();
});

async function saveAndNext() {
  const t = trials[idx];
  const payload = { id: t.id, target: tgtCoords, dest: dstCoords };
  const r = await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const res = await r.json();
  if (res.ok) {
    annotations[t.id] = { target: tgtCoords, dest: dstCoords };
    setStatus(`저장: ${t.id}  target=${tgtCoords.length}개  dest=${dstCoords.length}개`, 'ok');
    // advance to next unannotated
    let next = -1;
    for (let i = idx+1; i < trials.length; i++) {
      if (!(trials[i].id in annotations)) { next = i; break; }
    }
    if (next >= 0) loadTrial(next);
    else {
      for (let i = 0; i <= idx; i++) {
        if (!(trials[i].id in annotations)) { loadTrial(i); return; }
      }
      setStatus(`모든 ${trials.length}개 이미지 annotation 완료! 🎉`, 'ok');
    }
  } else {
    setStatus('저장 실패: ' + res.error, 'err');
  }
}

function resetCurrent() {
  tgtCoords = [];
  dstCoords = [];
  render();
}

function navigate(delta) {
  const next = Math.max(0, Math.min(trials.length - 1, idx + delta));
  loadTrial(next);
}

function setStatus(msg, cls) {
  const s = document.getElementById('status');
  s.textContent = msg;
  s.className = cls || '';
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key.toLowerCase()) {
    case 't': mode = 'target'; updateModeBtn(); break;
    case 'd': mode = 'dest';   updateModeBtn(); break;
    case 'n': case 'enter': saveAndNext(); break;
    case 'r': resetCurrent(); break;
    case 'arrowleft':  navigate(-1); break;
    case 'arrowright': navigate(+1); break;
  }
});

init();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    trials: list[dict] = []
    ann_path: Path = REPO / "dataset/manual_annotations.json"
    annotations: dict = {}

    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode())

        elif path == "/api/state":
            payload = json.dumps({
                "trials": self.trials,
                "annotations": self.annotations,
            }).encode()
            self._send(200, "application/json", payload)

        elif path.startswith("/api/image/"):
            try:
                i = int(path.split("/")[-1])
                trial = self.trials[i]
                img_path = get_checkpoint_img(trial)
                data = img_path.read_bytes()
                mt = mimetypes.guess_type(str(img_path))[0] or "image/png"
                self._send(200, mt, data)
            except Exception as e:
                self._send(404, "text/plain", str(e).encode())

        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                tid = body["id"]
                self.annotations[tid] = {
                    "target": body.get("target", []),
                    "dest":   body.get("dest",   []),
                }
                self.ann_path.write_text(
                    json.dumps(self.annotations, indent=2, ensure_ascii=False)
                )
                done = sum(1 for t in self.trials if t["id"] in self.annotations)
                total = len(self.trials)
                print(f"  [{done:3d}/{total}] {tid}: "
                      f"target={len(body.get('target',[]))}개  "
                      f"dest={len(body.get('dest',[]))}개")
                self._send(200, "application/json", json.dumps({"ok": True}).encode())
            except Exception as e:
                self._send(500, "application/json",
                           json.dumps({"ok": False, "error": str(e)}).encode())
        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Manual annotation tool")
    parser.add_argument("--manifest", default="dataset/manifest_train_v3.jsonl")
    parser.add_argument("--out", default="dataset/manual_annotations.json")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    manifest_path = REPO / args.manifest
    ann_path = REPO / args.out

    trials = load_trials(manifest_path)
    annotations: dict = {}
    if ann_path.exists():
        annotations = json.loads(ann_path.read_text())
        done = sum(1 for t in trials if t["id"] in annotations)
        print(f"기존 annotation 로드: {done}/{len(trials)}개")

    Handler.trials = trials
    Handler.ann_path = ann_path
    Handler.annotations = annotations

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n총 {len(trials)}개 trial")
    print(f"브라우저에서 열기: http://localhost:{args.port}")
    print(f"저장 위치: {ann_path}")
    print("종료: Ctrl+C\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
