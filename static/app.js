const img     = document.getElementById('stream');
const logEl   = document.getElementById('log');
const statusEl = document.getElementById('status');
const connEl  = document.getElementById('conn');

let ws = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/control`);

  ws.onopen = () => setConnState(true);
  ws.onclose = () => {
    setConnState(false);
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.logs) {
      for (const line of msg.logs) appendLog(line);
    }
    if (msg.state) updateStatus(msg.state);
  };
}

function setConnState(connected) {
  connEl.textContent = connected ? 'connected' : 'reconnecting…';
  connEl.className = connected ? 'ok' : 'bad';
}

function appendLog(line) {
  const div = document.createElement('div');
  div.textContent = line;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
}

function updateStatus(state) {
  const bits = [];
  if (state.pending_role) {
    bits.push(`ASSIGN → click the person to make them ${state.pending_role}`);
  } else if (state.drawing_mode) {
    bits.push('DRAWING BOUNDARY — click corners, then Finish');
  } else {
    bits.push('EDIT MODE — drag a corner to move it');
  }
  bits.push(state.use_pose ? 'POSE (ankles)' : 'BBOX (bottom-centre)');
  const pen = state.penalty_counts || {};
  bits.push(`Penalties  F1:${pen.F1 ?? 0}  F2:${pen.F2 ?? 0}`);
  statusEl.textContent = bits.join('   |   ');
}

function sendMsg(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function sendKey(key) {
  sendMsg({ type: 'key', key });
}

function toImgCoords(e) {
  const rect = img.getBoundingClientRect();
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  return [
    Math.round((e.clientX - rect.left) * scaleX),
    Math.round((e.clientY - rect.top) * scaleY),
  ];
}

// ── Mouse interaction on the video (click-to-assign, drag-to-move corner) ──
img.addEventListener('contextmenu', (e) => e.preventDefault());

img.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  const [x, y] = toImgCoords(e);
  sendMsg({ type: 'mouse', event: 'down', x, y });
});

img.addEventListener('mousemove', (e) => {
  if (e.buttons !== 1) return;   // only while left button held (dragging a corner)
  const [x, y] = toImgCoords(e);
  sendMsg({ type: 'mouse', event: 'move', x, y });
});

window.addEventListener('mouseup', (e) => {
  if (e.button !== 0) return;
  const [x, y] = toImgCoords(e);
  sendMsg({ type: 'mouse', event: 'up', x, y });
});

// Touch support (tap = click, drag = move corner)
img.addEventListener('touchstart', (e) => {
  const t = e.touches[0];
  const [x, y] = toImgCoords(t);
  sendMsg({ type: 'mouse', event: 'down', x, y });
  e.preventDefault();
}, { passive: false });

img.addEventListener('touchmove', (e) => {
  const t = e.touches[0];
  const [x, y] = toImgCoords(t);
  sendMsg({ type: 'mouse', event: 'move', x, y });
  e.preventDefault();
}, { passive: false });

img.addEventListener('touchend', (e) => {
  sendMsg({ type: 'mouse', event: 'up', x: 0, y: 0 });
}, { passive: false });

// ── Buttons ──────────────────────────────────────────────────────────────
document.querySelectorAll('[data-key]').forEach((btn) => {
  btn.addEventListener('click', () => sendKey(btn.dataset.key));
});

// ── Keyboard shortcuts (mirrors the local OpenCV runner) ───────────────────
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'Escape') {
    sendKey('esc');
    return;
  }
  if (e.key.length === 1) {
    sendKey(e.key);
    e.preventDefault();
  }
});

connect();
