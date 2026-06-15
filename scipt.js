/* ====================================================================
   AutonomDrive Dashboard — script.js
   Full in-browser simulation: RL agent, road physics, canvas renderer
   ==================================================================== */

"use strict";

// ── State ──────────────────────────────────────────────────────────
const SIM = {
  running: false,
  paused: false,
  intervalId: null,
  stepsPerTick: 1,
  episode: 0,
  step: 0,
  totalEpisodes: 100,

  // Session totals
  totalCompletions: 0,
  totalCollisions: 0,
  totalDistanceKm: 0,

  // Episode history (for chart)
  rewardHistory: [],
  avgHistory: [],

  // Action usage counts
  actionCounts: [0, 0, 0, 0, 0],
  logCount: 0,
};

const ACTIONS = ["Accelerate", "Brake", "Lane Left", "Lane Right", "Maintain"];
const LANE_COUNT = 3;
const ROAD_LEN   = 1000;
const DT         = 1 / 30;

// ── Car State ──────────────────────────────────────────────────────
let car = { lane: 1, pos: 0, speed: 30, heading: 0, targetLane: 1, laneProgress: 0 };
let obstacles = [];
let sensors   = [-90, -45, 0, 45, 90];  // angles
let sensorReadings = [0, 0, 0, 0, 0];

// ── Simple Q-table agent (no torch needed) ─────────────────────────
const N_STATES  = 243; // 3^5 discretised sensor buckets
const N_ACTIONS = 5;
let qTable = Array.from({ length: N_STATES }, () => Array(N_ACTIONS).fill(0));
let epsilon = 1.0;
const EPSILON_MIN   = 0.05;
const EPSILON_DECAY = 0.9995;
const GAMMA         = 0.99;
const ALPHA         = 0.1;

function stateIndex(readings) {
  // Discretise each sensor into 0/1/2 (clear/near/danger)
  const buckets = readings.map(r => r < 0.3 ? 0 : r < 0.7 ? 1 : 2);
  return buckets.reduce((acc, b, i) => acc + b * Math.pow(3, i), 0);
}

function chooseAction(readings) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * N_ACTIONS);
  const idx = stateIndex(readings);
  const q = qTable[idx];
  return q.indexOf(Math.max(...q));
}

function updateQ(readings, action, reward, nextReadings) {
  const s  = stateIndex(readings);
  const ns = stateIndex(nextReadings);
  const maxNext = Math.max(...qTable[ns]);
  qTable[s][action] += ALPHA * (reward + GAMMA * maxNext - qTable[s][action]);
  epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
}

// ── Road & Physics ─────────────────────────────────────────────────
function spawnObstacles(count) {
  obstacles = [];
  for (let i = 0; i < count; i++) {
    obstacles.push({
      lane: Math.floor(Math.random() * LANE_COUNT),
      pos: 80 + Math.random() * 860,
      speed: 15 + Math.random() * 35,
      type: ["vehicle", "truck", "cone", "debris"][Math.floor(Math.random() * 4)],
      laneTimer: 80 + Math.floor(Math.random() * 120)
    });
  }
}

function readSensors() {
  return sensors.map(angle => {
    const rad = (car.heading + angle) * Math.PI / 180;
    const dx = Math.cos(rad), dy = Math.sin(rad);
    let best = 150;
    for (const obs of obstacles) {
      const relPos  = obs.pos  - car.pos;
      const relLane = (obs.lane - car.lane) * 10;
      const t = relPos * dx + relLane * dy;
      if (t <= 0) continue;
      const perp = Math.abs(relPos * (-dy) + relLane * dx);
      if (perp > 5) continue;
      if (t < best) best = t;
    }
    return 1 - best / 150;
  });
}

function applyAction(action) {
  if (action === 0) car.speed = Math.min(120, car.speed + 5);
  else if (action === 1) car.speed = Math.max(0, car.speed - 7.5);
  else if (action === 2 && car.lane > 0 && car.laneProgress === 0) {
    car.targetLane = car.lane - 1; car.laneProgress = 3;
  } else if (action === 3 && car.lane < LANE_COUNT - 1 && car.laneProgress === 0) {
    car.targetLane = car.lane + 1; car.laneProgress = 3;
  }

  if (car.laneProgress > 0) {
    car.laneProgress--;
    car.heading = (car.targetLane - car.lane) * 5;
    if (car.laneProgress === 0) { car.lane = car.targetLane; car.heading = 0; }
  } else { car.heading *= 0.7; }

  car.pos += car.speed * DT;
}

function computeReward(action) {
  let r = 0.1;
  if (car.speed >= 30 && car.speed <= 80) r += 0.4;
  if (car.speed < 5) r -= 0.2;
  if (car.lane === 1) r += 0.05;

  for (const obs of obstacles) {
    const gap = Math.abs(car.pos - obs.pos);
    if (car.lane === obs.lane) {
      if (gap < 5)  return [-15, true, "collision"];
      if (gap < 20) r -= 0.5;
    }
  }
  if (car.lane < 0 || car.lane >= LANE_COUNT) return [-8, true, "off-road"];
  if (car.pos >= ROAD_LEN) return [25, true, "complete"];
  return [r, false, null];
}

function stepObstacles() {
  for (const obs of obstacles) {
    obs.pos += obs.speed * DT;
    if (obs.pos > ROAD_LEN) obs.pos = Math.random() * 60;
    obs.laneTimer--;
    if (obs.laneTimer <= 0 && obs.type === "vehicle") {
      const dir = Math.floor(Math.random() * 3) - 1;
      const nl  = obs.lane + dir;
      if (nl >= 0 && nl < LANE_COUNT) obs.lane = nl;
      obs.laneTimer = 80 + Math.floor(Math.random() * 120);
    }
  }
}

// ── Episode Management ─────────────────────────────────────────────
let epReward = 0;
let epSteps  = 0;
let prevReadings = [0, 0, 0, 0, 0];

function resetEpisode() {
  car = { lane: 1, pos: 0, speed: 25 + Math.random() * 10, heading: 0, targetLane: 1, laneProgress: 0 };
  spawnObstacles(+document.getElementById("obsSlider").value);
  prevReadings = [0, 0, 0, 0, 0];
  epReward = 0;
  epSteps  = 0;
  SIM.step = 0;
}

function tick() {
  if (!SIM.running) return;

  for (let i = 0; i < SIM.stepsPerTick; i++) {
    sensorReadings = readSensors();
    const action = chooseAction(sensorReadings);
    SIM.actionCounts[action]++;

    const prevR = [...sensorReadings];
    applyAction(action);
    stepObstacles();

    const nextR = readSensors();
    const [reward, done, event] = computeReward(action);
    updateQ(prevR, action, reward, nextR);

    epReward += reward;
    epSteps++;
    SIM.step++;
    SIM.totalDistanceKm += (car.speed * DT) / 1000;

    if (done || epSteps >= 600) {
      SIM.episode++;
      SIM.rewardHistory.push(+epReward.toFixed(2));

      const window10 = SIM.rewardHistory.slice(-10);
      SIM.avgHistory.push(+(window10.reduce((a, b) => a + b, 0) / window10.length).toFixed(2));

      if (event === "complete") {
        SIM.totalCompletions++;
        addLog(`Ep ${SIM.episode}: Route completed! Reward: ${epReward.toFixed(1)}`, "complete");
      } else if (event === "collision") {
        SIM.totalCollisions++;
        addLog(`Ep ${SIM.episode}: Collision at pos ${car.pos.toFixed(0)}`, "collision");
      } else if (event === "off-road") {
        addLog(`Ep ${SIM.episode}: Off-road exit`, "warn");
      } else {
        addLog(`Ep ${SIM.episode}: Timeout — ${epSteps} steps`, "info");
      }

      resetEpisode();

      if (SIM.episode >= SIM.totalEpisodes) {
        stopSim();
        addLog(`Training complete — ${SIM.episode} episodes`, "complete");
        return;
      }
    }
  }

  updateUI(sensorReadings, prevReadings);
}

// ── Canvas Renderer ────────────────────────────────────────────────
const CANVAS_H = 220;
const LANE_H   = 60;
const ROAD_TOP = (CANVAS_H - LANE_COUNT * LANE_H) / 2;

function lerp(a, b, t) { return a + (b - a) * t; }

function drawRoad() {
  const canvas = document.getElementById("roadCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  // Background
  ctx.fillStyle = "#0c1020";
  ctx.fillRect(0, 0, W, H);

  // Road surface
  const roadTop = ROAD_TOP;
  const roadH   = LANE_COUNT * LANE_H;
  ctx.fillStyle = "#1a2035";
  ctx.beginPath();
  ctx.roundRect(20, roadTop, W - 40, roadH, 4);
  ctx.fill();

  // Lane markings
  for (let i = 1; i < LANE_COUNT; i++) {
    const y = roadTop + i * LANE_H;
    ctx.setLineDash([20, 14]);
    ctx.strokeStyle = "#2a3850";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(20, y); ctx.lineTo(W - 20, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Road edges
  ctx.strokeStyle = "#3b4d6a";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(20, roadTop);         ctx.lineTo(W - 20, roadTop);         ctx.stroke();
  ctx.beginPath(); ctx.moveTo(20, roadTop + roadH); ctx.lineTo(W - 20, roadTop + roadH); ctx.stroke();

  // Progress bar at top
  const prog = car.pos / ROAD_LEN;
  ctx.fillStyle = "#1e2738";
  ctx.fillRect(20, roadTop - 10, W - 40, 4);
  ctx.fillStyle = "#00d4ff";
  ctx.fillRect(20, roadTop - 10, (W - 40) * prog, 4);

  // Obstacles
  for (const obs of obstacles) {
    const x = 20 + (obs.pos / ROAD_LEN) * (W - 40);
    const y = roadTop + obs.lane * LANE_H + LANE_H / 2;
    if (x < 20 || x > W - 20) continue;

    const colors = { vehicle: "#3b82f6", truck: "#f59e0b", cone: "#ef4444", debris: "#6b7a99" };
    const c = colors[obs.type] || "#6b7a99";

    // Glow
    const grd = ctx.createRadialGradient(x, y, 0, x, y, 22);
    grd.addColorStop(0, c + "44");
    grd.addColorStop(1, "transparent");
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(x, y, 22, 0, Math.PI * 2); ctx.fill();

    // Body
    ctx.fillStyle = c;
    if (obs.type === "truck") {
      ctx.fillRect(x - 20, y - 10, 40, 20);
    } else if (obs.type === "cone") {
      ctx.beginPath(); ctx.moveTo(x, y - 12); ctx.lineTo(x + 8, y + 8); ctx.lineTo(x - 8, y + 8); ctx.closePath(); ctx.fill();
    } else {
      ctx.beginPath(); ctx.roundRect(x - 14, y - 8, 28, 16, 3); ctx.fill();
    }
  }

  // Car
  const carScreenX = Math.max(60, Math.min(W - 60, 20 + (car.pos / ROAD_LEN) * (W - 40)));
  const carScreenY = roadTop + car.lane * LANE_H + LANE_H / 2;

  // Headlight glow
  const headGrd = ctx.createRadialGradient(carScreenX + 22, carScreenY, 0, carScreenX + 22, carScreenY, 40);
  headGrd.addColorStop(0, "rgba(0,212,255,0.3)");
  headGrd.addColorStop(1, "transparent");
  ctx.fillStyle = headGrd;
  ctx.beginPath(); ctx.arc(carScreenX + 22, carScreenY, 40, 0, Math.PI * 2); ctx.fill();

  // Car body
  ctx.save();
  ctx.translate(carScreenX, carScreenY);
  ctx.rotate(car.heading * Math.PI / 180);
  ctx.fillStyle = "#00d4ff";
  ctx.beginPath(); ctx.roundRect(-18, -9, 36, 18, 4); ctx.fill();
  ctx.fillStyle = "#001828";
  ctx.beginPath(); ctx.roundRect(-8, -6, 20, 12, 2); ctx.fill();
  // Headlights
  ctx.fillStyle = "#ffffff";
  ctx.beginPath(); ctx.arc(16, -5, 2.5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(16,  5, 2.5, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  // Sensor rays
  const rayColors = ["rgba(239,68,68,.5)", "rgba(245,158,11,.5)", "rgba(0,212,255,.4)", "rgba(245,158,11,.5)", "rgba(239,68,68,.5)"];
  sensors.forEach((angle, i) => {
    const rad = (car.heading + angle) * Math.PI / 180;
    const len = sensorReadings[i] * 80;
    if (len < 2) return;
    ctx.beginPath();
    ctx.moveTo(carScreenX, carScreenY);
    ctx.lineTo(carScreenX + Math.cos(rad) * len, carScreenY + Math.sin(rad) * len * 0.4);
    ctx.strokeStyle = rayColors[i];
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function drawRewardChart() {
  const canvas = document.getElementById("rewardChart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const data = SIM.rewardHistory;
  const avgs = SIM.avgHistory;

  ctx.fillStyle = "#0c1020";
  ctx.fillRect(0, 0, W, H);

  if (data.length < 2) return;

  const pad   = { top: 16, bottom: 28, left: 48, right: 16 };
  const inner = { w: W - pad.left - pad.right, h: H - pad.top - pad.bottom };
  const maxN  = Math.min(data.length, 120);
  const slice = data.slice(-maxN);
  const aslice = avgs.slice(-maxN);

  const allVals = [...slice, ...aslice].filter(v => isFinite(v));
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const rangeV = maxV - minV || 1;

  const xScale = i => pad.left + (i / (slice.length - 1)) * inner.w;
  const yScale = v => pad.top + (1 - (v - minV) / rangeV) * inner.h;

  // Grid lines
  ctx.strokeStyle = "#1e2738";
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const y = pad.top + t * inner.h;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const val = maxV - t * rangeV;
    ctx.fillStyle = "#3d4a66";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText(val.toFixed(1), 4, y + 3);
  });

  // Reward area fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  grad.addColorStop(0, "rgba(0,212,255,0.2)");
  grad.addColorStop(1, "rgba(0,212,255,0)");
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(slice[0]));
  slice.forEach((v, i) => ctx.lineTo(xScale(i), yScale(v)));
  ctx.lineTo(xScale(slice.length - 1), H - pad.bottom);
  ctx.lineTo(xScale(0), H - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Reward line
  ctx.beginPath();
  slice.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(0), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  ctx.strokeStyle = "#00d4ff";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Average line
  ctx.beginPath();
  aslice.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(0), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2;
  ctx.stroke();

  // X labels
  ctx.fillStyle = "#3d4a66";
  ctx.font = "10px JetBrains Mono, monospace";
  [0, Math.floor(slice.length / 2), slice.length - 1].forEach(i => {
    const ep = SIM.episode - (slice.length - 1 - i);
    ctx.fillText(`Ep${ep}`, xScale(i) - 10, H - 6);
  });
}

// ── UI Updates ─────────────────────────────────────────────────────
function updateUI(readings) {
  // Road + chart
  drawRoad();
  if (SIM.episode % 2 === 0) drawRewardChart();

  // Stats
  document.getElementById("statEpisode").textContent = SIM.episode;
  document.getElementById("statSteps").textContent   = SIM.step;
  document.getElementById("statEpsilon").textContent  = epsilon.toFixed(3);
  document.getElementById("statReward").textContent   = epReward.toFixed(1);

  // KPIs
  document.getElementById("kpiComplete").textContent = SIM.totalCompletions;
  document.getElementById("kpiCollide").textContent  = SIM.totalCollisions;
  document.getElementById("kpiDist").textContent     = SIM.totalDistanceKm.toFixed(1);

  // Road meta
  document.getElementById("carSpeed").textContent    = car.speed.toFixed(1);
  document.getElementById("carLane").textContent     = ["Left", "Centre", "Right"][car.lane] ?? car.lane;
  document.getElementById("carProgress").textContent = (car.pos / ROAD_LEN * 100).toFixed(0) + "%";

  // Last action
  const lastAction = ACTIONS[SIM.actionCounts.map((c, i) => [c, i]).sort((a, b) => b[0] - a[0])[0][1]];
  const recentAction = SIM.step > 0 ? ACTIONS[chooseAction(readings)] : "—";
  document.getElementById("actionDisplay").textContent = recentAction;

  // Action bars
  const totalActs = SIM.actionCounts.reduce((a, b) => a + b, 0) || 1;
  document.getElementById("actionBars").innerHTML = ACTIONS.map((name, i) => {
    const pct = (SIM.actionCounts[i] / totalActs * 100).toFixed(0);
    return `<div class="action-bar">
      <span>${name}</span>
      <div class="action-bar-track"><div class="action-bar-fill" style="width:${pct}%"></div></div>
      <span>${pct}%</span>
    </div>`;
  }).join("");

  // Q-Values
  const idx = stateIndex(readings);
  const q   = qTable[idx];
  const maxQ = Math.max(...q);
  document.getElementById("qvalDisplay").innerHTML = ACTIONS.map((name, i) => {
    const norm = maxQ !== 0 ? Math.max(0, (q[i] / Math.abs(maxQ)) * 100) : 0;
    const cls  = q[i] === maxQ ? "best" : "rest";
    return `<div class="qval-row">
      <span>${name}</span>
      <div class="qval-track"><div class="qval-fill ${cls}" style="width:${norm.toFixed(0)}%"></div></div>
      <span class="qval-num">${q[i].toFixed(2)}</span>
    </div>`;
  }).join("");

  // Sensor Readings
  const angleNames = ["L90°", "L45°", "Fwd", "R45°", "R90°"];
  document.getElementById("sensorReadings").innerHTML = readings.map((v, i) => {
    const pct = (v * 100).toFixed(0);
    const cls = v < 0.3 ? "sensor-ok" : v < 0.7 ? "sensor-warn" : "sensor-crit";
    return `<div class="sensor-row">
      <span>${angleNames[i]}</span>
      <div class="sensor-bar"><div class="sensor-bar-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="sensor-val">${pct}%</span>
    </div>`;
  }).join("");
}

// ── Log ────────────────────────────────────────────────────────────
function addLog(msg, type = "info") {
  SIM.logCount++;
  document.getElementById("logCount").textContent = SIM.logCount;
  const el = document.createElement("div");
  el.className = `log-entry ${type}`;
  const now = new Date();
  el.textContent = `[${now.toLocaleTimeString()}] ${msg}`;
  const body = document.getElementById("logBody");
  body.prepend(el);
  while (body.children.length > 60) body.removeChild(body.lastChild);
}

// ── Clock ──────────────────────────────────────────────────────────
function updateClock() {
  const d = new Date();
  document.getElementById("clockEl").textContent =
    [d.getHours(), d.getMinutes(), d.getSeconds()].map(n => String(n).padStart(2, "0")).join(":");
}
setInterval(updateClock, 1000); updateClock();

// ── Control API ────────────────────────────────────────────────────
function startSim() {
  if (SIM.running) return;
  SIM.running = true;
  SIM.totalEpisodes = +document.getElementById("episodeSlider").value;
  SIM.episode = 0;
  SIM.totalCompletions = 0;
  SIM.totalCollisions  = 0;
  SIM.totalDistanceKm  = 0;
  SIM.rewardHistory    = [];
  SIM.avgHistory       = [];
  SIM.actionCounts     = [0, 0, 0, 0, 0];
  SIM.logCount = 0;
  document.getElementById("logBody").innerHTML = "";
  document.getElementById("logCount").textContent = "0";

  epsilon = 1.0;
  qTable  = Array.from({ length: N_STATES }, () => Array(N_ACTIONS).fill(0));
  resetEpisode();

  const speed = +document.getElementById("speedSlider").value;
  SIM.intervalId = setInterval(tick, Math.max(16, 1000 / speed));

  document.getElementById("btnStart").disabled = true;
  document.getElementById("btnStop").disabled  = false;
  setStatus("running", "● RUNNING");
  addLog(`Simulation started — ${SIM.totalEpisodes} episodes, ${+document.getElementById("obsSlider").value} obstacles`, "info");
}

function stopSim() {
  SIM.running = false;
  clearInterval(SIM.intervalId);
  document.getElementById("btnStart").disabled = false;
  document.getElementById("btnStop").disabled  = true;
  setStatus("stopped", "■ STOPPED");
  addLog("Simulation stopped", "warn");
  drawRewardChart();
}

function resetSim() {
  stopSim();
  SIM.episode = 0;
  SIM.rewardHistory = [];
  SIM.avgHistory    = [];
  SIM.actionCounts  = [0, 0, 0, 0, 0];
  epsilon = 1.0;
  car = { lane: 1, pos: 0, speed: 30, heading: 0, targetLane: 1, laneProgress: 0 };
  obstacles = [];
  drawRoad();
  drawRewardChart();
  setStatus("ready", "● READY");
  addLog("Simulation reset", "info");
  document.getElementById("statEpisode").textContent = "0";
  document.getElementById("statSteps").textContent   = "0";
  document.getElementById("statEpsilon").textContent  = "1.000";
  document.getElementById("statReward").textContent   = "0.0";
  document.getElementById("kpiComplete").textContent  = "0";
  document.getElementById("kpiCollide").textContent   = "0";
  document.getElementById("kpiDist").textContent      = "0";
}

function updateSpeed(v) {
  document.getElementById("speedVal").textContent = v;
  if (SIM.running) {
    clearInterval(SIM.intervalId);
    SIM.intervalId = setInterval(tick, Math.max(16, 1000 / v));
  }
}

function exportLog() {
  const payload = {
    timestamp: new Date().toISOString(),
    episodes: SIM.episode,
    rewards: SIM.rewardHistory,
    averages: SIM.avgHistory,
    completions: SIM.totalCompletions,
    collisions: SIM.totalCollisions,
    distanceKm: SIM.totalDistanceKm
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download: `autonomdrive_${Date.now()}.json`
  });
  a.click();
  addLog("Log exported", "info");
}

function setStatus(cls, text) {
  const el = document.getElementById("statusBadge");
  el.className = `badge ${cls}`;
  el.textContent = text;
}

// ── Initial render ─────────────────────────────────────────────────
spawnObstacles(8);
drawRoad();
drawRewardChart();
addLog("Dashboard ready. Press ▶ Start to begin training.", "info");