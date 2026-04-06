# ============================================================
#  OLAS — Optimised Lab Automation System
#  Flask Scheduler  |  Deploy on Render.com (free tier)
#  Year: 2026
# ============================================================

from flask import Flask, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import requests
import datetime
import schedule
import threading
import time
import os
import logging

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("OLAS")

app = Flask(__name__)

# ── Config — set these as Environment Variables on Render ────
NODE_ID  = os.environ.get("RAINMAKER_NODE_ID",  "aHjSGbCmWDjvmETWDMrupL")
RM_EMAIL = os.environ.get("RAINMAKER_EMAIL",    "sdhumal197@gmail.com")
RM_PASS  = os.environ.get("RAINMAKER_PASSWORD", "Pass@123")

SWITCHES  = ["Switch1", "Switch2", "Switch3", "Switch4"]
FEATURES  = [
    "hour", "minute", "day_of_week", "is_weekend", "time_block",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin",  "dow_cos"
]
API_URL   = "https://api.rainmaker.espressif.com/v1/user/nodes/params"
LOGIN_URL = "https://api.rainmaker.espressif.com/v1/login2"

# ── Token cache ───────────────────────────────────────────────
_token_cache = {
    "access_token" : None,
    "refresh_token": None,
    "fetched_at"   : 0
}

def login_and_get_tokens():
    """
    Full login using email + password via /v1/login2.
    Stores both access_token and refresh_token in cache.
    """
    resp = requests.post(
        LOGIN_URL,
        json={"user_name": RM_EMAIL, "password": RM_PASS},
        timeout=10
    )
    if resp.status_code == 200:
        data = resp.json()
        _token_cache["access_token"]  = data["accesstoken"]
        _token_cache["refresh_token"] = data["refreshtoken"]
        _token_cache["fetched_at"]    = time.time()
        log.info("RainMaker login successful — tokens obtained")
    else:
        raise RuntimeError(
            f"RainMaker login failed: {resp.status_code}  {resp.text[:200]}"
        )

def get_access_token() -> str:
    """
    Returns a valid access token.
    - If token is fresh (< 50 min old) → returns cached token
    - If token is stale → tries silent refresh via Cognito
    - If refresh fails → does full re-login with email + password
    Access tokens last ~60 min. We refresh at 50 min to stay safe.
    """
    now = time.time()
    age = now - _token_cache["fetched_at"]

    if _token_cache["access_token"] is None or age > 3000:
        if _token_cache["refresh_token"]:
            try:
                resp = requests.post(
                    "https://cognito-idp.us-east-1.amazonaws.com/",
                    headers={
                        "Content-Type": "application/x-amz-json-1.1",
                        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
                    },
                    json={
                        "AuthFlow"      : "REFRESH_TOKEN_AUTH",
                        "ClientId"      : "1p3enpe49h9v0lqd7i4s5bub",
                        "AuthParameters": {
                            "REFRESH_TOKEN": _token_cache["refresh_token"]
                        }
                    },
                    timeout=10
                )
                new_token = resp.json()["AuthenticationResult"]["AccessToken"]
                _token_cache["access_token"] = new_token
                _token_cache["fetched_at"]   = now
                log.info("Access token refreshed silently via refresh_token")
            except Exception as e:
                log.warning(f"Silent refresh failed ({e}) — doing full re-login")
                login_and_get_tokens()
        else:
            login_and_get_tokens()

    return _token_cache["access_token"]

# ── Login on startup ──────────────────────────────────────────
if RM_EMAIL != "your@email.com":
    try:
        login_and_get_tokens()
    except Exception as e:
        log.error(f"Startup login failed: {e}")
        log.warning("Scheduler will retry login on first prediction attempt")

# ── Load model ───────────────────────────────────────────────
log.info("Loading lab_model.pkl ...")
with open("lab_model.pkl", "rb") as f:
    models = pickle.load(f)
log.info(f"Model loaded — classifiers: {list(models.keys())}")

# ── In-memory log of last 50 predictions ─────────────────────
prediction_log = []

# ── Feature builder ──────────────────────────────────────────
def build_features(dt: datetime.datetime) -> pd.DataFrame:
    row = {
        "hour"       : dt.hour,
        "minute"     : dt.minute,
        "day_of_week": dt.weekday(),
        "is_weekend" : 1 if dt.weekday() == 6 else 0,
        "time_block" : dt.hour // 6,
        "minute_sin" : np.sin(2 * np.pi * dt.minute / 60),
        "minute_cos" : np.cos(2 * np.pi * dt.minute / 60),
        "hour_sin"   : np.sin(2 * np.pi * dt.hour   / 24),
        "hour_cos"   : np.cos(2 * np.pi * dt.hour   / 24),
        "dow_sin"    : np.sin(2 * np.pi * dt.weekday() / 7),
        "dow_cos"    : np.cos(2 * np.pi * dt.weekday() / 7),
    }
    return pd.DataFrame([row])[FEATURES]

# ── Session detector ─────────────────────────────────────────
def current_session(dt: datetime.datetime) -> str:
    t = dt.time()
    if datetime.time(9, 15) <= t < datetime.time(11, 15):
        return "Session 1  (9:15 – 11:15)"
    if datetime.time(11, 30) <= t < datetime.time(13, 30):
        return "Session 2  (11:30 – 13:30)"
    if datetime.time(14, 15) <= t < datetime.time(16, 15):
        return "Session 3  (14:15 – 16:15)"
    if dt.weekday() == 6:
        return "Sunday — no college"
    return "Outside session hours"

# ── Core: predict → compare → send to RainMaker ──────────────
def predict_and_control(source: str = "scheduler"):
    now     = datetime.datetime.now()
    feats   = build_features(now)
    session = current_session(now)

    # Run all 4 classifiers
    predictions = {}
    for sw in SWITCHES:
        pred = int(models[sw].predict(feats)[0])
        prob = models[sw].predict_proba(feats)[0][pred]
        predictions[sw] = {
            "state"     : bool(pred),
            "confidence": round(float(prob), 3)
        }

    # Build RainMaker payload — "Power" matches write_callback in ESP32 firmware
    payload = {
        "node_id": NODE_ID,
        "payload": {
            sw: {"Power": predictions[sw]["state"]}
            for sw in SWITCHES
        }
    }

    api_status = "not_sent"

    if RM_EMAIL != "your@email.com":
        try:
            # Always fetch a fresh (or cached) token — never hardcoded
            fresh_headers = {
                "Authorization": f"Bearer {get_access_token()}",
                "Content-Type" : "application/json"
            }
            r = requests.put(
                API_URL,
                headers=fresh_headers,
                json=payload,
                timeout=10
            )
            api_status = "ok" if r.status_code == 200 else f"error_{r.status_code}"
            if r.status_code == 200:
                log.info(f"Commands sent OK  |  {session}")
            else:
                log.warning(f"API returned {r.status_code}: {r.text[:120]}")
        except Exception as e:
            api_status = "connection_error"
            log.error(f"API call failed: {e}")
    else:
        api_status = "credentials_not_set"
        log.warning("RAINMAKER_EMAIL not configured — running in preview mode")

    # Store entry in memory log
    entry = {
        "timestamp"  : now.strftime("%Y-%m-%d %H:%M:%S"),
        "session"    : session,
        "source"     : source,
        "predictions": predictions,
        "api_status" : api_status,
    }
    prediction_log.insert(0, entry)
    if len(prediction_log) > 50:
        prediction_log.pop()

    states_str = "  ".join([
        f"{sw}: {'ON ' if predictions[sw]['state'] else 'OFF'}"
        for sw in SWITCHES
    ])
    log.info(f"[{source}]  {states_str}  |  API: {api_status}")
    return entry

# ── Scheduler thread ─────────────────────────────────────────
schedule.every(30).minutes.do(lambda: predict_and_control("scheduler"))

def run_scheduler():
    log.info("Scheduler started — firing every 30 minutes")
    predict_and_control("startup")
    while True:
        schedule.run_pending()
        time.sleep(30)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Dashboard HTML (OLAS Iron Man theme) ─────────────────────
DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OLAS — Control Dashboard 2026</title>
  <meta http-equiv="refresh" content="60">
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    body{background:#03070F;font-family:'Courier New',Courier,monospace;color:#00C8FF;min-height:100vh;padding:16px;}

    @keyframes pulse-ring  {0%,100%{opacity:.7;}50%{opacity:.1;}}
    @keyframes pulse-ring2 {0%,100%{opacity:.3;}50%{opacity:.05;}}
    @keyframes blink       {0%,100%{opacity:1;}50%{opacity:.15;}}
    @keyframes scan        {0%{top:0;opacity:.5;}50%{top:100%;opacity:.15;}100%{top:0;opacity:.5;}}
    @keyframes bar-grow    {from{width:0;}}
    @keyframes fadeIn      {from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
    @keyframes ticker      {0%{transform:translateX(0);}100%{transform:translateX(-50%);}}

    .ring1 {animation:pulse-ring  2.4s ease-in-out infinite;}
    .ring2 {animation:pulse-ring2 2.4s ease-in-out infinite .5s;}
    .blink {animation:blink 1.8s ease-in-out infinite;}
    .bar   {animation:bar-grow 1.2s ease-out forwards;}
    .fadein{animation:fadeIn .7s ease-out both;}

    .shell{max-width:1020px;margin:0 auto;border:1px solid #0D2A44;border-radius:12px;overflow:hidden;background:#060A14;}

    /* ticker */
    .ticker-wrap{background:#030710;border-bottom:1px solid #0A1E30;overflow:hidden;height:22px;display:flex;align-items:center;}
    .ticker-inner{display:flex;white-space:nowrap;animation:ticker 30s linear infinite;}
    .ticker-item{font-size:8px;letter-spacing:2px;color:#0D3A5A;padding:0 28px;}
    .ticker-item span{color:#00C8FF;}

    /* header */
    .hdr{display:flex;align-items:center;gap:20px;padding:22px 28px 18px;border-bottom:1px solid #0D2A44;}
    .hdr-title{font-size:36px;font-weight:bold;letter-spacing:10px;color:#00D4FF;line-height:1;}
    .hdr-sub  {font-size:9px;letter-spacing:3px;color:#4A8FAA;margin-top:5px;}
    .hdr-ver  {font-size:8px;letter-spacing:2px;color:#1A5C7A;margin-top:3px;}
    .hdr-right{margin-left:auto;text-align:right;}
    .online-dot{width:7px;height:7px;border-radius:50%;background:#00FF88;display:inline-block;vertical-align:middle;margin-right:5px;}
    .online-lbl{font-size:9px;letter-spacing:2px;color:#00FF88;vertical-align:middle;}
    .hdr-tags  {font-size:8px;letter-spacing:1px;color:#1A5C7A;margin-top:6px;}
    .hdr-time  {font-size:10px;letter-spacing:1px;color:#2A6080;margin-top:4px;}

    /* section label */
    .sec-lbl{font-size:8px;letter-spacing:3px;color:#4A8FAA;margin-bottom:12px;text-transform:uppercase;}

    /* 2-col */
    .grid2{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #0D2A44;}
    .col{padding:20px 24px;}
    .col-left{border-right:1px solid #0D2A44;position:relative;overflow:hidden;}
    .scan-line{position:absolute;left:0;right:0;height:1px;background:#00C8FF;opacity:.1;animation:scan 5s ease-in-out infinite;}

    /* relay cards */
    .relay-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
    .relay-card{border-radius:6px;padding:10px 12px;}
    .r-on {background:#071830;border:1px solid #0D3A5A;}
    .r-off{background:#0D0B14;border:1px solid #160E24;}
    .r-pin{font-size:8px;letter-spacing:2px;margin-bottom:5px;}
    .r-pin-on {color:#4A8FAA;}
    .r-pin-off{color:#201830;}
    .r-state{font-size:13px;font-weight:bold;letter-spacing:2px;}
    .r-state-on {color:#00D4FF;}
    .r-state-off{color:#201030;}
    .r-track{height:2px;border-radius:1px;margin-top:8px;background:#0D2A44;}
    .r-fill-on {height:2px;border-radius:1px;background:#00C8FF;}
    .r-fill-off{height:2px;border-radius:1px;background:#1A0A28;}

    /* confidence bars */
    .conf-row{margin-bottom:11px;}
    .conf-hdr{display:flex;justify-content:space-between;margin-bottom:4px;}
    .conf-lbl{font-size:9px;letter-spacing:1px;color:#4A8FAA;}
    .conf-pct{font-size:9px;color:#00D4FF;}
    .conf-track{height:4px;background:#0D2A44;border-radius:2px;}
    .conf-fill {height:4px;background:#00C8FF;border-radius:2px;}
    .conf-footer{border-top:1px solid #0D2A44;margin-top:12px;padding-top:10px;display:flex;justify-content:space-between;align-items:center;}
    .conf-model{font-size:8px;letter-spacing:1px;color:#1A5C7A;}
    .badge-active{background:#071830;border:1px solid #00C8FF;border-radius:4px;padding:2px 8px;color:#00D4FF;font-size:8px;letter-spacing:1px;}

    /* status bar */
    .statusbar{display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:1px solid #0D2A44;}
    .sb-cell{padding:12px 18px;text-align:center;}
    .sb-cell:not(:last-child){border-right:1px solid #0D2A44;}
    .sb-lbl{font-size:8px;letter-spacing:2px;color:#4A8FAA;margin-bottom:5px;}
    .sb-val{font-size:10px;font-weight:bold;letter-spacing:1px;}
    .v-cyan {color:#00D4FF;}
    .v-green{color:#00FF88;}
    .v-amber{color:#EF9F27;}
    .v-red  {color:#FF4444;}

    /* prediction pills */
    .pred-section{margin:0 24px;padding:16px 0;border-bottom:1px solid #0D2A44;}
    .pred-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0;}
    .pred-pill{border-radius:4px;padding:6px 14px;font-size:10px;font-weight:bold;letter-spacing:2px;}
    .p-on {background:#071830;border:1px solid #00C8FF;color:#00D4FF;}
    .p-off{background:#0D0B14;border:1px solid #160E24;color:#1A2A3A;}
    .pred-meta{font-size:8px;letter-spacing:1px;color:#1A5C7A;margin-top:10px;}

    /* actions */
    .actions{padding:14px 24px;display:flex;gap:10px;align-items:center;border-bottom:1px solid #0D2A44;flex-wrap:wrap;}
    .btn{background:#071830;border:1px solid #0D3A5A;color:#4A8FAA;font-family:'Courier New',monospace;font-size:9px;letter-spacing:2px;padding:7px 16px;border-radius:4px;cursor:pointer;transition:all .15s;text-decoration:none;display:inline-block;}
    .btn:hover{border-color:#00C8FF;color:#00D4FF;background:#0D2040;}
    .btn-primary{border-color:#00C8FF;color:#00D4FF;}
    .btn-primary:hover{background:#0D2A44;}
    .auto-lbl{font-size:8px;letter-spacing:1px;color:#1A3A4A;margin-left:4px;}

    /* log table */
    .log-wrap{padding:16px 24px 20px;}
    .log-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
    .log-count{font-size:8px;letter-spacing:1px;color:#1A5C7A;}
    table{width:100%;border-collapse:collapse;font-size:10px;}
    thead tr{border-bottom:1px solid #0D3A5A;}
    th{color:#4A8FAA;padding:6px 10px;text-align:left;letter-spacing:1px;font-weight:normal;font-size:8px;}
    tbody tr{border-bottom:1px solid #080E18;transition:background .1s;}
    tbody tr:hover{background:#071222;}
    td{padding:8px 10px;}
    .td-time{color:#2A6080;font-size:9px;}
    .td-sess{color:#1A4050;font-size:9px;}
    .td-src {color:#1A3A4A;font-size:9px;}
    .td-on  {color:#00D4FF;font-weight:bold;}
    .td-off {color:#1A2030;}
    .tbl-badge{border-radius:3px;padding:2px 7px;font-size:8px;letter-spacing:1px;}
    .tb-ok  {background:#071830;border:1px solid #0D3A5A;color:#00C8FF;}
    .tb-warn{background:#1A0E00;border:1px solid #3A2500;color:#EF9F27;}
    .tb-err {background:#1A0000;border:1px solid #3A0000;color:#FF4444;}

    /* footer */
    .footer{padding:10px 24px 14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;}
    .footer-left{font-size:8px;letter-spacing:2px;color:#1A3A4A;}
    .footer-right{display:flex;gap:14px;}
    .footer-tag{font-size:8px;letter-spacing:1px;color:#0D2030;}

    @media(max-width:640px){
      .grid2{grid-template-columns:1fr;}
      .col-left{border-right:none;border-bottom:1px solid #0D2A44;}
      .statusbar{grid-template-columns:1fr;}
      .sb-cell:not(:last-child){border-right:none;border-bottom:1px solid #0D2A44;}
    }
  </style>
</head>
<body>
<div class="shell fadein">

  <!-- TICKER -->
  <div class="ticker-wrap">
    <div class="ticker-inner">
      <span class="ticker-item">OLAS v1.0 &nbsp;·&nbsp; ML-IoT &nbsp;·&nbsp; 2026</span>
      <span class="ticker-item">RANDOM FOREST &nbsp;·&nbsp; <span>200 TREES</span> &nbsp;·&nbsp; <span>~95.5% ACCURACY</span></span>
      <span class="ticker-item">ESP32 &nbsp;·&nbsp; 4-CHANNEL RELAY &nbsp;·&nbsp; GPIO 23 · 19 · 18 · 26</span>
      <span class="ticker-item">SESSIONS: <span>9:15–11:15</span> &nbsp;·&nbsp; <span>11:30–13:30</span> &nbsp;·&nbsp; <span>14:15–16:15</span></span>
      <span class="ticker-item">SCHEDULER FIRES EVERY <span>30 MIN</span> &nbsp;·&nbsp; RENDER.COM FREE TIER</span>
      <span class="ticker-item">OLAS v1.0 &nbsp;·&nbsp; ML-IoT &nbsp;·&nbsp; 2026</span>
      <span class="ticker-item">RANDOM FOREST &nbsp;·&nbsp; <span>200 TREES</span> &nbsp;·&nbsp; <span>~95.5% ACCURACY</span></span>
      <span class="ticker-item">ESP32 &nbsp;·&nbsp; 4-CHANNEL RELAY &nbsp;·&nbsp; GPIO 23 · 19 · 18 · 26</span>
    </div>
  </div>

  <!-- HEADER -->
  <div class="hdr">
    <svg width="76" height="76" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r="56" fill="none" stroke="#00C8FF" stroke-width="0.3" opacity="0.08" class="ring2"/>
      <circle cx="40" cy="40" r="48" fill="none" stroke="#00C8FF" stroke-width="0.6" opacity="0.3"  class="ring1"/>
      <circle cx="40" cy="40" r="34" fill="#060D1C" stroke="#00C8FF" stroke-width="1.2"/>
      <circle cx="40" cy="40" r="26" fill="#071222" stroke="#00A0CC" stroke-width="0.5"/>
      <polygon points="40,18 56,28 56,52 40,62 24,52 24,28" fill="none" stroke="#00C8FF" stroke-width="0.8" opacity="0.55"/>
      <polygon points="40,24 51,30.5 51,49.5 40,56 29,49.5 29,30.5" fill="#00C8FF" opacity="0.05"/>
      <text x="40" y="46" text-anchor="middle" dominant-baseline="central"
            fill="#00D4FF" font-size="15" font-weight="bold"
            font-family="Courier New" letter-spacing="1">O</text>
    </svg>
    <div>
      <div class="hdr-title">OLAS</div>
      <div class="hdr-sub">OPTIMISED LAB AUTOMATION SYSTEM</div>
      <div class="hdr-ver">ML-IoT Integration &nbsp;·&nbsp; 2026 &nbsp;·&nbsp; v1.0</div>
    </div>
    <div class="hdr-right">
      {% if logs %}
      <div>
        <span class="online-dot blink"></span>
        <span class="online-lbl">SYSTEM ONLINE</span>
      </div>
      {% else %}
      <div>
        <span class="online-dot blink" style="background:#EF9F27;"></span>
        <span class="online-lbl" style="color:#EF9F27;">AWAITING DATA</span>
      </div>
      {% endif %}
      <div class="hdr-tags">ESP32 &nbsp;·&nbsp; RAINMAKER &nbsp;·&nbsp; RENDER.COM</div>
    </div>
  </div>

  {% if logs %}
  {% set last = logs[0] %}

  <!-- RELAY + ML GRID -->
  <div class="grid2">

    <!-- Left: relay status -->
    <div class="col col-left">
      <div class="scan-line"></div>
      <div class="sec-lbl">Relay status</div>
      <div class="relay-grid">
        {% set sw_gpio = [
          ('Switch1','GPIO 23'),('Switch2','GPIO 19'),
          ('Switch3','GPIO 18'),('Switch4','GPIO 26')
        ] %}
        {% for sw, gpio in sw_gpio %}
        {% set on = last.predictions[sw].state %}
        {% set w  = (last.predictions[sw].confidence * 100)|int %}
        <div class="relay-card {{ 'r-on' if on else 'r-off' }}">
          <div class="r-pin {{ 'r-pin-on' if on else 'r-pin-off' }}">
            {{ sw.upper() }} &nbsp;·&nbsp; {{ gpio }}
          </div>
          <div class="r-state {{ 'r-state-on' if on else 'r-state-off' }}">
            {{ 'ACTIVE' if on else 'STANDBY' }}
          </div>
          <div class="r-track">
            <div class="bar {{ 'r-fill-on' if on else 'r-fill-off' }}" style="width:{{ w }}%"></div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

    <!-- Right: ML confidence -->
    <div class="col">
      <div class="sec-lbl">ML Prediction Engine</div>
      {% for sw in ['Switch1','Switch2','Switch3','Switch4'] %}
      {% set pct = (last.predictions[sw].confidence * 100)|round(1) %}
      <div class="conf-row">
        <div class="conf-hdr">
          <span class="conf-lbl">{{ sw }} confidence</span>
          <span class="conf-pct">{{ pct }}%</span>
        </div>
        <div class="conf-track">
          <div class="bar conf-fill" style="width:{{ pct }}%"></div>
        </div>
      </div>
      {% endfor %}
      <div class="conf-footer">
        <span class="conf-model">MODEL: RANDOM FOREST &nbsp;·&nbsp; 200 TREES</span>
        <span class="badge-active">ACTIVE</span>
      </div>
    </div>
  </div>

  <!-- STATUS BAR -->
  <div class="statusbar">
    <div class="sb-cell">
      <div class="sb-lbl">Session</div>
      <div class="sb-val v-cyan">{{ last.session[:26] }}</div>
    </div>
    <div class="sb-cell">
      <div class="sb-lbl">Last run</div>
      <div class="sb-val v-green">{{ last.timestamp[11:] }} &nbsp;·&nbsp; {{ last.source }}</div>
    </div>
    <div class="sb-cell">
      <div class="sb-lbl">API Status</div>
      {% if last.api_status == 'ok' %}
        <div class="sb-val v-green">RAINMAKER &nbsp;·&nbsp; 200 OK</div>
      {% elif 'not_set' in last.api_status or 'credentials' in last.api_status %}
        <div class="sb-val v-amber">CREDENTIALS NOT SET</div>
      {% else %}
        <div class="sb-val v-red">{{ last.api_status|upper }}</div>
      {% endif %}
    </div>
  </div>

  <!-- PREDICTION PILLS -->
  <div class="pred-section">
    <div class="sec-lbl">Last prediction</div>
    <div class="pred-row">
      {% for sw in ['Switch1','Switch2','Switch3','Switch4'] %}
      {% set on  = last.predictions[sw].state %}
      {% set pct = (last.predictions[sw].confidence * 100)|int %}
      <div class="pred-pill {{ 'p-on' if on else 'p-off' }}">
        {{ sw[-1] }} &nbsp; {{ 'ON' if on else 'OFF' }} &nbsp; {{ pct }}%
      </div>
      {% endfor %}
    </div>
    <div class="pred-meta">
      {{ last.timestamp }} &nbsp;·&nbsp; {{ last.session }} &nbsp;·&nbsp; source: {{ last.source }}
    </div>
  </div>

  {% endif %}

  <!-- ACTIONS -->
  <div class="actions">
    <a href="/trigger" class="btn btn-primary">[ FORCE PREDICTION ]</a>
    <a href="/status"  class="btn">[ JSON STATUS ]</a>
    <a href="/predict_time/9/30/0"   class="btn">[ TEST 9:30 MON ]</a>
    <a href="/predict_time/14/15/2"  class="btn">[ TEST 14:15 WED ]</a>
    <span class="auto-lbl">AUTO-REFRESH &nbsp;60s</span>
  </div>

  <!-- LOG TABLE -->
  <div class="log-wrap">
    <div class="log-hdr">
      <div class="sec-lbl" style="margin:0">Prediction log</div>
      <div class="log-count">{{ logs|length }} entries</div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Session</th>
          <th>S1</th><th>S2</th><th>S3</th><th>S4</th>
          <th>Source</th>
          <th>API</th>
        </tr>
      </thead>
      <tbody>
        {% for entry in logs %}
        <tr>
          <td class="td-time">{{ entry.timestamp }}</td>
          <td class="td-sess">{{ entry.session[:22] }}</td>
          {% for sw in ['Switch1','Switch2','Switch3','Switch4'] %}
          <td class="{{ 'td-on' if entry.predictions[sw].state else 'td-off' }}">
            {{ 'ON' if entry.predictions[sw].state else 'OFF' }}
          </td>
          {% endfor %}
          <td class="td-src">{{ entry.source }}</td>
          <td>
            {% if entry.api_status == 'ok' %}
              <span class="tbl-badge tb-ok">200 OK</span>
            {% elif 'credentials' in entry.api_status or 'not_set' in entry.api_status %}
              <span class="tbl-badge tb-warn">NO CREDS</span>
            {% elif 'error' in entry.api_status %}
              <span class="tbl-badge tb-err">{{ entry.api_status|upper }}</span>
            {% else %}
              <span class="tbl-badge tb-warn">{{ entry.api_status|upper }}</span>
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div class="footer-left">OLAS &nbsp;·&nbsp; ML-IoT Mini Project &nbsp;·&nbsp; 2026</div>
    <div class="footer-right">
      <span class="footer-tag">ESP32</span>
      <span class="footer-tag">RAINMAKER</span>
      <span class="footer-tag">GOOGLE HOME</span>
      <span class="footer-tag">RENDER.COM</span>
      <span class="footer-tag">RANDOM FOREST</span>
    </div>
  </div>

</div>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD, logs=prediction_log)

@app.route("/status")
def status():
    last = prediction_log[0] if prediction_log else None
    return jsonify({
        "project"     : "OLAS 2026",
        "status"      : "running",
        "scheduler"   : "active — every 30 min",
        "node_id"     : NODE_ID[:8] + "..." if len(NODE_ID) > 8 else NODE_ID,
        "last_run"    : last["timestamp"] if last else None,
        "last_session": last["session"]   if last else None,
        "token_age_s" : round(time.time() - _token_cache["fetched_at"]) if _token_cache["fetched_at"] else None,
    })

@app.route("/trigger", methods=["GET", "POST"])
def trigger():
    entry = predict_and_control("manual_trigger")
    return jsonify({
        "message"    : "Prediction triggered",
        "timestamp"  : entry["timestamp"],
        "session"    : entry["session"],
        "predictions": entry["predictions"],
        "api_status" : entry["api_status"],
    })

@app.route("/predict_time/<int:hour>/<int:minute>/<int:dow>")
def predict_time(hour, minute, dow):
    """
    Test prediction for any time without sending to ESP32.
    /predict_time/9/30/0   → Monday 9:30   (Session 1)
    /predict_time/12/0/1   → Tuesday 12:00 (Session 2)
    /predict_time/14/15/2  → Wednesday 14:15 (Session 3)
    /predict_time/17/0/0   → Monday 17:00  (Outside)
    dow: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
    """
    dt    = datetime.datetime.now().replace(hour=hour, minute=minute)
    feats = build_features(dt)
    preds = {}
    for sw in SWITCHES:
        state = int(models[sw].predict(feats)[0])
        prob  = models[sw].predict_proba(feats)[0][state]
        preds[sw] = {"state": bool(state), "confidence": round(float(prob), 3)}
    return jsonify({
        "query"      : f"{hour:02d}:{minute:02d}  day_of_week={dow}",
        "session"    : current_session(dt),
        "predictions": preds
    })

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
