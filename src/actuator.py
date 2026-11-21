"""Simulated actuator (dimmable LED driver) for the smart lighting system.

The classifier only *decides* a brightness label. This module closes the
sensor -> AI -> actuator loop by turning that decision into what a real fixture
would do:

* an MQTT-style command message (topic + JSON payload, no real broker needed),
* an 8-bit PWM duty cycle for the LED driver,
* a soft fade (limited ramp rate) instead of an instant jump, and
* exact power / energy accounting while the light is fading.

Everything is pure Python and deterministic. ``DimmableBulb`` models one bulb,
``simulate_actuator`` drives it with a stream of AI decisions, and the HTML
helpers render a virtual bulb for the Streamlit dashboard.

Usage:
    python actuator.py              # simulate one day, write log/events/figure
    python actuator.py --seed 7     # a different simulated day
"""

import argparse
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    ACTUATOR_EVENTS_CSV,
    ACTUATOR_LOG_CSV,
    ACTUATOR_RESPONSE_PNG,
    BRIGHTNESS_LEVELS,
    BULB_FULL_WATT,
    FADE_RATE_PCT_PER_S,
    MQTT_TOPIC,
    PWM_BITS,
    PWM_FREQUENCY_HZ,
    SIM_SEED,
    STEP_MINUTES,
    create_directories,
)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def level_to_pct(label):
    """Convert a brightness label ('OFF', '20%', ...) to a percentage."""
    label = str(label)
    if label not in BRIGHTNESS_LEVELS:
        raise ValueError(f"Unknown brightness label {label!r}; expected {BRIGHTNESS_LEVELS}.")
    return 0.0 if label == "OFF" else float(label.rstrip("%"))


def pwm_duty(level_pct, bits=PWM_BITS):
    """Convert a brightness percentage into an integer PWM duty value."""
    maximum = 2 ** bits - 1
    return int(round(min(max(level_pct, 0.0), 100.0) / 100.0 * maximum))


def fade_time_s(from_pct, to_pct, rate=FADE_RATE_PCT_PER_S):
    """Seconds needed to ramp between two brightness levels at ``rate`` %/s."""
    if rate <= 0:
        raise ValueError("fade rate must be positive.")
    return abs(to_pct - from_pct) / rate


def command_message(label, from_pct=0.0, rate=FADE_RATE_PCT_PER_S, bits=PWM_BITS):
    """Build the simulated MQTT command (topic + JSON payload) for a decision."""
    target = level_to_pct(label)
    payload = {
        "state": "OFF" if target == 0 else "ON",
        "brightness": int(target),
        "pwm": pwm_duty(target, bits),
        "fade_s": round(fade_time_s(from_pct, target, rate), 2),
    }
    return {"topic": MQTT_TOPIC, "payload": json.dumps(payload, separators=(",", ":"))}


def clock_label(step_index, step_minutes=STEP_MINUTES):
    """Human-readable simulation clock, e.g. 'Day 1 08:35'."""
    total = step_index * step_minutes
    day, minute = divmod(total, 24 * 60)
    return f"Day {day + 1} {minute // 60:02d}:{minute % 60:02d}"


# ---------------------------------------------------------------------------
# The actuator
# ---------------------------------------------------------------------------
class DimmableBulb:
    """A dimmable LED bulb behind a PWM driver with a limited fade rate.

    Power is linear in brightness (``rated_watt * level / 100``), which matches
    ``WATT_MAP``. While fading, the level changes linearly, so the energy over
    any interval is computed exactly (trapezoid rule) rather than approximated.
    """

    def __init__(self, rated_watt=BULB_FULL_WATT, fade_rate=FADE_RATE_PCT_PER_S,
                 pwm_bits=PWM_BITS, initial="OFF"):
        if fade_rate <= 0:
            raise ValueError("fade_rate must be positive.")
        if rated_watt <= 0:
            raise ValueError("rated_watt must be positive.")
        self.rated_watt = float(rated_watt)
        self.fade_rate = float(fade_rate)
        self.pwm_bits = int(pwm_bits)
        self.level_pct = level_to_pct(initial)
        self.target_label = str(initial)
        self.target_pct = self.level_pct
        self.energy_wh = 0.0
        self.elapsed_s = 0.0
        self.level_changes = 0   # commands that actually changed the target level
        self.turn_on_events = 0  # OFF -> lit transitions (driver / lamp wear metric)
        self.events = []

    # -- state ------------------------------------------------------------
    @property
    def power_w(self):
        """Instantaneous electrical power in watts."""
        return self.rated_watt * self.level_pct / 100.0

    @property
    def pwm_duty(self):
        """Current PWM duty value (0 .. 2**bits - 1)."""
        return pwm_duty(self.level_pct, self.pwm_bits)

    @property
    def state(self):
        """'OFF', 'FADING' (level still moving to the target) or 'ON'."""
        if abs(self.level_pct - self.target_pct) > 1e-9:
            return "FADING"
        return "OFF" if self.level_pct <= 1e-9 else "ON"

    # -- control ----------------------------------------------------------
    def command(self, label, time_label=None):
        """Send a brightness command. Returns the event dict, or None if unchanged.

        Re-sending the current target is ignored, exactly as a real controller
        would suppress redundant messages to save bus traffic.
        """
        label = str(label)
        level_to_pct(label)  # validates the label
        if label == self.target_label:
            return None

        previous = self.target_label
        message = command_message(label, self.level_pct, self.fade_rate, self.pwm_bits)
        event = {
            "time": time_label if time_label is not None else f"t={self.elapsed_s:.1f}s",
            "from": previous,
            "to": label,
            "topic": message["topic"],
            "payload": message["payload"],
        }
        self.level_changes += 1
        if previous == "OFF":
            self.turn_on_events += 1
        self.target_label = label
        self.target_pct = level_to_pct(label)
        self.events.append(event)
        return event

    def advance(self, seconds):
        """Run the bulb for ``seconds`` and return its average power (W).

        The level ramps toward the target at ``fade_rate`` %/s, then holds.
        Energy is integrated exactly over the ramp and the hold.
        """
        if seconds < 0:
            raise ValueError("seconds must be >= 0.")
        if seconds == 0:
            return self.power_w

        gap = self.target_pct - self.level_pct
        ramp_time = abs(gap) / self.fade_rate
        if ramp_time > seconds:
            # still fading when the interval ends
            end = self.level_pct + math.copysign(self.fade_rate * seconds, gap)
            average_pct = (self.level_pct + end) / 2.0
            self.level_pct = end
        else:
            ramp_avg = (self.level_pct + self.target_pct) / 2.0
            average_pct = (ramp_avg * ramp_time + self.target_pct * (seconds - ramp_time)) / seconds
            self.level_pct = self.target_pct

        self.level_pct = min(max(self.level_pct, 0.0), 100.0)
        power = self.rated_watt * average_pct / 100.0
        self.energy_wh += power * seconds / 3600.0
        self.elapsed_s += seconds
        return power


# ---------------------------------------------------------------------------
# Driving the actuator with AI decisions
# ---------------------------------------------------------------------------
def simulate_actuator(decisions, step_minutes=STEP_MINUTES, bulb=None):
    """Drive a bulb with a stream of AI decisions, one decision per step.

    ``decisions`` needs a ``predicted_brightness`` column (any extra columns,
    such as occupancy or daylight, are carried into the log).

    Returns ``(log, events, summary)``:
    * ``log``     - one row per step: commanded level, actual level at the end
                    of the step, PWM duty, average power, cumulative energy.
    * ``events``  - one row per command actually sent (MQTT-style messages).
    * ``summary`` - totals and a cross-check against the instant-switch model.
    """
    if "predicted_brightness" not in decisions.columns:
        raise ValueError("decisions must contain a 'predicted_brightness' column.")
    bulb = bulb if bulb is not None else DimmableBulb()
    step_seconds = step_minutes * 60.0

    rows = []
    for index, label in enumerate(decisions["predicted_brightness"].tolist()):
        event = bulb.command(label, time_label=clock_label(index, step_minutes))
        average_power = bulb.advance(step_seconds)
        rows.append(
            {
                "commanded": label,
                "actual_pct": round(bulb.level_pct, 4),
                "pwm_duty": bulb.pwm_duty,
                "avg_power_w": round(average_power, 4),
                "energy_wh_cum": round(bulb.energy_wh, 6),
                "state": bulb.state,
                "command_sent": event is not None,
            }
        )

    log = pd.concat(
        [decisions.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    )
    events = pd.DataFrame(
        bulb.events, columns=["time", "from", "to", "topic", "payload"]
    )

    instant_wh = (
        log["commanded"].map(level_to_pct).sum() / 100.0
        * bulb.rated_watt * step_minutes / 60.0
    )
    total_days = len(log) * step_minutes / (24 * 60)
    summary = {
        "steps": len(log),
        "days": total_days,
        "energy_wh": float(bulb.energy_wh),
        "energy_kwh": float(bulb.energy_wh / 1000.0),
        "instant_switch_wh": float(instant_wh),
        "fade_effect_pct": float(
            (bulb.energy_wh - instant_wh) / instant_wh * 100.0
        ) if instant_wh else 0.0,
        "commands_sent": int(len(events)),
        "level_changes": int(bulb.level_changes),
        "turn_on_events": int(bulb.turn_on_events),
        "commands_per_day": float(len(events) / total_days) if total_days else 0.0,
        "fade_rate_pct_per_s": bulb.fade_rate,
        "pwm_bits": bulb.pwm_bits,
        "pwm_frequency_hz": PWM_FREQUENCY_HZ,
        "full_range_fade_s": 100.0 / bulb.fade_rate,
    }
    return log, events, summary


def fade_closeup(commands=((1.0, "90%"), (5.0, "50%"), (8.0, "OFF")),
                 duration_s=12.0, dt=0.05):
    """Fine-grained (50 ms) trace of the soft fade for a few commands.

    Makes the seconds-scale ramp visible, which is invisible at the 5-minute
    resolution of the day simulation. Returns a DataFrame (t_s, level_pct, pwm).
    """
    bulb = DimmableBulb()
    pending = sorted(commands)
    rows = []
    ticks = int(round(duration_s / dt))
    for tick in range(ticks + 1):
        now = tick * dt
        while pending and now + 1e-9 >= pending[0][0]:
            bulb.command(pending.pop(0)[1], time_label=f"{now:.2f}s")
        rows.append({"t_s": round(now, 4), "level_pct": bulb.level_pct, "pwm": bulb.pwm_duty})
        bulb.advance(dt)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_actuator_figure(log):
    """Four-panel figure: command vs bulb level, PWM, power/energy, fade close-up."""
    day = log[log["day"] == log["day"].min()] if "day" in log.columns else log
    time_h = day["time_h"]
    occupied = day["occupancy"].to_numpy() == 1 if "occupancy" in day.columns else None

    fig, axes = plt.subplots(4, 1, figsize=(11, 12.5))
    ax1, ax2, ax3, ax4 = axes

    def shade(ax):
        if occupied is not None:
            ax.fill_between(time_h, 0, 1, where=occupied, step="post", alpha=0.18,
                            color="#e9c46a", transform=ax.get_xaxis_transform(),
                            label="Room occupied")

    shade(ax1)
    ax1.step(time_h, day["commanded"].map(level_to_pct), where="post", color="#264653",
             linewidth=2, label="AI command")
    ax1.step(time_h, day["actual_pct"], where="post", color="#e76f51", linewidth=1.2,
             linestyle="--", label="Bulb level (end of step)")
    ax1.set_ylabel("Brightness (%)")
    ax1.set_ylim(-4, 104)
    ax1.set_title("Actuator Response Over One Simulated Day")
    ax1.legend(loc="upper right", ncol=3, fontsize=8)

    shade(ax2)
    ax2.fill_between(time_h, day["pwm_duty"], step="post", color="#2a9d8f", alpha=0.65)
    ax2.set_ylabel(f"PWM duty (0-{2 ** PWM_BITS - 1})")
    ax2.set_ylim(0, 2 ** PWM_BITS)

    shade(ax3)
    ax3.step(time_h, day["avg_power_w"], where="post", color="#6a4c93", linewidth=1.6,
             label="Power (W)")
    ax3.set_ylabel("Power (W)")
    ax3.set_xlabel("Time of day (hour)")
    twin = ax3.twinx()
    day_start_wh = day["energy_wh_cum"].iloc[0] - day["avg_power_w"].iloc[0] * STEP_MINUTES / 60.0
    twin.plot(time_h, day["energy_wh_cum"] - day_start_wh,
              color="#f4a261", linewidth=2, label="Cumulative energy (Wh)")
    twin.set_ylabel("Cumulative energy (Wh)")
    twin.grid(False)
    handles = ax3.get_legend_handles_labels()
    handles_twin = twin.get_legend_handles_labels()
    ax3.legend(handles[0] + handles_twin[0], handles[1] + handles_twin[1],
               loc="center right", fontsize=8)
    for axis in (ax1, ax2, ax3):
        axis.set_xlim(0, 24)
        axis.set_xticks(range(0, 25, 2))

    closeup = fade_closeup()
    ax4.plot(closeup["t_s"], closeup["level_pct"], color="#e76f51", linewidth=2.2,
             label="Bulb level")
    for time_s, label in ((1.0, "cmd 90%"), (5.0, "cmd 50%"), (8.0, "cmd OFF")):
        ax4.axvline(time_s, color="#264653", linestyle=":", linewidth=1)
        ax4.text(time_s + 0.08, 96, label, fontsize=8, color="#264653")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Brightness (%)")
    ax4.set_ylim(-4, 104)
    ax4.set_title(f"Soft-fade close-up: {FADE_RATE_PCT_PER_S:.0f} %/s ramp "
                  f"(full range in {100 / FADE_RATE_PCT_PER_S:.1f} s)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML rendering of the virtual bulb (self-contained: SVG + CSS + JS)
# ---------------------------------------------------------------------------
_PAGE_STYLE = """
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#e8ecf3}
.card{background:linear-gradient(160deg,#141a26,#0d111a);border:1px solid #2b3345;
border-radius:14px;padding:14px 16px}
.row{display:flex;flex-wrap:wrap;gap:18px;align-items:center}
.stage{flex:0 0 215px;text-align:center}
.grid{flex:1 1 300px;display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
.cell{background:#1b2232;border-radius:9px;padding:7px 10px}
.cell .k{font-size:11px;color:#8d98b0;text-transform:uppercase;letter-spacing:.04em}
.cell .v{font-size:19px;font-weight:600;margin-top:2px}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:13px;font-weight:600}
.on{background:#1f6f43;color:#d4ffe6}.off{background:#3a4254;color:#c9d1e3}
.bar{height:9px;border-radius:5px;background:#26304a;overflow:hidden;margin-top:7px}
.bar>div{height:100%;background:linear-gradient(90deg,#4aa3ff,#ffe082)}
.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:12px}
button,select{background:#232c40;color:#e8ecf3;border:1px solid #3a4562;border-radius:8px;
padding:6px 14px;font-size:14px;cursor:pointer}
button:hover{background:#2c3752}
input[type=range]{flex:1 1 220px;accent-color:#ffd54f}
canvas{width:100%;height:92px;display:block;margin-top:10px;background:#101521;border-radius:9px}
.badge{font-size:12px;color:#ffd54f;min-height:16px;margin-top:6px}
.cap{font-size:13px;color:#aab4ca;margin-top:6px}
#lit,#halo{transition:opacity var(--fade,0.35s) linear}
"""


def _bulb_svg():
    """Inline SVG of a bulb; opacity of #lit / #halo encodes brightness."""
    return (
        '<svg viewBox="-40 -34 280 316" width="215" height="243" role="img" aria-label="virtual bulb">'
        '<defs>'
        '<radialGradient id="g" cx="50%" cy="42%" r="58%">'
        '<stop offset="0" stop-color="#fffbe6"/><stop offset=".55" stop-color="#ffe082"/>'
        '<stop offset="1" stop-color="#ffb300"/></radialGradient>'
        '<filter id="blur" x="-60%" y="-60%" width="220%" height="220%">'
        '<feGaussianBlur stdDeviation="15"/></filter></defs>'
        '<circle id="halo" cx="100" cy="92" r="92" fill="#ffd54f" filter="url(#blur)" opacity="0"/>'
        '<path d="M70 168 C70 148 40 134 40 94 A60 60 0 0 1 160 94 C160 134 130 148 130 168 Z" '
        'fill="#262d3d" stroke="#7b8498" stroke-width="3"/>'
        '<path id="lit" d="M70 168 C70 148 40 134 40 94 A60 60 0 0 1 160 94 C160 134 130 148 130 168 Z" '
        'fill="url(#g)" opacity="0"/>'
        '<path d="M86 150 L86 112 L100 128 L114 112 L114 150" fill="none" stroke="#a9b2c6" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity=".75"/>'
        '<rect x="72" y="168" width="56" height="14" rx="3" fill="#8b93a7"/>'
        '<rect x="74" y="184" width="52" height="12" rx="3" fill="#767e92"/>'
        '<rect x="80" y="198" width="40" height="11" rx="5" fill="#5f677b"/>'
        '</svg>'
    )


def _page(body, script):
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<style>{_PAGE_STYLE}</style></head><body>{body}<script>{script}</script></body></html>"
    )


def bulb_html(level_pct, label, from_pct=None, power_w=None, rate=FADE_RATE_PCT_PER_S,
              bits=PWM_BITS):
    """Standalone HTML for one virtual bulb at ``level_pct``.

    If ``from_pct`` is given the bulb starts at that level and fades to
    ``level_pct`` over the physically correct time (``|delta| / rate`` seconds).
    """
    level_pct = float(level_pct)
    start = level_pct if from_pct is None else float(from_pct)
    fade = fade_time_s(start, level_pct, rate)
    watts = BULB_FULL_WATT * level_pct / 100.0 if power_w is None else power_w
    body = (
        '<div class="card"><div class="row"><div class="stage">'
        + _bulb_svg()
        + '</div><div class="grid">'
        f'<div class="cell"><div class="k">Command</div><div class="v" id="cmd">{label}</div></div>'
        f'<div class="cell"><div class="k">Brightness</div><div class="v" id="pct">{level_pct:.0f}%</div></div>'
        f'<div class="cell"><div class="k">PWM duty</div><div class="v" id="pwm">'
        f'{pwm_duty(level_pct, bits)} / {2 ** bits - 1}</div></div>'
        f'<div class="cell"><div class="k">Power</div><div class="v" id="pw">{watts:.0f} W</div></div>'
        f'<div class="cell"><div class="k">Fade time</div><div class="v" id="fd">{fade:.1f} s</div></div>'
        f'<div class="cell"><div class="k">State</div><div class="v">'
        f'<span class="pill {"on" if level_pct > 0 else "off"}" id="st">'
        f'{"ON" if level_pct > 0 else "OFF"}</span></div></div>'
        "</div></div></div>"
    )
    script = (
        "(function(){function vis(p){return p<=0?0:Math.pow(p/100,0.6);}"
        "var lit=document.getElementById('lit'),halo=document.getElementById('halo');"
        f"var from={start!r},to={level_pct!r},fade={fade!r};"
        "lit.style.setProperty('--fade',fade+'s');halo.style.setProperty('--fade',fade+'s');"
        "lit.style.opacity=vis(from);halo.style.opacity=vis(from)*0.9;"
        "requestAnimationFrame(function(){requestAnimationFrame(function(){"
        "lit.style.opacity=vis(to);halo.style.opacity=vis(to)*0.9;});});})();"
    )
    return _page(body, script)


_PLAYER_BODY = """
<div class="card">
<div class="row"><div class="stage">__SVG__</div>
<div class="grid">
<div class="cell"><div class="k">Time</div><div class="v" id="t">--:--</div></div>
<div class="cell"><div class="k">Room</div><div class="v"><span class="pill off" id="occ">Empty</span></div></div>
<div class="cell"><div class="k">Ambient</div><div class="v" id="amb">-</div>
<div class="bar"><div id="dl" style="width:0%"></div></div></div>
<div class="cell"><div class="k">AI command</div><div class="v" id="cmd">OFF</div></div>
<div class="cell"><div class="k">Bulb level</div><div class="v" id="pct">0%</div></div>
<div class="cell"><div class="k">PWM duty</div><div class="v" id="pwm">0 / 255</div></div>
<div class="cell"><div class="k">Power</div><div class="v" id="pw">0 W</div></div>
<div class="cell"><div class="k">Energy today</div><div class="v" id="wh">0.0 Wh</div></div>
</div></div>
<div class="badge" id="badge"></div>
<canvas id="cv" height="92"></canvas>
<div class="ctl">
<button id="play" type="button">Play</button>
<input id="scrub" type="range" min="0" max="__MAX__" value="0" step="1">
<select id="speed"><option value="6">Slow</option><option value="12" selected>Normal</option>
<option value="24">Fast</option></select>
</div>
<div class="cap">Drag the slider or press Play. Yellow shading = room occupied, orange line =
daylight, white line = bulb level, vertical marker = current time.</div>
</div>
"""

_PLAYER_SCRIPT = """
(function(){
var D=__DATA__, N=D.length, idx=__START__, timer=null;
function el(i){return document.getElementById(i);}
function vis(p){return p<=0?0:Math.pow(p/100,0.6);}
function draw(i){
  var cv=el('cv'), ctx=cv.getContext&&cv.getContext('2d'); if(!ctx){return;}
  var dpr=window.devicePixelRatio||1, W=cv.clientWidth||600, H=92;
  cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H); var x=function(k){return 6+(W-12)*k/Math.max(N-1,1);};
  var y=function(p){return H-14-(H-26)*p/100;}, k;
  ctx.fillStyle='rgba(233,196,106,0.13)';
  for(k=0;k<N;k++){ if(D[k].occ){ctx.fillRect(x(k),4,Math.max(x(k+1)-x(k),1),H-18);} }
  ctx.strokeStyle='rgba(244,162,97,0.9)'; ctx.lineWidth=1.4; ctx.beginPath();
  for(k=0;k<N;k++){var yy=y(D[k].dl*100); if(k===0){ctx.moveTo(x(k),yy);}else{ctx.lineTo(x(k),yy);}} ctx.stroke();
  ctx.strokeStyle='#ffffff'; ctx.lineWidth=2.6; ctx.beginPath();
  for(k=0;k<N;k++){var py=y(D[k].pct); if(k===0){ctx.moveTo(x(k),py);}else{ctx.lineTo(x(k),py);}} ctx.stroke();
  ctx.fillStyle='#8d98b0'; ctx.font='10px sans-serif';
  [0,6,12,18,24].forEach(function(h){var kk=Math.min(N-1,Math.round(h/24*N)); ctx.fillText(h+'h',x(kk)-6,H-2);});
  ctx.strokeStyle='#4aa3ff'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(x(i),2); ctx.lineTo(x(i),H-14); ctx.stroke();
}
function render(i){
  var r=D[i], v=vis(r.pct);
  el('lit').style.opacity=v; el('halo').style.opacity=v*0.9;
  el('t').textContent=r.t; el('amb').textContent=r.amb;
  el('occ').textContent=r.occ?'Occupied':'Empty'; el('occ').className='pill '+(r.occ?'on':'off');
  el('dl').style.width=(r.dl*100).toFixed(0)+'%';
  el('cmd').textContent=r.cmd; el('pct').textContent=r.pct.toFixed(0)+'%';
  el('pwm').textContent=r.pwm+' / __PWMMAX__'; el('pw').textContent=r.p.toFixed(0)+' W';
  el('wh').textContent=r.wh.toFixed(1)+' Wh';
  el('badge').textContent=r.ev?('MQTT command published: '+r.cmd):'';
  el('scrub').value=i; draw(i);
}
function pause(){ if(timer){clearInterval(timer);timer=null;} el('play').textContent='Play'; }
function tick(){ if(idx>=N-1){pause();return;} idx++; render(idx); }
function play(){
  if(timer){pause();return;}
  if(idx>=N-1){idx=0;render(idx);}
  el('play').textContent='Pause'; timer=setInterval(tick,1000/parseFloat(el('speed').value));
}
el('play').addEventListener('click',play);
el('speed').addEventListener('change',function(){ if(timer){clearInterval(timer);timer=setInterval(tick,1000/parseFloat(el('speed').value));} });
el('scrub').addEventListener('input',function(){ idx=parseInt(this.value,10); render(idx); });
window.addEventListener('resize',function(){draw(idx);});
window.__lightingPlayer={goto:function(i){idx=Math.max(0,Math.min(N-1,i));render(idx);},
  index:function(){return idx;},playing:function(){return timer!==null;},frames:N};
render(idx);
})();
"""


def day_player_html(log, day=None, start_step=None):
    """Self-contained interactive HTML player that replays one simulated day.

    Everything (data, bulb, timeline, controls) runs client-side in JavaScript,
    so playback is smooth and does not trigger Streamlit reruns.
    """
    frame = log if "day" not in log.columns else log[log["day"] == (
        log["day"].min() if day is None else day)]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("log has no rows to replay.")
    base = float(frame["energy_wh_cum"].iloc[0] - frame["avg_power_w"].iloc[0]
                 * STEP_MINUTES / 60.0)

    records = []
    for index, row in frame.iterrows():
        minute = index * STEP_MINUTES
        records.append(
            {
                "t": f"{minute // 60 % 24:02d}:{minute % 60:02d}",
                "amb": str(row["ambient"]) if "ambient" in frame.columns else "-",
                "occ": int(row["occupancy"]) if "occupancy" in frame.columns else 0,
                "dl": float(row["daylight_index"]) if "daylight_index" in frame.columns else 0.0,
                "cmd": str(row["commanded"]),
                "pct": float(row["actual_pct"]),
                "pwm": int(row["pwm_duty"]),
                "p": float(row["avg_power_w"]),
                "wh": float(row["energy_wh_cum"]) - base,
                "ev": 1 if bool(row["command_sent"]) else 0,
            }
        )

    if start_step is None:
        lit = [i for i, rec in enumerate(records) if rec["pct"] > 0]
        start_step = lit[0] if lit else 0
    start_step = int(min(max(start_step, 0), len(records) - 1))

    data = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    body = _PLAYER_BODY.replace("__SVG__", _bulb_svg()).replace("__MAX__", str(len(records) - 1))
    script = (
        _PLAYER_SCRIPT.replace("__DATA__", data)
        .replace("__START__", str(start_step))
        .replace("__PWMMAX__", str(2 ** PWM_BITS - 1))
    )
    return _page(body, script)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    """Simulate one day of actuator behaviour and export log, events and figure."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Simulate the lighting actuator for one day.")
    parser.add_argument("--seed", type=int, default=SIM_SEED, help="day-simulation seed")
    args = parser.parse_args()

    from inference import simulate_day_decisions  # local import keeps the module light

    create_directories()
    decisions = simulate_day_decisions(n_days=1, seed=args.seed)
    log, events, summary = simulate_actuator(decisions)

    log.to_csv(ACTUATOR_LOG_CSV, index=False)
    events.to_csv(ACTUATOR_EVENTS_CSV, index=False)
    figure = build_actuator_figure(log)
    figure.savefig(ACTUATOR_RESPONSE_PNG, dpi=150)
    plt.close(figure)

    print("=" * 68)
    print("ACTUATOR SIMULATION (one simulated day, seed %d)" % args.seed)
    print("=" * 68)
    print(f"Driver        : {summary['pwm_bits']}-bit PWM @ {summary['pwm_frequency_hz']} Hz, "
          f"soft fade {summary['fade_rate_pct_per_s']:.0f} %/s "
          f"(0-100 % in {summary['full_range_fade_s']:.1f} s)")
    print(f"Commands sent : {summary['commands_sent']} MQTT-style messages "
          f"({summary['turn_on_events']} turn-on events)")
    print(f"Energy        : {summary['energy_wh']:.1f} Wh "
          f"(instant-switch model {summary['instant_switch_wh']:.1f} Wh, "
          f"fade effect {summary['fade_effect_pct']:+.3f} %)")
    print("\nFirst commands of the day:")
    print(events.head(6).to_string(index=False))
    print(f"\nSaved: {ACTUATOR_LOG_CSV.name}, {ACTUATOR_EVENTS_CSV.name}, "
          f"{ACTUATOR_RESPONSE_PNG.name} in {ACTUATOR_LOG_CSV.parent}")
    print("=" * 68)


if __name__ == "__main__":
    main()
