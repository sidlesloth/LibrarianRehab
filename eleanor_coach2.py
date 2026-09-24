"""
Eleanor's Arm Coach - Sense / Think / Act prototype
====================================================
Stroke rehabilitation coach for a 72-year-old retired librarian with left-side
hemiparesis and mild cognitive impairment.

Session  : 1) warm-up: seated left-arm raise (4 flowers)
           2) she chooses "Garden" or "Kitchen" by holding her hand on a big circle
           3) garden = pick apples that appear at different places (reach + hold)
              kitchen = stir the soup in slow circles (rhythm and pace are checked)
           4) memory question: which color was your first flower? (reach an answer)
Exercise : seated left-arm raise (shoulder flexion / abduction).
Story    : she reaches for the sun and every successful raise makes a flower
           bloom in her garden (her own goals: gardening and cooking).
Feedback : one short spoken sentence at a time (TTS) + large on-screen text,
           a gold "sun" target circle, an angle arc and a garden progress row.

SENSE  : MediaPipe Pose (left shoulder, elbow, wrist, optional hip), visibility gating,
         One Euro filter, short hold of last valid landmarks.
THINK  : joint angles, calibration of her personal rest posture, state machine
         POSITIONING -> CALIBRATING -> RAISING -> HOLD -> LOWERING -> REST -> ... -> DONE,
         compensation detection (trunk lean, shoulder shrug, bent elbow),
         adaptive target (up after 3 successes, down after 2 partial reps).
ACT    : feedback manager with priorities, cooldown, phrase pools without
         immediate repeats, and a non-blocking TTS thread.

Setup
-----
    pip install opencv-python mediapipe numpy pyttsx3
    python eleanor_coach.py
    (first run downloads a ~10 MB pose model, so it needs internet once)

Camera set-up (important!)
    A front-facing webcam cannot see a forward raise well (the arm points at the
    camera and the angle is foreshortened). Use ONE of:
      a) sideways raise (abduction) while facing the camera, or
      b) forward raise with Eleanor seated side-on, left side towards the camera.
    Her left shoulder, elbow and wrist must be visible. The left hip is
    optional: if visible, trunk lean and shoulder shrug are checked too; if it
    is hidden (e.g. by a desk) the arm angle is measured against straight down.

Keys: q / Esc = quit, d = debug numbers, g / k = choose garden / kitchen,
      1 / 2 = answer the memory question, n = skip to the next part (for demos).
Logs: session_log.csv (warm-up reps) and activity_log.csv (apples, stirs, memory answer).
"""

import csv
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyttsx3
except ImportError:  # the coach still works with on-screen text only
    pyttsx3 = None

# ----------------------------------------------------------------------------
# Configuration (tune these during testing)
# ----------------------------------------------------------------------------
CAMERA_INDEX = 0
WARMUP_REPS = 4            # warm-up arm raises (one flower each)
MAX_EXTRA_ATTEMPTS = 3    # warm-up ends after WARMUP_REPS + this many attempts
START_TARGET = 60.0       # degrees of shoulder flexion for first reps
MIN_TARGET = 40.0
MAX_TARGET = 100.0
TARGET_STEP = 5.0
HOLD_SECONDS = 2.0
REST_SECONDS = 5.0
IDLE_PROMPT_SECONDS = 7.0
TOLERANCE = 8.0           # hysteresis: stay in HOLD down to target - TOLERANCE
RAISE_START = 15.0        # degrees above rest that counts as "started"
LOWER_END = 20.0          # degrees above rest that counts as "arm is down"
TRUNK_LEAN_LIMIT = 12.0   # degrees of trunk lean change vs calibration
SHRUG_LIMIT = 0.12        # shoulder rise as a fraction of torso length
ELBOW_MIN = 150.0         # elbow angle below this = "bent arm"
VIS_MIN = 0.5             # MediaPipe landmark visibility threshold (arm)
HIP_VIS_MIN = 0.6         # stricter for the optional hip (often hidden by a desk)
LOST_GRACE = 0.7          # seconds to keep last valid landmarks
CALIB_SAMPLES = 45
SPEECH_COOLDOWN = 5.0     # seconds between non-urgent utterances
SPEECH_RATE = 125         # words per minute (slow for Eleanor), non-Windows voice
WIN_SPEECH_RATE = -2      # Windows voice speed: -10 (slowest) .. 10 (fastest)
LOG_PATH = "session_log.csv"
ACTIVITY_LOG = "activity_log.csv"
# guided session: warm-up -> choice (garden / kitchen) -> memory question
DWELL_S = 2.0             # hold the hand on a circle this long to select it
CHOICE_SPOTS = {"garden": 90.0, "kitchen": 40.0}   # arm angle (deg from straight down)
FRUIT_COUNT = 6           # apples to pick
FRUIT_HOLD = 1.0          # seconds the hand must stay on the apple
FRUIT_PAUSE = 3.5         # pause after a pick so the praise can be heard
FRUIT_IDLE = 9.0          # gentle reminder after this long without progress
FRUIT_STUCK = 25.0        # move the apple closer after this long
STIR_LOOPS = 8            # stirs to complete
STIR_MIN_LOOP_S = 1.5     # faster than this = "a little slower, please"
STIR_WINDOW_S = 3.0       # seconds of wrist history used to find the stirring centre
MODEL_PATH = "pose_landmarker_full.task"  # downloaded automatically on first run
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

FONT = cv2.FONT_HERSHEY_SIMPLEX
FLOWERS = [("red", (60, 60, 220)), ("blue", (220, 120, 40)), ("yellow", (60, 200, 230)),
           ("purple", (170, 80, 150)), ("orange", (40, 140, 240)), ("pink", (180, 120, 230)),
           ("white", (235, 235, 235)), ("teal", (170, 170, 40))]

# Short phrases (<= ~8 words), several variants each, to avoid monotony.
PHRASES = {
    "welcome": ["Hello Eleanor. Let's grow a garden.", "Welcome back, Eleanor. Let's begin."],
    "position": ["Please sit so I can see your arm.", "Move back a little, please.", "Show me your left arm."],
    "calibrate": ["Sit tall. Rest your arm down."],
    "arm_down": ["Rest your left arm down, please."],
    "calib_done": ["Lovely. Now lift your left arm slowly."],
    "raise_cue": ["Lift your left arm slowly.", "Reach for the sun.", "Slowly lift your arm up."],
    "hold": ["Hold it there.", "Nice. Stay up there."],
    "higher": ["A little higher, please.", "Reach up a bit more."],
    "lower": ["Now lower it gently.", "Bring your arm down slowly."],
    "rest": ["Rest and breathe.", "Take your time."],
    "success": ["Well done! A flower blooms.", "Lovely reach!", "Beautiful. Your garden grows."],
    "success_best": ["Higher than last time. Well done!", "That's your best reach yet!"],
    "success_comp": ["Good reach. Next time, sit tall.", "Well done. Keep your shoulder relaxed."],
    "success_up": ["Great progress. Let's reach a bit higher."],
    "partial": ["Good try. That's a start.", "Nice effort. Next one."],
    "tired": ["Your arm looks tired. Rest a moment."],
    "idle": ["Whenever you're ready.", "Take your time. Lift your arm."],
    "lean": ["Sit tall. Lift just the arm.", "Try not to lean sideways."],
    "shrug": ["Relax your shoulder down.", "Let your shoulder drop."],
    "elbow": ["Try to keep your arm straight.", "Stretch your arm out."],
    "lost": ["I can't see your arm.", "Please stay in view."],
    "finish": ["All done. Wonderful work, Eleanor!", "That's everything. Well done, Eleanor!"],
    "warm_done": ["Great warm-up. Now choose your activity."],
    "choice_prompt": ["Garden or kitchen today?", "Reach a circle and hold it."],
    "chose_garden": ["Garden it is. Let's pick apples."],
    "chose_kitchen": ["Kitchen it is. Let's stir the soup."],
    "fruit_cue": ["Reach for the apple.", "Take the apple gently.", "Here's the next apple."],
    "fruit_success": ["Lovely. Apple picked!", "Nice reach. Into the basket.", "Well done. Another apple."],
    "fruit_idle": ["Whenever you're ready. Reach for the apple."],
    "fruit_closer": ["Let's try a closer one."],
    "stir_start": ["Stir the soup in slow circles."],
    "stir_loop": ["Nice stirring.", "Lovely. Keep it going.", "Smooth and steady."],
    "stir_slow": ["A little slower, nice and gentle."],
    "stir_half": ["Halfway there. Well done."],
    "stir_idle": ["Circle your hand slowly.", "Whenever you're ready. Stir the soup."],
}


# ----------------------------------------------------------------------------
# SENSE: filtering and tracking
# ----------------------------------------------------------------------------
class OneEuro:
    """One Euro filter (Casiez et al. 2012): smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=0.8, beta=5.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class ArmTracker:
    """Left arm (+ optional left hip) from MediaPipe Pose, filtered and gated by visibility.

    MediaPipe is run on the UNFLIPPED frame, so 'left' really is Eleanor's left.
    Only the display image is mirrored (see draw code), which avoids the classic
    left/right swap bug.

    Works with both MediaPipe APIs: the legacy `mp.solutions.pose` (older
    versions) and the current Tasks API `PoseLandmarker` (newer versions,
    needs a small model file that is downloaded on first run).
    """

    IDS = {"l_sh": 11, "l_el": 13, "l_wr": 15, "l_hip": 23}
    REQUIRED = ("l_sh", "l_el", "l_wr")

    def __init__(self):
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                                               min_detection_confidence=0.6,
                                               min_tracking_confidence=0.6)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if not os.path.exists(MODEL_PATH):
                print("Downloading pose model (one time, ~10 MB)...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6)
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.t0 = time.time()
        self.last_ts = -1
        self.filters = {k: (OneEuro(), OneEuro()) for k in self.IDS}
        self.last = {}
        self.last_seen = -1e9
        self.hip_seen = -1e9

    def _landmarks(self, frame_bgr, t):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            res = self.pose.process(rgb)
            return res.pose_landmarks.landmark if res.pose_landmarks else None
        ts = max(int((t - self.t0) * 1000), self.last_ts + 1)  # must strictly increase
        self.last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(image, ts)
        return res.pose_landmarks[0] if res.pose_landmarks else None

    def _update(self, name, lm, t):
        fx, fy = self.filters[name]
        self.last[name] = (fx(lm.x, t), fy(lm.y, t))

    def process(self, frame_bgr, t):
        """Returns (visible, points). points are filtered normalized (x, y)."""
        lms = self._landmarks(frame_bgr, t)
        if lms is not None and all(lms[self.IDS[k]].visibility >= VIS_MIN for k in self.REQUIRED):
            for name in self.REQUIRED:
                self._update(name, lms[self.IDS[name]], t)
            self.last_seen = t
            hip = lms[self.IDS["l_hip"]]
            if hip.visibility >= HIP_VIS_MIN and hip.y < 0.97:  # hip really inside the frame
                self._update("l_hip", hip, t)
                self.hip_seen = t
        visible = (all(k in self.last for k in self.REQUIRED)
                   and (t - self.last_seen) < LOST_GRACE)
        if not visible:
            return False, None
        pts = {k: self.last[k] for k in self.REQUIRED}
        if "l_hip" in self.last and (t - self.hip_seen) < LOST_GRACE:
            pts["l_hip"] = self.last["l_hip"]
        return True, pts


# ----------------------------------------------------------------------------
# THINK: geometry and the coaching state machine
# ----------------------------------------------------------------------------
@dataclass
class Measurement:
    flexion: float   # shoulder angle between trunk (or straight down) and upper arm; 0 = arm down
    elbow: float     # 180 = straight arm
    lean: float      # signed trunk tilt from vertical (deg); 0 if hip not visible
    vert: float      # vertical shoulder-hip distance (px); 0 if hip not visible
    torso: float     # shoulder-hip length (px); 0 if hip not visible
    has_hip: bool = True


def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def measure(pts, w, h):
    # convert to pixel space first so angles are not distorted by the aspect ratio
    p = {k: np.array([v[0] * w, v[1] * h]) for k, v in pts.items()}
    sh, el, wr = p["l_sh"], p["l_el"], p["l_wr"]
    hip = p.get("l_hip")
    if hip is not None:
        ref = hip - sh
        up = sh - hip
        lean = math.degrees(math.atan2(up[0], -up[1]))
        vert, torso = float(hip[1] - sh[1]), float(np.linalg.norm(up))
    else:
        ref = np.array([0.0, 1.0])  # straight down in the image
        lean, vert, torso = 0.0, 0.0, 0.0
    return Measurement(
        flexion=angle_between(ref, el - sh),
        elbow=angle_between(sh - el, wr - el),
        lean=lean, vert=vert, torso=torso, has_hip=hip is not None,
    )


class Debounce:
    """A condition must hold continuously for `dur` seconds before it counts."""

    def __init__(self, dur):
        self.dur, self.since = dur, None

    def update(self, cond, t):
        if not cond:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        return t - self.since >= self.dur

    def reset(self):
        self.since = None


class State(Enum):
    POSITIONING = auto()
    CALIBRATING = auto()
    RAISING = auto()
    HOLD = auto()
    LOWERING = auto()
    REST = auto()
    CHOICE = auto()
    FRUIT = auto()
    STIR = auto()
    RECALL = auto()
    DONE = auto()


HINTS = {
    State.POSITIONING: "Sit back so your left arm is in view",
    State.CALIBRATING: "Sit tall with your left arm resting down",
    State.RAISING: "Reach your left arm up to the sun",
    State.HOLD: "Hold it there",
    State.LOWERING: "Lower your arm gently",
    State.REST: "Rest and breathe",
    State.CHOICE: "Reach a circle and hold it",
    State.FRUIT: "Reach the apple and hold it",
    State.STIR: "Stir the soup in slow circles",
    State.RECALL: "Reach the right color and hold it",
    State.DONE: "Session complete",
}

WARMUP_STATES = (State.RAISING, State.HOLD, State.LOWERING, State.REST)


class Session:
    """Holds the exercise logic. update() is called once per frame."""

    def __init__(self, fb):
        self.fb = fb
        self.state = State.POSITIONING
        self.state_t = time.time()
        self.target = START_TARGET
        self.base = None
        self.samples = []
        self.calib_progress = 0.0
        self.vis_since = None
        self.reps = []
        self.successes = 0
        self.success_streak = 0
        self.partial_streak = 0
        self.best_peak = 0.0
        self.extra_rest = 0.0
        self.arc_sign = 1
        self.warn_now = False
        self.hold_frac = 0.0
        self.deb = {"lean": Debounce(0.6), "shrug": Debounce(0.6), "elbow": Debounce(0.8)}
        self.px = None            # left shoulder/elbow/wrist in image pixels (unflipped)
        self.out_sign = 1         # +1 / -1: which image side her arm moves to
        self.reach = None         # straight-arm reach in pixels
        self.activity = None      # "garden" or "kitchen"
        self.act_score = 0
        self.dwell, self.dwell_frac = {}, {}
        self.prompt_t = self.state_t
        self.recall_slots = {}
        self.recall_result = None
        self.finish_said = False
        self._reset_fruit(self.state_t)
        self._reset_stir(self.state_t)
        self._reset_rep(self.state_t)

    # -- helpers -------------------------------------------------------------
    def enter(self, state, t):
        self.state, self.state_t = state, t

    def _reset_rep(self, t):
        self.started = False
        self.peak = 0.0
        self.best_prog = 0.0
        self.last_progress = t
        self.hold_start = None
        self.hold_frac = 0.0
        self.flags = {"lean": False, "shrug": False, "elbow": False}
        self.rep_start = t
        for d in self.deb.values():
            d.reset()

    def _begin_rep(self, t, cue=True):
        self._reset_rep(t)
        self.enter(State.RAISING, t)
        if cue:
            self.fb.say("raise_cue", urgent=True)

    def _done_condition(self):
        return (self.successes >= WARMUP_REPS
                or len(self.reps) >= WARMUP_REPS + MAX_EXTRA_ATTEMPTS)

    # -- main entry ------------------------------------------------------------
    def update(self, m, t, px=None):
        fb = self.fb
        fb.now = t
        self.px = px
        if self.state == State.DONE:
            self._update_done(t)
            return
        if m is None:  # arm lost
            self.vis_since = None
            self.warn_now = False
            if self.state in (State.POSITIONING, State.CALIBRATING):
                if t - self.state_t > 2:
                    fb.say("position")
            elif self.state in (State.RAISING, State.HOLD, State.LOWERING, State.CHOICE,
                                State.FRUIT, State.STIR, State.RECALL):
                fb.say("lost")
                if self.state == State.HOLD:
                    self.state, self.hold_start, self.hold_frac = State.RAISING, None, 0.0
                self.dwell, self.dwell_frac = {}, {}
                self.fruit_hold_start, self.fruit_hold_frac = None, 0.0
                self.stir_last_ang = None
                self.stir_win.clear()
            return
        if self.vis_since is None:
            self.vis_since = t
        {
            State.POSITIONING: self._positioning,
            State.CALIBRATING: self._calibrating,
            State.RAISING: self._raising,
            State.HOLD: self._hold,
            State.LOWERING: self._lowering,
            State.REST: self._rest,
            State.CHOICE: self._choice,
            State.FRUIT: self._fruit,
            State.STIR: self._stir,
            State.RECALL: self._recall,
        }[self.state](m, t)

    # -- states ------------------------------------------------------------------
    def _positioning(self, m, t):
        if t - self.vis_since >= 2.0:
            self.samples = []
            self.enter(State.CALIBRATING, t)
            self.fb.say("calibrate", urgent=True)

    def _calibrating(self, m, t):
        if m.flexion > 25:
            self.fb.say("arm_down")
            return
        reach_now = (float(np.linalg.norm(self.px["l_wr"] - self.px["l_sh"]))
                     if self.px is not None else 0.0)
        self.samples.append((m.flexion, m.lean, m.vert, m.torso, 1.0 if m.has_hip else 0.0, reach_now))
        self.calib_progress = min(1.0, len(self.samples) / CALIB_SAMPLES)
        if len(self.samples) >= CALIB_SAMPLES:
            arr = np.array(self.samples)
            with_hip = arr[arr[:, 4] > 0.5]
            self.base = {"flex": float(np.median(arr[:, 0])), "lean": None, "vert": None, "torso": None}
            reach = float(np.median(arr[:, 5]))
            self.reach = reach if reach > 50 else None
            if len(with_hip) >= CALIB_SAMPLES // 2:
                med = np.median(with_hip[:, 1:4], axis=0)
                self.base.update(lean=float(med[0]), vert=float(med[1]), torso=float(med[2]))
            self._begin_rep(t, cue=False)
            self.fb.say("calib_done", urgent=True)

    def _check_form(self, m, t, elbow=True):
        if m.flexion < 30:  # only judge form once the arm is actually up
            for d in self.deb.values():
                d.reset()
            self.warn_now = False
            return
        bad = {"lean": False, "shrug": False, "elbow": False}
        if m.has_hip and self.base["lean"] is not None:
            lean_dev = abs(m.lean - self.base["lean"])
            shrug = (m.vert - self.base["vert"]) / max(self.base["torso"], 1.0)
            bad["lean"] = self.deb["lean"].update(lean_dev > TRUNK_LEAN_LIMIT, t)
            bad["shrug"] = self.deb["shrug"].update(shrug > SHRUG_LIMIT, t)
        else:  # hip not visible: trunk checks are skipped, arm checks still work
            self.deb["lean"].reset()
            self.deb["shrug"].reset()
        bad["elbow"] = self.deb["elbow"].update(elbow and m.elbow < ELBOW_MIN, t)
        self.warn_now = any(bad.values())
        for key in ("lean", "shrug", "elbow"):  # priority order
            if bad[key]:
                self.flags[key] = True
        for key in ("lean", "shrug", "elbow"):
            if bad[key]:
                self.fb.say(key)
                break

    def _raising(self, m, t):
        f, rest = m.flexion, self.base["flex"]
        self._learn_geometry(m)
        if not self.started and f > rest + RAISE_START:
            self.started = True
            self.last_progress = t
        self.peak = max(self.peak, f)
        if f > self.best_prog + 3:
            self.best_prog, self.last_progress = f, t
        self._check_form(m, t)

        if f >= self.target:
            self.enter(State.HOLD, t)
            self.hold_start = t
            self.fb.say("hold", urgent=True)
        elif self.started and self.peak >= rest + 25 and f < rest + 10:
            self.finish_rep(t, success=False)  # went up, came back without reaching target
        elif t - self.last_progress > IDLE_PROMPT_SECONDS:
            self.fb.say("higher" if self.started else "idle")
            self.last_progress = t

    def _hold(self, m, t):
        f = m.flexion
        self._learn_geometry(m)
        self.peak = max(self.peak, f)
        if f < self.target - TOLERANCE:  # hysteresis: only leave HOLD when clearly below
            self.enter(State.RAISING, t)
            self.hold_start, self.hold_frac, self.last_progress = None, 0.0, t
            self.fb.say("higher")
            return
        self._check_form(m, t)
        held = t - self.hold_start
        self.hold_frac = min(1.0, held / HOLD_SECONDS)
        if held >= HOLD_SECONDS:
            self.finish_rep(t, success=True)

    def _lowering(self, m, t):
        if m.flexion < self.base["flex"] + LOWER_END:
            self.hold_frac = 0.0
            if self._done_condition():
                self._warmup_done(t)
            else:
                self.enter(State.REST, t)
                self.fb.say("rest")
        elif t - self.state_t > 3:
            self.fb.say("lower")

    def _rest(self, m, t):
        if t - self.state_t >= REST_SECONDS + self.extra_rest:
            self.extra_rest = 0.0
            self._begin_rep(t)

    # -- rep bookkeeping and adaptation --------------------------------------------
    def finish_rep(self, t, success):
        fb = self.fb
        peak = self.peak
        improved = success and peak > self.best_peak + 3
        self.best_peak = max(self.best_peak, peak)
        comp = any(self.flags.values())
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": len(self.reps) + 1,
            "target_deg": round(self.target, 1), "peak_deg": round(peak, 1),
            "success": int(success), "lean": int(self.flags["lean"]),
            "shrug": int(self.flags["shrug"]), "elbow": int(self.flags["elbow"]),
            "duration_s": round(t - self.rep_start, 1),
        }
        self.reps.append(rec)
        self._log(rec)

        if success:
            self.successes += 1
            self.success_streak += 1
            self.partial_streak = 0
            if self.successes == 1:
                fb.say_raw(f"Your first flower is {FLOWERS[0][0]}. Remember it!")
            elif self.success_streak >= 3 and self.target < MAX_TARGET:
                self.target = min(MAX_TARGET, self.target + TARGET_STEP)
                self.success_streak = 0
                fb.say("success_up", urgent=True)
            elif comp:
                fb.say("success_comp", urgent=True)
            elif improved:
                fb.say("success_best", urgent=True)
            else:
                fb.say("success", urgent=True)
        else:
            self.success_streak = 0
            self.partial_streak += 1
            if self.partial_streak >= 2:  # looks fatigued: ease off and rest longer
                self.target = max(MIN_TARGET, self.target - TARGET_STEP)
                self.partial_streak = 0
                self.extra_rest = 4.0
                fb.say("tired", urgent=True)
            else:
                fb.say("partial", urgent=True)
        self.enter(State.LOWERING, t)

    def _log(self, rec):
        new = not os.path.exists(LOG_PATH)
        try:
            with open(LOG_PATH, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rec.keys()))
                if new:
                    wr.writeheader()
                wr.writerow(rec)
        except OSError as exc:
            print("Could not write log:", exc)

    # -- guided session: choice -> activity -> memory question -------------------
    @property
    def hit_r(self):
        return max(40.0, 0.22 * (self.reach or 200.0))

    def spot(self, theta_deg):
        """A point the arm can reach: shoulder + direction(theta) * 85% of reach (image pixels)."""
        th = math.radians(theta_deg)
        d = np.array([self.out_sign * math.sin(th), math.cos(th)])
        return self.px["l_sh"] + d * 0.85 * (self.reach or 200.0)

    def _learn_geometry(self, m):
        """Learn her reach and which side her arm moves to during the warm-up."""
        if self.px is None:
            return
        if m.flexion > 40:
            self.out_sign = 1 if self.px["l_el"][0] >= self.px["l_sh"][0] else -1
        if m.elbow > 150:
            self.reach = max(self.reach or 0.0,
                             float(np.linalg.norm(self.px["l_wr"] - self.px["l_sh"])))

    def _dwell_select(self, spots, t):
        """Hold the wrist inside a circle for DWELL_S seconds to select it."""
        wr, chosen = self.px["l_wr"], None
        self.dwell_frac = {}
        for name, pos in spots.items():
            if np.linalg.norm(wr - pos) <= self.hit_r:
                start = self.dwell.setdefault(name, t)
                frac = (t - start) / DWELL_S
                self.dwell_frac[name] = min(1.0, frac)
                if frac >= 1.0:
                    chosen = name
            else:
                self.dwell.pop(name, None)
        return chosen

    def _alog(self, activity, item, value):
        new = not os.path.exists(ACTIVITY_LOG)
        try:
            with open(ACTIVITY_LOG, "a", newline="") as fh:
                wr = csv.writer(fh)
                if new:
                    wr.writerow(["time", "activity", "item", "value"])
                wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), activity, item, value])
        except OSError as exc:
            print("Could not write log:", exc)

    def _warmup_done(self, t):
        self.enter(State.CHOICE, t)
        self.dwell, self.dwell_frac = {}, {}
        self.prompt_t = t
        self.fb.say("warm_done", urgent=True)

    def _choice(self, m, t):
        if self.px is None:
            return
        chosen = self._dwell_select({n: self.spot(th) for n, th in CHOICE_SPOTS.items()}, t)
        if chosen:
            self._start_activity(chosen, t)
        elif t - self.prompt_t > 12:
            self.prompt_t = t
            self.fb.say("choice_prompt")

    def _start_activity(self, name, t):
        self.activity = name
        self.dwell, self.dwell_frac = {}, {}
        self.act_score = 0
        if name == "garden":
            self._reset_fruit(t)
            self.enter(State.FRUIT, t)
            self.fb.say("chose_garden", urgent=True)
        else:
            self._reset_stir(t)
            self.enter(State.STIR, t)
            self.fb.say("chose_kitchen", urgent=True)
        self._alog(name, "chosen", 1)

    # ---- garden: pick apples that appear at different places ----
    def _reset_fruit(self, t):
        self.fruit_done = 0
        self.fruit_theta = None
        self.fruit_last = None
        self.fruit_pool = [45.0, 70.0, 90.0]
        self.fruit_streak = 0
        self.fruit_spawn_t = t + FRUIT_PAUSE
        self.fruit_spawned_t = t
        self.fruit_nudge = t
        self.fruit_hold_start = None
        self.fruit_hold_frac = 0.0

    def _fruit(self, m, t):
        if self.px is None:
            return
        if self.fruit_theta is None:
            if t >= self.fruit_spawn_t:
                options = [a for a in self.fruit_pool if a != self.fruit_last] or self.fruit_pool
                self.fruit_theta = self.fruit_last = random.choice(options)
                self.fruit_spawned_t = self.fruit_nudge = t
                self.fb.say("fruit_cue", urgent=True)
            return
        self._check_form(m, t, elbow=False)  # bent elbows are fine when picking
        if np.linalg.norm(self.px["l_wr"] - self.spot(self.fruit_theta)) <= self.hit_r:
            if self.fruit_hold_start is None:
                self.fruit_hold_start = t
            self.fruit_hold_frac = min(1.0, (t - self.fruit_hold_start) / FRUIT_HOLD)
            if self.fruit_hold_frac >= 1.0:
                self._fruit_picked(t)
            return
        self.fruit_hold_start, self.fruit_hold_frac = None, 0.0
        waited = t - self.fruit_spawned_t
        if waited > FRUIT_STUCK:  # too hard: move the apple closer and make the pool easier
            self.fruit_theta = max(30.0, self.fruit_theta - 15.0)
            self.fruit_pool = [a for a in self.fruit_pool if a < 100.0]
            self.fruit_spawned_t = self.fruit_nudge = t
            self.fruit_streak = 0
            self._alog("garden", "made_easier", round(waited, 1))
            self.fb.say("fruit_closer", urgent=True)
        elif waited > FRUIT_IDLE and t - self.fruit_nudge > FRUIT_IDLE:
            self.fruit_nudge = t
            self.fb.say("fruit_idle")

    def _fruit_picked(self, t):
        self._alog("garden", self.fruit_done + 1, round(t - self.fruit_spawned_t, 1))
        self.fruit_done += 1
        self.fruit_streak += 1
        self.act_score = self.fruit_done
        if self.fruit_streak >= 3 and 100.0 not in self.fruit_pool:
            self.fruit_pool.append(100.0)  # harder (higher) apples after a good streak
        self.fruit_theta = None
        self.fruit_hold_start, self.fruit_hold_frac = None, 0.0
        if self.fruit_done >= FRUIT_COUNT:
            self._to_recall(t)
        else:
            self.fruit_spawn_t = t + FRUIT_PAUSE
            self.fb.say("fruit_success", urgent=True)

    # ---- kitchen: stir the pot (count circular loops, check the pace) ----
    def _reset_stir(self, t):
        self.stir_win = deque()
        self.stir_center = None
        self.stir_draw = None
        self.stir_last_ang = None
        self.stir_cum = 0.0
        self.stir_loops = 0
        self.stir_last_motion = t + 3.0
        self.stir_last_loop = t
        self.stir_nudge = t

    def _stir(self, m, t):
        if self.px is None:
            return
        wr = self.px["l_wr"]
        self.stir_win.append((t, wr[0], wr[1]))
        while self.stir_win and t - self.stir_win[0][0] > STIR_WINDOW_S:
            self.stir_win.popleft()
        if len(self.stir_win) < 20:
            return
        center = np.mean([(x, y) for _, x, y in self.stir_win], axis=0)
        self.stir_center = center
        self.stir_draw = center if self.stir_draw is None else 0.85 * self.stir_draw + 0.15 * center
        rel = wr - center
        if np.linalg.norm(rel) < max(25.0, 0.10 * (self.reach or 200.0)):  # hand (almost) still
            self.stir_last_ang = None
            if t - self.stir_last_motion > 2.0:
                self.stir_cum = 0.0
            if t - self.stir_last_motion > 8.0 and t - self.stir_nudge > 8.0:
                self.stir_nudge = t
                self.fb.say("stir_idle")
            return
        self.stir_last_motion = t
        ang = math.atan2(rel[1], rel[0])
        if self.stir_last_ang is not None:
            self.stir_cum += (ang - self.stir_last_ang + math.pi) % (2 * math.pi) - math.pi
        self.stir_last_ang = ang
        if abs(self.stir_cum) >= 2 * math.pi:
            self.stir_cum -= math.copysign(2 * math.pi, self.stir_cum)
            self._stir_loop(t)

    def _stir_loop(self, t):
        self.stir_loops += 1
        self.act_score = self.stir_loops
        loop_s = t - self.stir_last_loop
        self.stir_last_loop = t
        self._alog("kitchen", self.stir_loops, round(loop_s, 1))
        if self.stir_loops >= STIR_LOOPS:
            self._to_recall(t)
        elif loop_s < STIR_MIN_LOOP_S:
            self.fb.say("stir_slow", urgent=True)
        elif self.stir_loops == STIR_LOOPS // 2:
            self.fb.say("stir_half", urgent=True)
        else:
            self.fb.say("stir_loop")

    # ---- memory question: which color was your first flower? ----
    def _to_recall(self, t):
        self.enter(State.RECALL, t)
        self.dwell, self.dwell_frac = {}, {}
        pair = [FLOWERS[0], random.choice(FLOWERS[1:])]
        random.shuffle(pair)
        self.recall_slots = {pair[0][0]: (90.0, pair[0][1]), pair[1][0]: (40.0, pair[1][1])}
        self.fb.say_raw("Lovely work. Which color was your first flower?", urgent=True)

    def _recall(self, m, t):
        if self.px is None:
            return
        spots = {n: self.spot(th) for n, (th, _) in self.recall_slots.items()}
        chosen = self._dwell_select(spots, t)
        if chosen:
            self._answer_recall(chosen, t)
        elif t - self.state_t > 30:
            self._answer_recall(None, t)

    def _answer_recall(self, chosen, t):
        correct = FLOWERS[0][0]
        ok = chosen == correct
        self.recall_result = ok if chosen else None
        self._alog("recall", "first_flower_color", int(ok))
        if chosen is None:
            self.fb.say_raw(f"It was {correct}. That's okay.", urgent=True)
        elif ok:
            self.fb.say_raw(f"Yes! It was {correct}. Well remembered!", urgent=True)
        else:
            self.fb.say_raw(f"It was {correct}. Nice try!", urgent=True)
        self.dwell, self.dwell_frac = {}, {}
        self.enter(State.DONE, t)

    def _update_done(self, t):
        if not self.finish_said and t - self.state_t > 6:
            self.finish_said = True
            self.fb.say("finish", urgent=True)

    def on_key(self, ch, t):
        """Keyboard fallback / helper keys: g,k = choose; 1,2 = answer; n = skip ahead."""
        self.fb.now = t
        if self.state == State.CHOICE and ch in ("g", "k"):
            self._start_activity("garden" if ch == "g" else "kitchen", t)
        elif self.state == State.RECALL and ch in ("1", "2") and self.recall_slots:
            names = sorted(self.recall_slots, key=lambda n: -self.recall_slots[n][0])  # upper first
            self._answer_recall(names[int(ch) - 1], t)
        elif ch == "n":
            if self.state in WARMUP_STATES and self.base is not None:
                self._warmup_done(t)
            elif self.state in (State.FRUIT, State.STIR):
                self._to_recall(t)


# ----------------------------------------------------------------------------
# ACT: speech and feedback management
# ----------------------------------------------------------------------------
class Speaker(threading.Thread):
    """Text-to-speech in its own thread so the video loop never blocks.

    Windows: uses the built-in Windows voice through PowerShell (no extra install,
    and it avoids pyttsx3 problems when it runs in a background thread).
    Mac/Linux: uses pyttsx3 if it is installed.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.busy = False
        self.proc = None
        self.warned = False
        self.start()

    def _speak_windows(self, text):
        safe = text.replace("'", "''")
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Rate = {WIN_SPEECH_RATE}; $s.Speak('{safe}')")
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", script],
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.proc.wait()

    def run(self):
        engine = None
        if sys.platform != "win32" and pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", SPEECH_RATE)
            except Exception as exc:  # noqa: BLE001
                print("TTS unavailable, using text only:", exc)
        while True:
            text = self.q.get()
            if text is None:
                break
            self.busy = True
            try:
                if sys.platform == "win32":
                    self._speak_windows(text)
                elif engine is not None:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                if not self.warned:
                    print("Voice not working, showing text only:", exc)
                    self.warned = True
            self.busy = False

    def _stop_current(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    def say(self, text, flush=False):
        if flush:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
            self._stop_current()
        self.q.put(text)

    def stop(self):
        self._stop_current()
        self.q.put(None)


class Feedback:
    """Chooses phrases (no immediate repeats), applies cooldown, drives text + speech."""

    def __init__(self, speaker):
        self.sp = speaker
        self.text = ""
        self.now = time.time()
        self.last_spoken = -1e9
        self.last_phrase = {}

    def say(self, key, urgent=False):
        if not urgent and (self.now - self.last_spoken < SPEECH_COOLDOWN or self.sp.busy):
            return False
        pool = PHRASES[key]
        options = [p for p in pool if p != self.last_phrase.get(key)] or pool
        phrase = random.choice(options)
        self.last_phrase[key] = phrase
        self.say_raw(phrase, urgent)
        return True

    def say_raw(self, text, urgent=True):
        self.text = text
        self.last_spoken = self.now
        self.sp.say(text, flush=urgent)


# ----------------------------------------------------------------------------
# ACT: drawing (display image is mirrored; MediaPipe ran on the unflipped frame)
# ----------------------------------------------------------------------------
def to_disp(p, w, h):
    return int((1 - p[0]) * w), int(p[1] * h)


def dark_panel(img, x0, y0, x1, y1, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (25, 25, 25), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_bar(img, frac, label, y):
    h, w = img.shape[:2]
    x0, x1 = 40, w - 40
    cv2.rectangle(img, (x0, y), (x1, y + 28), (255, 255, 255), 2)
    cv2.rectangle(img, (x0, y), (x0 + int((x1 - x0) * frac), y + 28), (80, 200, 80), -1)
    cv2.putText(img, label, (x0 + 8, y - 8), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_garden(img, sess):
    x0, y0, gap, n = 28, 24, 62, WARMUP_REPS
    dark_panel(img, 5, 5, x0 + n * gap - 10, y0 + 138)
    for i in range(n):
        cx, base_y = x0 + 24 + i * gap, y0 + 92
        cv2.fillPoly(img, [np.array([[cx - 18, base_y], [cx + 18, base_y],
                                     [cx + 12, base_y + 26], [cx - 12, base_y + 26]], np.int32)],
                     (70, 110, 170))
        if i < sess.successes:
            _, col = FLOWERS[i % len(FLOWERS)]
            cv2.line(img, (cx, base_y), (cx, base_y - 38), (60, 160, 60), 5)
            cv2.circle(img, (cx, base_y - 52), 17, col, -1)
            cv2.circle(img, (cx, base_y - 52), 17, (255, 255, 255), 2)
            cv2.circle(img, (cx, base_y - 52), 6, (40, 90, 140), -1)
        else:
            cv2.circle(img, (cx, base_y - 28), 9, (170, 170, 170), 2)
    cv2.putText(img, f"Flowers: {sess.successes} of {n}", (x0, y0 + 128),
                FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def draw_sun(img, sh, el, hv, s, sess):
    # gold "sun" target at the target angle
    th = math.radians(s * sess.target)
    u = hv / (np.linalg.norm(hv) + 1e-6)
    d = np.array([u[0] * math.cos(th) - u[1] * math.sin(th),
                  u[0] * math.sin(th) + u[1] * math.cos(th)])
    length = max(170.0, 1.25 * np.linalg.norm(el - sh))
    tip = (sh + d * length).astype(int)
    cv2.line(img, tuple(sh), tuple(tip), (0, 215, 255), 2)
    for k in range(8):
        ang = math.radians(k * 45)
        r = np.array([math.cos(ang), math.sin(ang)])
        cv2.line(img, tuple((tip + r * 27).astype(int)), tuple((tip + r * 40).astype(int)),
                 (0, 215, 255), 4)
    cv2.circle(img, tuple(tip), 22, (0, 215, 255), -1)
    cv2.circle(img, tuple(tip), 22, (255, 255, 255), 3)
    cv2.putText(img, "sun", (tip[0] - 22, tip[1] - 52), FONT, 0.8, (0, 215, 255), 2, cv2.LINE_AA)


def draw_arm(img, pts, m, sess, debug):
    h, w = img.shape[:2]
    sh, el, wr = (np.array(to_disp(pts[k], w, h)) for k in ("l_sh", "l_el", "l_wr"))
    hip = np.array(to_disp(pts["l_hip"], w, h)) if "l_hip" in pts else None
    col = (0, 140, 255) if sess.warn_now else (0, 200, 0)  # orange = fix form (also spoken)
    if hip is not None:
        cv2.line(img, tuple(hip), tuple(sh), (255, 255, 255), 8)
    cv2.line(img, tuple(sh), tuple(el), col, 10)
    cv2.line(img, tuple(el), tuple(wr), col, 10)
    for pnt in ((hip,) if hip is not None else ()) + (sh, el, wr):
        cv2.circle(img, tuple(pnt), 10, (255, 255, 255), -1)

    hv = (hip - sh).astype(float) if hip is not None else np.array([0.0, 1.0])
    ev = (el - sh).astype(float)
    if m is not None and m.flexion > 20:
        sess.arc_sign = 1 if (hv[0] * ev[1] - hv[1] * ev[0]) >= 0 else -1
    s = sess.arc_sign
    if m is not None and sess.base is not None:
        a1 = math.degrees(math.atan2(hv[1], hv[0]))
        a2 = a1 + s * m.flexion
        start, end = (a1, a2) if s > 0 else (a2, a1)
        cv2.ellipse(img, tuple(sh), (70, 70), 0, start, end, (255, 255, 255), 5)

        if sess.state in WARMUP_STATES:
            draw_sun(img, sh, el, hv, s, sess)
    if debug and m is not None:
        cv2.putText(img, f"flex {m.flexion:5.1f} elbow {m.elbow:5.1f} lean {m.lean:5.1f} "
                    f"hip {int(m.has_hip)} target {sess.target:.0f}", (20, 245), FONT, 0.7,
                    (255, 255, 0), 2, cv2.LINE_AA)


def draw_banner(img, text, hint):
    h, w = img.shape[:2]
    dark_panel(img, 0, h - 150, w, h, 0.7)
    scale = 1.6
    while cv2.getTextSize(text, FONT, scale, 3)[0][0] > w - 40 and scale > 0.6:
        scale -= 0.1
    cv2.putText(img, text, (20, h - 85), FONT, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, hint, (20, h - 30), FONT, 0.9, (200, 220, 255), 2, cv2.LINE_AA)


def put_label(img, text, center, scale=1.0):
    size = cv2.getTextSize(text, FONT, scale, 3)[0]
    org = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), 3, cv2.LINE_AA)


def draw_ring(img, center, radius, frac, color=(80, 200, 80)):
    if frac > 0:
        cv2.ellipse(img, center, (radius, radius), 0, -90, -90 + 360 * frac, color, 8, cv2.LINE_AA)


def spot_disp(sess, theta, w):
    p = sess.spot(theta)
    return int(w - p[0]), int(p[1])  # mirror x for the display image


def draw_choice(img, sess):
    if sess.px is None:
        return
    w = img.shape[1]
    r = int(sess.hit_r)
    for name, theta in CHOICE_SPOTS.items():
        c = spot_disp(sess, theta, w)
        cv2.circle(img, c, r, (80, 170, 80) if name == "garden" else (40, 140, 240), -1)
        cv2.circle(img, c, r, (255, 255, 255), 3)
        put_label(img, name.capitalize(), c)
        draw_ring(img, c, r + 10, sess.dwell_frac.get(name, 0.0))


def draw_recall(img, sess):
    if sess.px is None:
        return
    w = img.shape[1]
    r = int(sess.hit_r)
    for name, (theta, col) in sess.recall_slots.items():
        c = spot_disp(sess, theta, w)
        cv2.circle(img, c, r, col, -1)
        cv2.circle(img, c, r, (255, 255, 255), 3)
        put_label(img, name, c, 0.9)
        draw_ring(img, c, r + 10, sess.dwell_frac.get(name, 0.0))


def draw_fruit(img, sess):
    w = img.shape[1]
    dark_panel(img, 5, 168, 330, 218)
    cv2.putText(img, f"Apples: {sess.fruit_done} of {FRUIT_COUNT}", (16, 202), FONT, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    if sess.fruit_theta is None or sess.px is None:
        return
    c = spot_disp(sess, sess.fruit_theta, w)
    r = int(sess.hit_r * 0.8)
    cv2.circle(img, c, r, (50, 50, 220), -1)
    cv2.circle(img, c, r, (255, 255, 255), 3)
    cv2.ellipse(img, (c[0] + r // 3, c[1] - r), (16, 8), -30, 0, 360, (60, 170, 60), -1)
    draw_ring(img, c, int(sess.hit_r) + 12, sess.fruit_hold_frac)


def draw_stir(img, sess):
    h, w = img.shape[:2]
    dark_panel(img, 5, 168, 330, 218)
    cv2.putText(img, f"Stirs: {sess.stir_loops} of {STIR_LOOPS}", (16, 202), FONT, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    if sess.px is None:
        return
    centre = sess.stir_draw if sess.stir_draw is not None else sess.spot(60.0)
    cx, cy = int(w - centre[0]), int(centre[1])
    wr = (int(w - sess.px["l_wr"][0]), int(sess.px["l_wr"][1]))
    cv2.line(img, wr, (cx, cy), (60, 90, 150), 8)          # spoon
    cv2.circle(img, wr, 14, (60, 90, 150), -1)
    cv2.ellipse(img, (cx, cy), (80, 48), 0, 0, 360, (70, 70, 70), -1)   # pot
    cv2.ellipse(img, (cx, cy - 6), (66, 36), 0, 0, 360, (40, 140, 240), -1)  # soup
    cv2.ellipse(img, (cx, cy), (80, 48), 0, 0, 360, (255, 255, 255), 3)
    draw_ring(img, (cx, cy), 110, min(1.0, abs(sess.stir_cum) / (2 * math.pi)))


def draw_summary(img, sess):
    h, w = img.shape[:2]
    dark_panel(img, w // 2 - 330, h // 2 - 210, w // 2 + 330, h // 2 + 170, 0.8)
    best = max([r["peak_deg"] for r in sess.reps], default=0)
    act = {"garden": f"Apples picked: {sess.act_score}", "kitchen": f"Stirs: {sess.act_score}"}.get(
        sess.activity, "")
    lines = ["Session complete!", f"Flowers grown: {sess.successes}", act,
             f"Best reach: {int(best)} degrees", "Rest well, Eleanor."]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (w // 2 - 300, h // 2 - 140 + i * 60), FONT, 1.4 if i == 0 else 1.0,
                    (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "Press Q to close", (w // 2 - 300, h // 2 + 150), FONT, 0.8, (200, 220, 255), 2,
                cv2.LINE_AA)


def draw_ui(img, sess, pts, m, debug):
    if pts is not None and sess.state != State.DONE:
        draw_arm(img, pts, m, sess, debug)
    draw_garden(img, sess)
    st = sess.state
    if st == State.CALIBRATING:
        draw_bar(img, sess.calib_progress, "Getting to know your posture", img.shape[0] - 200)
    elif st == State.HOLD:
        draw_bar(img, sess.hold_frac, "Hold", img.shape[0] - 200)
    elif st == State.CHOICE:
        draw_choice(img, sess)
    elif st == State.FRUIT:
        draw_fruit(img, sess)
    elif st == State.STIR:
        draw_stir(img, sess)
    elif st == State.RECALL:
        draw_recall(img, sess)
    elif st == State.DONE:
        draw_summary(img, sess)
    draw_banner(img, sess.fb.text or "Hello Eleanor", HINTS[st])


# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = ArmTracker()
    speaker = Speaker()
    fb = Feedback(speaker)
    sess = Session(fb)
    fb.now = time.time()
    fb.say("welcome", urgent=True)
    debug = False
    win = "Eleanor's Arm Coach"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    sized = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera not available.")
                break
            t = time.time()
            h, w = frame.shape[:2]
            visible, pts = tracker.process(frame, t)          # SENSE
            m = measure(pts, w, h) if visible else None       # THINK: geometry
            px = ({k: np.array([v[0] * w, v[1] * h]) for k, v in pts.items()}
                  if visible else None)
            sess.update(m, t, px)                             # THINK + ACT decisions
            disp = cv2.flip(frame, 1)                         # mirror view for the user
            draw_ui(disp, sess, pts, m, debug)                # ACT: visuals
            if not sized:  # fit the window on a laptop screen, keep the camera's aspect ratio
                cv2.resizeWindow(win, 960, int(960 * h / w))
                cv2.moveWindow(win, 30, 10)
                sized = True
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            elif 32 <= key < 127:
                sess.on_key(chr(key).lower(), t)  # g/k choose, 1/2 answer, n skip
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the X
                break
    except KeyboardInterrupt:  # Ctrl+C in the terminal
        pass
    finally:  # always free the camera and close everything
        cap.release()
        cv2.destroyAllWindows()
        speaker.stop()
        print("Camera released. Bye!")


if __name__ == "__main__":
    main()