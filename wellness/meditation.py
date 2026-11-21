"""Medito-inspired open-source mindfulness, meditation packs, and ambient soundscapes.

Referencing the open-source Medito Foundation architecture (meditohq/medito-app):
- 100% privacy-preserving, accessible mindfulness for workplace well-being
- Goal-specific meditation packs (Stress, Focus, Sleep, Self-Compassion)
- Client-side Web Audio API ambient soundscape generators (Rain, Ocean Waves, Zen Bell, Brown Noise)
- Mindful meditation timer with interval bells
"""

from __future__ import annotations

from typing import Any


def get_meditation_packs() -> list[dict[str, Any]]:
    """Return catalog of structured, evidence-based Medito mindfulness packs."""
    return [
        {
            "id": "stress_release",
            "title": "The AnchorPoint Protocol (Stress & Decompression)",
            "icon": "waves",
            "duration": "3-5 mins per session",
            "description": "Somatic grounding techniques for high-stress sprints, tense meetings, and task overwhelm.",
            "sessions": [
                {
                    "title": "Session 1: Dropping the Anchor",
                    "duration": "3 mins",
                    "type": "Somatic Grounding",
                    "guidance": [
                        "Sit upright, feet flat on the floor. Feel the contact with your chair.",
                        "Notice the emotional storm or tension in your chest or shoulders without fighting it.",
                        "Slowly press your feet firmly into the floor. Acknowledge: 'There is stress here, and I am safely grounded.'",
                        "Take 3 extended, gentle breaths. Expand your awareness to include both the physical room and your body.",
                    ],
                },
                {
                    "title": "Session 2: Thought Defusion (Leaves on a Stream)",
                    "duration": "4 mins",
                    "type": "CBT & ACT Mindfulness",
                    "guidance": [
                        "Close your eyes and visualize a gently flowing stream with leaves drifting by.",
                        "As anxious work thoughts arise (deadlines, emails, blockers), place each thought on a leaf.",
                        "Watch the leaf float down the stream around the bend. Do not push it away or dive into the water.",
                        "If you get hooked by a thought, gently note: 'I got hooked,' and step back to the riverbank.",
                    ],
                },
                {
                    "title": "Session 3: 3-Minute Post-Meeting Reset",
                    "duration": "3 mins",
                    "type": "Autonomic Decompression",
                    "guidance": [
                        "Close laptop screen or step away from your monitor.",
                        "Release your jaw, drop your shoulders, and soften the muscles behind your eyes.",
                        "Take one deep double-inhale (physiological sigh) and release an audible sigh out.",
                        "Spend 60 seconds resting in total stillness before opening your next task.",
                    ],
                },
            ],
        },
        {
            "id": "deep_work_focus",
            "title": "Cerebral Defrag & Flow (Deep Work & Focus)",
            "icon": "target",
            "duration": "2-4 mins per session",
            "description": "Cultivate one-pointed attention, clear mental residue, and enter effortless flow state.",
            "sessions": [
                {
                    "title": "Session 1: Clearing Mental Tabs",
                    "duration": "2 mins",
                    "type": "Cognitive Decluttering",
                    "guidance": [
                        "Imagine your mind having 20 browser tabs open simultaneously.",
                        "Take a slow breath in. As you exhale, mentally close tab #1 (slack messages).",
                        "With the next exhale, close tab #2 (past meeting discussion).",
                        "Bring 100% of your remaining energy to the single document or task in front of you.",
                    ],
                },
                {
                    "title": "Session 2: Breath as an Anchor (Shamatha)",
                    "duration": "4 mins",
                    "type": "One-Pointed Concentration",
                    "guidance": [
                        "Rest your attention at the exact tip of your nose or the rise of your belly.",
                        "Follow each breath from its beginning, through the turn, to the very end of the exhale.",
                        "When your mind wanders - which is normal - simply label it 'thinking' with kindness, and return to the anchor.",
                    ],
                },
            ],
        },
        {
            "id": "body_scan_sleep",
            "title": "SomatoRelease & Sunset Detachment (Body Scan & Wind-Down)",
            "icon": "moon",
            "duration": "5-8 mins",
            "description": "Progressive somatic release for physical desk strain and evening work-life boundary transition.",
            "sessions": [
                {
                    "title": "Session 1: Progressive Desk Strain Release",
                    "duration": "5 mins",
                    "type": "Progressive Relaxation",
                    "guidance": [
                        "Bring attention to the crown of your head and forehead. Consciously release tension in your brow.",
                        "Unclench your teeth and let your tongue rest on the floor of your mouth.",
                        "Roll your shoulders back and down. Feel your collarbones broaden.",
                        "Notice your forearms and wrists from keyboard typing; let them feel heavy and soft.",
                        "Breathe into your lower back and abdomen. Let your belly soften completely.",
                    ],
                },
                {
                    "title": "Session 2: Evening Gratitude & Work Detachment",
                    "duration": "4 mins",
                    "type": "Work-Life Boundary",
                    "guidance": [
                        "Acknowledge whatever was left unfinished today: 'What is done is done; what is left will wait until tomorrow.'",
                        "Name one small moment of competence or human kindness that happened today.",
                        "Mentally hand over your professional responsibilities to tomorrow-you.",
                    ],
                },
            ],
        },
        {
            "id": "compassion_metta",
            "title": "Kith & Kin Empathy Circle (Loving-Kindness & Compassion)",
            "icon": "heart",
            "duration": "3-5 mins",
            "description": "Foster psychological safety, empathy, and reduce interpersonal friction with colleagues.",
            "sessions": [
                {
                    "title": "Session 1: Self-Compassion for Mistakes",
                    "duration": "3 mins",
                    "type": "Mindful Self-Compassion",
                    "guidance": [
                        "Recall a workplace mistake or uncomfortable moment that triggered self-criticism.",
                        "Place a hand over your heart or palm on your forearm. Feel the warm touch.",
                        "Say silently: 'This was a difficult moment. Making errors is part of being human. May I be gentle with myself.'",
                    ],
                },
                {
                    "title": "Session 2: Compassion for Difficult Colleagues",
                    "duration": "4 mins",
                    "type": "Metta Practice",
                    "guidance": [
                        "Bring to mind a colleague with whom you experienced friction recently.",
                        "Recognize: 'Just like me, this person experiences pressure, stress, and wants to do good work.'",
                        "Silently extend the wish: 'May you be free from stress. May our collaboration find ease.'",
                    ],
                },
            ],
        },
    ]


def get_soundscape_catalog() -> list[dict[str, str]]:
    """Return catalog of ambient soundscapes available in the Web Audio player."""
    return [
        {
            "id": "rain",
            "name": "Verdant Canopy Rain",
            "description": "Filtered pink noise with soft droplet acoustics to calm an overactive mind.",
            "category": "Relaxation",
        },
        {
            "id": "ocean",
            "name": "Pacific Tidal Resonance",
            "description": "Rhythmic low-frequency swells modeled on a 0.08 Hz natural oceanic oscillation.",
            "category": "Stress Relief",
        },
        {
            "id": "bell",
            "name": "Zenith 432Hz Solfeggio Chime",
            "description": "Harmonic Solfeggio 432 Hz tone with resonant exponential decay for meditation.",
            "category": "Meditation",
        },
        {
            "id": "brown_noise",
            "name": "Amber Focus Frequency (Brown Noise)",
            "description": "Deeper than white noise; dampens sharp environmental distractions for deep work.",
            "category": "Focus",
        },
    ]


def render_medito_soundscape_widget() -> str:
    """Generate self-contained HTML5 & Web Audio API ambient meditation sanctuary with visual breathing pacer.
    
    Inspired by open-source meditation web apps (vishal1patidar/Meditation-with-me & meditohq/medito-app):
    - Visual breathing pacer ring (Inhale / Hold / Exhale animation)
    - 2-min, 5-min, and 10-min quick session presets
    - Zero external audio dependencies: 100% private client-side Web Audio synthesis (Rain, Ocean, Zen Bell, Brown Noise)
    """
    return """
    <div style="font-family:system-ui,-apple-system,sans-serif; background:radial-gradient(circle at 50% 20%, #064e3b 0%, #022c22 100%); color:#f0fdf4; border-radius:18px; padding:18px 20px; box-shadow:0 12px 30px rgba(2,44,34,0.45); margin:8px 0; border:1px solid rgba(45,212,191,0.25);">
        <!-- Header -->
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.12); padding-bottom:10px; margin-bottom:14px;">
            <div>
                <h3 style="margin:0; font-size:1.25rem; font-weight:800; display:flex; align-items:center; gap:8px; color:#5eead4;">
                    🧘 Meditation Sanctuary · Ambient Pacer
                </h3>
                <small style="color:#a7f3d0; font-size:0.80rem;">Guided breathing pacer & client-side ambient soundscapes</small>
            </div>
            <span style="background:rgba(20,184,166,0.2); border:1px solid rgba(45,212,191,0.4); padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:0.05em;">
                Private & Offline
            </span>
        </div>

        <!-- Duration Presets & Sound Selector -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;">
            <div>
                <label style="display:block; font-size:0.75rem; font-weight:700; margin-bottom:6px; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.04em;">1. Session Duration:</label>
                <div style="display:flex; gap:6px;" id="durationButtons">
                    <button type="button" onclick="selectDuration(120, this)" style="flex:1; padding:7px 0; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2); color:#e2e8f0; border-radius:8px; font-weight:700; font-size:0.82rem; cursor:pointer; transition:all 0.15s;">2 Mins</button>
                    <button type="button" onclick="selectDuration(300, this)" style="flex:1; padding:7px 0; background:#0f766e; border:1px solid #2dd4bf; color:#ffffff; border-radius:8px; font-weight:700; font-size:0.82rem; cursor:pointer; transition:all 0.15s;" class="dur-active">5 Mins</button>
                    <button type="button" onclick="selectDuration(600, this)" style="flex:1; padding:7px 0; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2); color:#e2e8f0; border-radius:8px; font-weight:700; font-size:0.82rem; cursor:pointer; transition:all 0.15s;">10 Mins</button>
                </div>
            </div>
            <div>
                <label style="display:block; font-size:0.75rem; font-weight:700; margin-bottom:6px; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.04em;">2. Ambient Soundscape:</label>
                <select id="soundscapeSelect" style="width:100%; padding:7px 10px; border-radius:8px; border:1px solid #14b8a6; background:#022c22; color:#f0fdf4; font-size:0.84rem; font-weight:600; outline:none;">
                    <option value="rain">🌧️ Verdant Canopy Rain (Pink Noise)</option>
                    <option value="ocean">🌊 Pacific Tidal Resonance (0.08Hz Swell)</option>
                    <option value="bell">🔔 Zenith 432Hz Solfeggio Chime</option>
                    <option value="brown">🍂 Amber Focus Frequency (Brown Noise)</option>
                </select>
            </div>
        </div>

        <!-- Center Breathing Visualizer (Meditation-with-me Signature Circle) -->
        <div style="background:rgba(0,0,0,0.3); border-radius:14px; padding:16px 14px; text-align:center; margin-bottom:14px; border:1px solid rgba(255,255,255,0.06); position:relative; overflow:hidden;">
            <div style="display:flex; justify-content:center; align-items:center; margin-bottom:10px;">
                <!-- Animated Breathing Pacer Circle (Clickable!) -->
                <div id="breathPacerOuter" onclick="toggleAudio()" title="Click circle to Start or Pause" style="width:140px; height:140px; border-radius:50%; background:radial-gradient(circle, rgba(45,212,191,0.25) 0%, rgba(15,118,110,0.1) 70%, transparent 100%); display:flex; flex-direction:column; justify-content:center; align-items:center; border:2px solid rgba(45,212,191,0.4); box-shadow:0 0 25px rgba(20,184,166,0.2); transition:transform 4s ease-in-out, box-shadow 4s ease-in-out; cursor:pointer;">
                    <div id="breathCueText" style="font-size:1.05rem; font-weight:800; color:#5eead4; letter-spacing:0.06em; text-transform:uppercase;">
                        Breathe In
                    </div>
                    <div id="breathSubText" style="font-size:0.72rem; color:#a7f3d0; margin-top:2px;">
                        Tap to Start / Pause
                    </div>
                </div>
            </div>

            <!-- Timer Countdown -->
            <div id="timerDisplay" style="font-size:2.4rem; font-weight:900; letter-spacing:2px; font-variant-numeric:tabular-nums; color:#ffffff; text-shadow:0 2px 10px rgba(0,0,0,0.4); margin-bottom:4px;">
                05:00
            </div>
            <div id="statusText" style="font-size:0.80rem; color:#a7f3d0; margin-bottom:14px; font-style:italic;">
                “Breathe in peace, exhale tension. Allow thoughts to settle naturally.”
            </div>

            <!-- Controls (Always Visible & Clickable) -->
            <div style="display:flex; justify-content:center; gap:10px; align-items:center;">
                <button id="playBtn" onclick="toggleAudio()" style="background:linear-gradient(135deg, #10b981 0%, #0d9488 100%); color:#022c22; font-weight:800; border:none; padding:10px 24px; border-radius:24px; font-size:0.92rem; cursor:pointer; display:flex; align-items:center; gap:8px; box-shadow:0 4px 14px rgba(16,185,129,0.4); transition:all 0.2s;">
                    ▶ Start Meditation
                </button>
                <button id="bellBtn" onclick="playZenBell()" style="background:rgba(255,255,255,0.12); color:#f0fdf4; font-weight:600; border:1px solid rgba(255,255,255,0.2); padding:10px 16px; border-radius:24px; font-size:0.82rem; cursor:pointer; transition:all 0.2s;">
                    🔔 Chime Bell
                </button>
                <button id="resetBtn" onclick="resetSession()" style="background:transparent; color:#94a3b8; font-weight:600; border:none; padding:6px 10px; font-size:0.80rem; cursor:pointer;">
                    Reset
                </button>
            </div>
        </div>

        <!-- Volume Slider & Audio Status -->
        <div style="display:flex; align-items:center; justify-content:space-between; font-size:0.78rem; color:#a7f3d0; padding:0 4px;">
            <span>Volume</span>
            <input id="volumeSlider" type="range" min="0" max="1" step="0.05" value="0.5" oninput="setVolume(this.value)" style="width:65%; accent-color:#2dd4bf; cursor:pointer;">
            <span id="volumeVal" style="font-weight:700; color:#5eead4;">50%</span>
        </div>
    </div>

    <script>
        let audioCtx = null;
        let isPlaying = false;
        let activeNodes = [];
        let gainNode = null;
        let timerInterval = null;
        let breathInterval = null;
        let timeLeft = 300;
        let totalDuration = 300;
        let breathPhase = 0; // 0: Inhale, 1: Hold, 2: Exhale

        function initAudio() {
            if (!audioCtx) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                audioCtx = new AudioContext();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }

        function selectDuration(sec, btn) {
            if (!isPlaying) {
                totalDuration = sec;
                timeLeft = sec;
                document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
                const buttons = document.querySelectorAll('#durationButtons button');
                buttons.forEach(b => {
                    b.style.background = 'rgba(255,255,255,0.08)';
                    b.style.borderColor = 'rgba(255,255,255,0.2)';
                    b.style.color = '#e2e8f0';
                });
                btn.style.background = '#0f766e';
                btn.style.borderColor = '#2dd4bf';
                btn.style.color = '#ffffff';
            }
        }

        function setVolume(val) {
            document.getElementById('volumeVal').innerText = Math.round(val * 100) + '%';
            if (gainNode && audioCtx) {
                gainNode.gain.setValueAtTime(parseFloat(val), audioCtx.currentTime);
            }
        }

        // --- Sound Synthesizers (Web Audio API) ---
        function createRainSound(targetGain) {
            const bufferSize = audioCtx.sampleRate * 2;
            const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
            const data = buffer.getChannelData(0);
            let lastOut = 0.0;
            for (let i = 0; i < bufferSize; i++) {
                const white = Math.random() * 2 - 1;
                lastOut = (lastOut * 0.95) + (white * 0.05);
                data[i] = lastOut * 3.5;
            }
            const noise = audioCtx.createBufferSource();
            noise.buffer = buffer;
            noise.loop = true;

            const filter = audioCtx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(1000, audioCtx.currentTime);

            noise.connect(filter);
            filter.connect(targetGain);
            noise.start();
            activeNodes.push(noise);
        }

        function createBrownNoise(targetGain) {
            const bufferSize = audioCtx.sampleRate * 2;
            const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
            const data = buffer.getChannelData(0);
            let lastOut = 0.0;
            for (let i = 0; i < bufferSize; i++) {
                const white = Math.random() * 2 - 1;
                lastOut = (lastOut + (0.02 * white)) / 1.02;
                data[i] = lastOut * 3.5;
            }
            const noise = audioCtx.createBufferSource();
            noise.buffer = buffer;
            noise.loop = true;

            const filter = audioCtx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(450, audioCtx.currentTime);

            noise.connect(filter);
            filter.connect(targetGain);
            noise.start();
            activeNodes.push(noise);
        }

        function createOceanWaves(targetGain) {
            const bufferSize = audioCtx.sampleRate * 2;
            const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
            const data = buffer.getChannelData(0);
            for (let i = 0; i < bufferSize; i++) {
                data[i] = (Math.random() * 2 - 1) * 0.6;
            }
            const noise = audioCtx.createBufferSource();
            noise.buffer = buffer;
            noise.loop = true;

            const filter = audioCtx.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.setValueAtTime(350, audioCtx.currentTime);
            filter.Q.setValueAtTime(1.5, audioCtx.currentTime);

            const lfo = audioCtx.createOscillator();
            lfo.frequency.setValueAtTime(0.08, audioCtx.currentTime);
            const lfoGain = audioCtx.createGain();
            lfoGain.gain.setValueAtTime(250, audioCtx.currentTime);
            lfo.connect(filter.frequency);
            lfo.start();

            noise.connect(filter);
            filter.connect(targetGain);
            noise.start();
            activeNodes.push(noise, lfo);
        }

        function playZenBell() {
            initAudio();
            const osc = audioCtx.createOscillator();
            const bellGain = audioCtx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(432, audioCtx.currentTime);
            
            bellGain.gain.setValueAtTime(0.45, audioCtx.currentTime);
            bellGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 3.8);

            osc.connect(bellGain);
            bellGain.connect(audioCtx.destination);

            osc.start();
            osc.stop(audioCtx.currentTime + 3.9);
        }

        function startSoundscape() {
            initAudio();
            gainNode = audioCtx.createGain();
            const vol = parseFloat(document.getElementById('volumeSlider').value);
            gainNode.gain.setValueAtTime(vol, audioCtx.currentTime);
            gainNode.connect(audioCtx.destination);

            const selected = document.getElementById('soundscapeSelect').value;
            if (selected === 'rain') {
                createRainSound(gainNode);
            } else if (selected === 'brown') {
                createBrownNoise(gainNode);
            } else if (selected === 'ocean') {
                createOceanWaves(gainNode);
            } else if (selected === 'bell') {
                playZenBell();
            }
        }

        function stopSoundscape() {
            activeNodes.forEach(node => {
                try { node.stop(); } catch(e) {}
            });
            activeNodes = [];
        }

        function formatTime(sec) {
            const m = Math.floor(sec / 60);
            const s = sec % 60;
            return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
        }

        // --- Visual Breathing Pacer Logic (4s Inhale, 4s Hold, 4s Exhale) ---
        function runBreathStep() {
            const pacer = document.getElementById('breathPacerOuter');
            const cue = document.getElementById('breathCueText');
            const sub = document.getElementById('breathSubText');

            if (breathPhase === 0) {
                // Inhale
                cue.innerText = 'Breathe In';
                sub.innerText = 'Expand gently';
                pacer.style.transform = 'scale(1.28)';
                pacer.style.boxShadow = '0 0 35px rgba(45,212,191,0.55)';
                pacer.style.borderColor = '#2dd4bf';
                breathPhase = 1;
            } else if (breathPhase === 1) {
                // Hold
                cue.innerText = 'Hold Gently';
                sub.innerText = 'Rest in stillness';
                pacer.style.transform = 'scale(1.28)';
                pacer.style.boxShadow = '0 0 25px rgba(20,184,166,0.35)';
                pacer.style.borderColor = '#5eead4';
                breathPhase = 2;
            } else {
                // Exhale
                cue.innerText = 'Breathe Out';
                sub.innerText = 'Release tension';
                pacer.style.transform = 'scale(0.88)';
                pacer.style.boxShadow = '0 0 10px rgba(20,184,166,0.15)';
                pacer.style.borderColor = 'rgba(45,212,191,0.3)';
                breathPhase = 0;
            }
        }

        function startBreathingPacer() {
            breathPhase = 0;
            runBreathStep();
            breathInterval = setInterval(runBreathStep, 4000);
        }

        function stopBreathingPacer() {
            clearInterval(breathInterval);
            const pacer = document.getElementById('breathPacerOuter');
            const cue = document.getElementById('breathCueText');
            const sub = document.getElementById('breathSubText');
            pacer.style.transform = 'scale(1.0)';
            pacer.style.boxShadow = '0 0 25px rgba(20,184,166,0.2)';
            pacer.style.borderColor = 'rgba(45,212,191,0.4)';
            cue.innerText = 'Ready';
            sub.innerText = 'Press Start';
        }

        function resetSession() {
            if (isPlaying) {
                toggleAudio();
            }
            timeLeft = totalDuration;
            document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
            document.getElementById('statusText').innerText = '“Breathe in peace, exhale tension. Allow thoughts to settle naturally.”';
            stopBreathingPacer();
        }

        function toggleAudio() {
            initAudio();
            const btn = document.getElementById('playBtn');
            const status = document.getElementById('statusText');

            if (!isPlaying) {
                isPlaying = true;
                btn.innerHTML = '⏸ Pause Session';
                btn.style.background = '#f59e0b';
                btn.style.color = '#78350f';
                status.innerText = 'Meditation session active. Follow the breathing pacer circle.';

                playZenBell();
                startSoundscape();
                startBreathingPacer();

                timerInterval = setInterval(() => {
                    if (timeLeft > 0) {
                        timeLeft--;
                        document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
                    } else {
                        clearInterval(timerInterval);
                        stopSoundscape();
                        stopBreathingPacer();
                        playZenBell();
                        isPlaying = false;
                        btn.innerHTML = '▶ Start Meditation';
                        btn.style.background = 'linear-gradient(135deg, #10b981 0%, #0d9488 100%)';
                        btn.style.color = '#022c22';
                        status.innerText = '✨ Meditation complete. Carry this centered calmness into your day.';
                        timeLeft = totalDuration;
                        document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
                    }
                }, 1000);

            } else {
                isPlaying = false;
                btn.innerHTML = '▶ Resume Meditation';
                btn.style.background = 'linear-gradient(135deg, #10b981 0%, #0d9488 100%)';
                btn.style.color = '#022c22';
                status.innerText = 'Session paused. Resume whenever you are ready.';
                clearInterval(timerInterval);
                stopSoundscape();
                stopBreathingPacer();
            }
        }
    </script>
    """

