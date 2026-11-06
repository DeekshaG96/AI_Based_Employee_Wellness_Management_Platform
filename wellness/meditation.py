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
            "icon": "🌊",
            "duration": "3–5 mins per session",
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
            "icon": "🎯",
            "duration": "2–4 mins per session",
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
                        "When your mind wanders—which is normal—simply label it 'thinking' with kindness, and return to the anchor.",
                    ],
                },
            ],
        },
        {
            "id": "body_scan_sleep",
            "title": "SomatoRelease & Sunset Detachment (Body Scan & Wind-Down)",
            "icon": "🌙",
            "duration": "5–8 mins",
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
            "icon": "💚",
            "duration": "3–5 mins",
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
            "name": "🌧️ Verdant Canopy Rain",
            "description": "Filtered pink noise with soft droplet acoustics to calm an overactive mind.",
            "category": "Relaxation",
        },
        {
            "id": "ocean",
            "name": "🌊 Pacific Tidal Resonance",
            "description": "Rhythmic low-frequency swells modeled on a 0.08 Hz natural oceanic oscillation.",
            "category": "Stress Relief",
        },
        {
            "id": "bell",
            "name": "🔔 Zenith 432Hz Solfeggio Chime",
            "description": "Harmonic Solfeggio 432 Hz tone with resonant exponential decay for meditation.",
            "category": "Meditation",
        },
        {
            "id": "brown_noise",
            "name": "☕ Amber Focus Frequency (Brown Noise)",
            "description": "Deeper than white noise; dampens sharp environmental distractions for deep work.",
            "category": "Focus",
        },
    ]


def render_medito_soundscape_widget() -> str:
    """Generate self-contained HTML5 & Web Audio API ambient soundscape generator and meditation timer.
    
    Zero external audio dependencies, 100% client-side synthesis.
    """
    return """
    <div style="font-family:system-ui,-apple-system,sans-serif; background:linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); color:#f0fdf4; border-radius:16px; padding:24px; box-shadow:0 10px 25px rgba(6,78,59,0.3); margin:12px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:12px; margin-bottom:16px;">
            <div>
                <h3 style="margin:0; font-size:1.25rem; font-weight:700; display:flex; align-items:center; gap:8px;">
                    🧘 Medito Mindfulness & Ambient Sanctuary
                </h3>
                <small style="color:#a7f3d0; font-size:0.8rem;">Open-source 100% client-side Web Audio synthesis & meditation timer</small>
            </div>
            <span style="background:rgba(255,255,255,0.12); padding:4px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; color:#6ee7b7;">
                🌱 Offline & Private
            </span>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;">
            <!-- Soundscape Selector -->
            <div>
                <label style="display:block; font-size:0.8rem; font-weight:600; margin-bottom:6px; color:#d1fae5;">1. Choose Ambient Backdrop:</label>
                <select id="soundscapeSelect" style="width:100%; padding:10px 12px; border-radius:8px; border:1px solid #10b981; background:#042f2e; color:#f0fdf4; font-size:0.9rem; font-weight:500;">
                    <option value="rain">🌧️ Verdant Canopy Rain (Pink Noise)</option>
                    <option value="ocean">🌊 Pacific Tidal Resonance (0.08Hz Swell)</option>
                    <option value="bell">🔔 Zenith 432Hz Solfeggio Chime (Tibetan Bell)</option>
                    <option value="brown">☕ Amber Focus Frequency (Brown Noise)</option>
                </select>
            </div>

            <!-- Timer Selector -->
            <div>
                <label style="display:block; font-size:0.8rem; font-weight:600; margin-bottom:6px; color:#d1fae5;">2. Meditation Duration:</label>
                <select id="timerDurationSelect" style="width:100%; padding:10px 12px; border-radius:8px; border:1px solid #10b981; background:#042f2e; color:#f0fdf4; font-size:0.9rem; font-weight:500;">
                    <option value="60">1 Minute (Quick Micro-Reset)</option>
                    <option value="180" selected>3 Minutes (Desk Meditation)</option>
                    <option value="300">5 Minutes (Deep Stress Relief)</option>
                    <option value="600">10 Minutes (Full Mindfulness Sitting)</option>
                </select>
            </div>
        </div>

        <!-- Timer Display & Controls -->
        <div style="background:rgba(0,0,0,0.25); border-radius:12px; padding:16px; text-align:center; margin-bottom:16px;">
            <div id="timerDisplay" style="font-size:2.4rem; font-weight:800; letter-spacing:2px; font-variant-numeric:tabular-nums; color:#34d399; margin-bottom:8px;">
                03:00
            </div>
            <div id="statusText" style="font-size:0.85rem; color:#a7f3d0; margin-bottom:14px;">
                Ready to practice. Put on headphones for best immersion.
            </div>

            <div style="display:flex; justify-content:center; gap:12px;">
                <button id="playBtn" onclick="toggleAudio()" style="background:#10b981; hover:background:#059669; color:#064e3b; font-weight:700; border:none; padding:10px 24px; border-radius:24px; font-size:0.95rem; cursor:pointer; display:flex; align-items:center; gap:6px; transition:all 0.2s;">
                    ▶ Start Session
                </button>
                <button id="bellBtn" onclick="playZenBell()" style="background:rgba(255,255,255,0.15); color:#f0fdf4; font-weight:600; border:1px solid rgba(255,255,255,0.25); padding:10px 18px; border-radius:24px; font-size:0.85rem; cursor:pointer;">
                    🔔 Chime Bell
                </button>
            </div>
        </div>

        <!-- Volume Slider -->
        <div style="display:flex; align-items:center; justify-content:space-between; font-size:0.8rem; color:#a7f3d0;">
            <span>Volume</span>
            <input id="volumeSlider" type="range" min="0" max="1" step="0.05" value="0.5" oninput="setVolume(this.value)" style="width:60%; accent-color:#34d399;">
            <span id="volumeVal">50%</span>
        </div>
    </div>

    <script>
        let audioCtx = null;
        let isPlaying = false;
        let activeNodes = [];
        let gainNode = null;
        let timerInterval = null;
        let timeLeft = 180;
        let totalTime = 180;

        function initAudio() {
            if (!audioCtx) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                audioCtx = new AudioContext();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
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
                // Pink noise filter approximation
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

            // 0.1 Hz LFO to modulate wave swelling
            const lfo = audioCtx.createOscillator();
            lfo.frequency.setValueAtTime(0.08, audioCtx.currentTime); // ~12 sec tidal period
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
            osc.frequency.setValueAtTime(432, audioCtx.currentTime); // 432 Hz Solfeggio / Zen Tone
            
            bellGain.gain.setValueAtTime(0.4, audioCtx.currentTime);
            bellGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 3.5);

            osc.connect(bellGain);
            bellGain.connect(audioCtx.destination);

            osc.start();
            osc.stop(audioCtx.currentTime + 3.6);
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

        function toggleAudio() {
            initAudio();
            const btn = document.getElementById('playBtn');
            const status = document.getElementById('statusText');

            if (!isPlaying) {
                isPlaying = true;
                btn.innerHTML = '⏸ Pause Session';
                btn.style.background = '#f59e0b';
                btn.style.color = '#78350f';
                status.innerText = '🌿 Mindfulness session in progress. Rest your eyes or gaze softly.';

                playZenBell(); // Opening chime
                startSoundscape();

                timerInterval = setInterval(() => {
                    if (timeLeft > 0) {
                        timeLeft--;
                        document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
                    } else {
                        // Finished
                        clearInterval(timerInterval);
                        stopSoundscape();
                        playZenBell(); // Closing chime
                        isPlaying = false;
                        btn.innerHTML = '▶ Start Session';
                        btn.style.background = '#10b981';
                        btn.style.color = '#064e3b';
                        status.innerText = '✨ Meditation complete. Take this calm awareness into your next task.';
                        timeLeft = parseInt(document.getElementById('timerDurationSelect').value);
                        document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
                    }
                }, 1000);

            } else {
                isPlaying = false;
                btn.innerHTML = '▶ Resume Session';
                btn.style.background = '#10b981';
                btn.style.color = '#064e3b';
                status.innerText = 'Session paused.';
                clearInterval(timerInterval);
                stopSoundscape();
            }
        }

        document.getElementById('timerDurationSelect').addEventListener('change', function(e) {
            if (!isPlaying) {
                timeLeft = parseInt(e.target.value);
                document.getElementById('timerDisplay').innerText = formatTime(timeLeft);
            }
        });
    </script>
    """

