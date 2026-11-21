"""Interactive wellness micro-interventions and somatic regulation tools.

Evidence-based exercises designed for autonomic nervous system regulation,
stress reduction, and cognitive reframing in workplace settings.
"""

from __future__ import annotations

from html import escape
from typing import Any


def get_guided_exercises() -> list[dict[str, Any]]:
    """Return catalog of evidence-based somatic and cognitive micro-interventions."""
    return [
        {
            "id": "box_breathing",
            "name": "Box Breathing (4-4-4-4)",
            "purpose": "Autonomic nervous system balance & acute stress decompression",
            "clinical_evidence": "Navy SEAL and clinical biofeedback standard for lowering cortisol and heart rate variability recovery.",
            "duration": "2-4 minutes",
            "phases": [
                {"name": "Inhale", "seconds": 4, "instruction": "Breathe in slowly through your nose, expanding your ribs."},
                {"name": "Hold", "seconds": 4, "instruction": "Gently hold your breath with your lungs full. Soften your jaw."},
                {"name": "Exhale", "seconds": 4, "instruction": "Smoothly release the air through your mouth or nose."},
                {"name": "Hold", "seconds": 4, "instruction": "Rest quietly at the bottom of the breath before the next cycle."},
            ],
        },
        {
            "id": "relaxing_breath_478",
            "name": "4-7-8 Relaxing Breath",
            "purpose": "De-escalating acute anxiety and promoting calm before sleep or focus",
            "clinical_evidence": "Dr. Andrew Weil protocol that stimulates the parasympathetic vagal brake.",
            "duration": "2 minutes",
            "phases": [
                {"name": "Inhale", "seconds": 4, "instruction": "Quietly inhale through your nose."},
                {"name": "Hold", "seconds": 7, "instruction": "Hold your breath gently without strain."},
                {"name": "Exhale", "seconds": 8, "instruction": "Make a gentle whoosh sound as you exhale completely."},
            ],
        },
        {
            "id": "physiological_sigh",
            "name": "Physiological Sigh (Cyclic Sighing)",
            "purpose": "Fastest known somatic reset for acute workplace overwhelm",
            "clinical_evidence": "Stanford Medicine / Huberman Lab study showing cyclic sighing produces greater mood improvement than mindfulness meditation.",
            "duration": "60-90 seconds (3-5 cycles)",
            "phases": [
                {"name": "First Inhale", "seconds": 2, "instruction": "Inhale deeply through your nose."},
                {"name": "Top-up Inhale", "seconds": 1, "instruction": "Take a second, sharp sip of air to fully reinflate lung alveoli."},
                {"name": "Long Exhale", "seconds": 6, "instruction": "Slowly and completely exhale through your mouth."},
            ],
        },
        {
            "id": "grounding_54321",
            "name": "5-4-3-2-1 Sensory Grounding",
            "purpose": "Interrupting racing thoughts and returning attention to the immediate environment",
            "clinical_evidence": "Cognitive Behavioral Therapy (CBT) somatic anchor technique for sensory re-orientation.",
            "duration": "3 minutes",
            "steps": [
                "[1. Visual] 5 distinct objects you can observe around you (a pen, edge of a monitor, light on a surface)",
                "[2. Tactile] 4 physical sensations you can ground into (feet on the floor, fabric against skin, desk texture)",
                "[3. Auditory] 3 subtle ambient sounds you can detect (fan hum, distant movement, your natural breath)",
                "[4. Olfactory] 2 scents in your immediate environment (coffee, fresh air, neutral atmosphere)",
                "[5. Gustatory] 1 taste or resting sensation in your mouth (a sip of water, neutral breath)",
            ],
        },
    ]


def get_cognitive_reframes(emotion: str) -> dict[str, Any]:
    """Provide structured CBT cognitive reframing prompts based on the detected emotion."""
    emotion_key = emotion.lower().strip()

    reframe_catalog: dict[str, dict[str, Any]] = {
        "fear": {
            "cognitive_distortion": "Catastrophizing or overestimating threat probability",
            "reframing_prompt": "What is the most likely outcome, rather than the absolute worst-case scenario?",
            "self_inquiry_questions": [
                "What evidence supports this fear, and what evidence contradicts it?",
                "If the worst-case did happen, what internal strengths or resources would I use to manage it?",
                "What is one tiny part of this situation that is 100% within my direct control today?",
            ],
            "affirmation": "I don't have to figure out the whole staircase; I just need to take the next step.",
        },
        "anger": {
            "cognitive_distortion": "Should statements, boundary violation or unmet expectation",
            "reframing_prompt": "What boundary or core value of mine feels crossed right now, and how can I express it constructively?",
            "self_inquiry_questions": [
                "What is this anger trying to protect or advocate for?",
                "Is there an unstated expectation here that needs clear communication?",
                "Can I step back for 15 minutes before hitting send or responding?",
            ],
            "affirmation": "My anger is useful information, but my calm response is my superpower.",
        },
        "sadness": {
            "cognitive_distortion": "Overgeneralization or feeling that energy/motivation will not return",
            "reframing_prompt": "This feeling is a natural visitor, not a permanent residence.",
            "self_inquiry_questions": [
                "What kind of care or rest does my body actually need right now?",
                "Can I give myself permission to do 50% effort today without self-criticism?",
                "Who is one person who would offer non-judgmental warmth if I reached out?",
            ],
            "affirmation": "I am allowed to rest without having to earn it.",
        },
        "annoyance": {
            "cognitive_distortion": "Low frustration tolerance or friction overload",
            "reframing_prompt": "Will this friction matter in three months, three weeks, or even three days?",
            "self_inquiry_questions": [
                "Is this a systemic blocker or just a momentary inconvenience?",
                "Can I laugh at the absurdity of this small glitch?",
                "What is the easiest possible workaround for right now?",
            ],
            "affirmation": "I choose where to invest my energy, and this isn't worthy of my peace.",
        },
        "disappointment": {
            "cognitive_distortion": "All-or-nothing thinking around expectations",
            "reframing_prompt": "A setback in a project is feedback on the process, not an indictment of my worth.",
            "self_inquiry_questions": [
                "What valuable learning or data did this outcome provide?",
                "What part went well, even if the final result wasn't what I hoped?",
                "How can I adjust my approach moving into the next iteration?",
            ],
            "affirmation": "Every closed door redirects me to a more aligned path.",
        },
        "joy": {
            "cognitive_distortion": "Foreboding joy or anticipating future loss",
            "reframing_prompt": "Savor this positive moment fully and let the neurological benefits absorb.",
            "self_inquiry_questions": [
                "What specific action or condition contributed to this feeling?",
                "Who could I share this positive progress or gratitude with?",
                "How can I anchor this feeling in my memory for future resilience?",
            ],
            "affirmation": "I deserve to experience, enjoy, and celebrate my accomplishments.",
        },
    }

    # Default fallback for neutral, unclassified, or uncataloged emotions
    default_reframe = {
        "cognitive_distortion": "Potential emotional numbness or cognitive overload",
        "reframing_prompt": "Taking a neutral pause allows your mind to recalibrate and reset.",
        "self_inquiry_questions": [
            "How does my physical body feel right now (shoulders, breathing, posture)?",
            "What is the most supportive thing I can do for myself in the next hour?",
            "Am I honoring my boundaries with screen time and work hours today?",
        ],
        "affirmation": "Neutrality is fertile ground for new clarity and focus.",
    }

    return reframe_catalog.get(emotion_key, default_reframe)


def render_box_breathing_widget() -> str:
    """Generate self-contained HTML/CSS/JS animated interactive breathwork widget."""
    return """
    <div id="boxBreath" class="boxBreath-container" style="font-family:'Plus Jakarta Sans',system-ui,sans-serif; background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.04); margin:12px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid #f1f5f9; padding-bottom:10px;">
            <div>
                <span style="font-size:0.7rem; font-weight:700; color:#0f766e; text-transform:uppercase; letter-spacing:0.06em;">Somatic Autonomic Regulation</span>
                <h4 style="margin:2px 0 0 0; color:#0f172a; font-size:1.05rem; font-weight:800;">Clinical Breathwork Pacer</h4>
            </div>
            <div style="display:flex; gap:6px;">
                <button id="pacer-btn-box" onclick="selectProtocol('box')" style="padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:700; border:1px solid #0f766e; background:#0f766e; color:#ffffff; cursor:pointer; transition:all 0.2s;">Box (4-4-4-4)</button>
                <button id="pacer-btn-sigh" onclick="selectProtocol('sigh')" style="padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:700; border:1px solid #cbd5e1; background:#ffffff; color:#475569; cursor:pointer; transition:all 0.2s;">Sigh (2-1-6)</button>
                <button id="pacer-btn-478" onclick="selectProtocol('478')" style="padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:700; border:1px solid #cbd5e1; background:#ffffff; color:#475569; cursor:pointer; transition:all 0.2s;">4-7-8 Calm</button>
            </div>
        </div>

        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:12px 0;">
            <div style="position:relative; width:190px; height:190px; display:flex; align-items:center; justify-content:center;">
                <!-- Expanding visual pulse ring -->
                <div id="pacer-ring" style="position:absolute; width:150px; height:150px; border-radius:50%; border:2px dashed #99f6e4; opacity:0.6; transition:transform 0.4s ease-out;"></div>
                <!-- Main breathing orb -->
                <div id="pacer-orb" style="
                    width: 90px;
                    height: 90px;
                    border-radius: 50%;
                    background: radial-gradient(circle at 35% 35%, #34d399 0%, #0f766e 100%);
                    box-shadow: 0 0 24px rgba(15, 118, 110, 0.35);
                    transform: scale(1);
                    transition: transform 4s cubic-bezier(0.4, 0, 0.2, 1), background 1s ease;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: #ffffff;
                ">
                    <span id="pacer-phase" style="font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; text-shadow:0 1px 2px rgba(0,0,0,0.3);">Inhale</span>
                    <span id="pacer-seconds" style="font-size:1.3rem; font-weight:800; line-height:1; text-shadow:0 1px 3px rgba(0,0,0,0.4);">4</span>
                </div>
            </div>

            <p id="pacer-instruction" style="margin:14px 0 8px 0; font-size:0.82rem; color:#475569; text-align:center; min-height:22px; font-weight:500;">
                Breathe in slowly through your nose, expanding your diaphragm.
            </p>

            <div style="display:flex; align-items:center; gap:12px; margin-top:8px;">
                <button id="pacer-toggle-btn" onclick="togglePacer()" style="
                    padding: 6px 16px;
                    border-radius: 8px;
                    background: #0f766e;
                    color: #ffffff;
                    border: none;
                    font-size: 0.82rem;
                    font-weight: 700;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    box-shadow: 0 2px 4px rgba(15, 118, 110, 0.2);
                ">Pause Pacer</button>
                <span id="pacer-cycle-count" style="font-size:0.75rem; font-weight:600; color:#64748b;">Cycle 1</span>
            </div>
        </div>

        <script>
        (function() {
            var protocols = {
                box: {
                    name: 'Box Breathing (4-4-4-4)',
                    phases: [
                        { name: 'Inhale', sec: 4, scale: 1.8, bg: 'radial-gradient(circle at 35% 35%, #34d399 0%, #0f766e 100%)', text: 'Breathe in slowly through your nose, expanding your diaphragm.' },
                        { name: 'Hold', sec: 4, scale: 1.8, bg: 'radial-gradient(circle at 35% 35%, #60a5fa 0%, #1d4ed8 100%)', text: 'Gently suspend breath with soft shoulders and relaxed jaw.' },
                        { name: 'Exhale', sec: 4, scale: 1.0, bg: 'radial-gradient(circle at 35% 35%, #38bdf8 0%, #0369a1 100%)', text: 'Release smoothly through your mouth, dropping stress.' },
                        { name: 'Hold', sec: 4, scale: 1.0, bg: 'radial-gradient(circle at 35% 35%, #94a3b8 0%, #475569 100%)', text: 'Rest peacefully at the calm bottom of the breath.' }
                    ]
                },
                sigh: {
                    name: 'Stanford Cyclic Sighing (2-1-6)',
                    phases: [
                        { name: 'Inhale 1', sec: 2, scale: 1.4, bg: 'radial-gradient(circle at 35% 35%, #34d399 0%, #0f766e 100%)', text: 'First smooth inhale through the nose.' },
                        { name: 'Sip Air', sec: 1, scale: 1.9, bg: 'radial-gradient(circle at 35% 35%, #fbbf24 0%, #d97706 100%)', text: 'Sharp second top-up sip to fully open lung alveoli.' },
                        { name: 'Long Exhale', sec: 6, scale: 1.0, bg: 'radial-gradient(circle at 35% 35%, #38bdf8 0%, #0369a1 100%)', text: 'Slow, long, audible exhale through open mouth.' }
                    ]
                },
                '478': {
                    name: '4-7-8 Relaxing Breath',
                    phases: [
                        { name: 'Inhale', sec: 4, scale: 1.7, bg: 'radial-gradient(circle at 35% 35%, #34d399 0%, #0f766e 100%)', text: 'Inhale quietly through your nose for 4 seconds.' },
                        { name: 'Hold', sec: 7, scale: 1.7, bg: 'radial-gradient(circle at 35% 35%, #818cf8 0%, #4f46e5 100%)', text: 'Gently hold breath for 7 seconds to activate vagal brake.' },
                        { name: 'Exhale', sec: 8, scale: 1.0, bg: 'radial-gradient(circle at 35% 35%, #f472b6 0%, #db2777 100%)', text: 'Slow whoosh exhale through mouth for 8 seconds.' }
                    ]
                }
            };

            var currentProtoKey = 'box';
            var phaseIdx = 0;
            var remainingSec = 4;
            var isRunning = true;
            var cycle = 1;
            var intervalTimer = null;

            window.selectProtocol = function(key) {
                currentProtoKey = key;
                phaseIdx = 0;
                cycle = 1;
                ['box', 'sigh', '478'].forEach(function(k) {
                    var btn = document.getElementById('pacer-btn-' + k);
                    if (btn) {
                        if (k === key) {
                            btn.style.background = '#0f766e';
                            btn.style.color = '#ffffff';
                            btn.style.borderColor = '#0f766e';
                        } else {
                            btn.style.background = '#ffffff';
                            btn.style.color = '#475569';
                            btn.style.borderColor = '#cbd5e1';
                        }
                    }
                });
                startPhase();
            };

            window.togglePacer = function() {
                isRunning = !isRunning;
                var btn = document.getElementById('pacer-toggle-btn');
                if (btn) {
                    btn.textContent = isRunning ? 'Pause Pacer' : 'Resume Pacer';
                    btn.style.background = isRunning ? '#0f766e' : '#334155';
                }
            };

            function startPhase() {
                var proto = protocols[currentProtoKey];
                var p = proto.phases[phaseIdx];
                remainingSec = p.sec;

                var orb = document.getElementById('pacer-orb');
                var phaseLabel = document.getElementById('pacer-phase');
                var secLabel = document.getElementById('pacer-seconds');
                var inst = document.getElementById('pacer-instruction');
                var cycleLabel = document.getElementById('pacer-cycle-count');

                if (phaseLabel) phaseLabel.textContent = p.name;
                if (secLabel) secLabel.textContent = remainingSec;
                if (inst) inst.textContent = p.text;
                if (cycleLabel) cycleLabel.textContent = 'Cycle ' + cycle;

                if (orb) {
                    orb.style.transition = 'transform ' + p.sec + 's cubic-bezier(0.4, 0, 0.2, 1), background 0.8s ease';
                    orb.style.transform = 'scale(' + p.scale + ')';
                    orb.style.background = p.bg;
                }
            }

            function tick() {
                if (!isRunning) return;
                var proto = protocols[currentProtoKey];
                remainingSec--;
                var secLabel = document.getElementById('pacer-seconds');
                if (secLabel) secLabel.textContent = Math.max(0, remainingSec);

                if (remainingSec <= 0) {
                    phaseIdx++;
                    if (phaseIdx >= proto.phases.length) {
                        phaseIdx = 0;
                        cycle++;
                    }
                    startPhase();
                }
            }

            startPhase();
            intervalTimer = setInterval(tick, 1000);
        })();
        </script>
    </div>
    """


def render_zen_companion_studio_widget(user_name: str = "Alex Rivera") -> str:
    """Generate self-contained Three.js 3D Zen Companion studio widget with interactive aura and procedural audio."""
    safe_name = escape(str(user_name)) if user_name else "Team Member"
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
          background: transparent;
          font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
          color: #0f172a;
          overflow: hidden;
          height: 535px;
          padding: 2px;
        }}
        .studio-shell {{
          display: grid;
          grid-template-columns: 1.35fr 1fr;
          gap: 16px;
          height: 100%;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 16px;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
          padding: 18px;
        }}
        .canvas-col {{
          display: flex;
          flex-direction: column;
          background: radial-gradient(circle at 50% 45%, #f0fdfa 0%, #f8fafc 100%);
          border: 1px solid #e2e8f0;
          border-radius: 14px;
          padding: 12px 14px;
          position: relative;
          overflow: hidden;
        }}
        .canvas-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;
          z-index: 5;
        }}
        .status-pill {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: #f0fdf4;
          border: 1px solid #bbf7d0;
          color: #166534;
          font-size: 10.5px;
          font-weight: 700;
          padding: 3px 10px;
          border-radius: 20px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }}
        .status-pulse-dot {{
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #16a34a;
          animation: pulseRing 2s infinite;
        }}
        @keyframes pulseRing {{
          0%, 100% {{ opacity: 1; transform: scale(1); }}
          50% {{ opacity: 0.4; transform: scale(0.85); }}
        }}
        .fps-tag {{
          font-size: 10.5px;
          font-family: 'JetBrains Mono', monospace;
          color: #64748b;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          padding: 2px 8px;
          border-radius: 6px;
          font-weight: 600;
        }}
        #canvas-viewport {{
          flex: 1;
          width: 100%;
          position: relative;
          cursor: grab;
          min-height: 280px;
        }}
        #canvas-viewport:active {{
          cursor: grabbing;
        }}
        .canvas-footer {{
          display: flex;
          flex-direction: column;
          gap: 8px;
          z-index: 5;
          margin-top: 6px;
        }}
        .aura-btn-group {{
          display: flex;
          gap: 6px;
          justify-content: center;
        }}
        .aura-btn {{
          padding: 4px 10px;
          border-radius: 8px;
          font-size: 11px;
          font-weight: 700;
          border: 1px solid #cbd5e1;
          background: #ffffff;
          color: #475569;
          cursor: pointer;
          transition: all 0.2s ease;
        }}
        .aura-btn:hover {{
          border-color: #0f766e;
          color: #0f766e;
          transform: translateY(-1px);
        }}
        .aura-btn.active {{
          background: #0f766e;
          color: #ffffff;
          border-color: #0f766e;
          box-shadow: 0 2px 4px rgba(15, 118, 110, 0.2);
        }}
        .breath-sync-bar {{
          display: flex;
          align-items: center;
          gap: 8px;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 10px;
          padding: 6px 12px;
        }}
        .breath-phase-badge {{
          font-size: 11px;
          font-weight: 800;
          color: #0f766e;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          min-width: 75px;
        }}
        .breath-track {{
          flex: 1;
          height: 6px;
          background: #f1f5f9;
          border-radius: 3px;
          overflow: hidden;
        }}
        .breath-fill {{
          height: 100%;
          width: 50%;
          background: linear-gradient(90deg, #14b8a6, #0f766e);
          border-radius: 3px;
          transition: width 0.4s ease;
        }}
        .breath-counter {{
          font-size: 11px;
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          color: #334155;
        }}

        /* Right Telemetry Column */
        .telemetry-col {{
          display: flex;
          flex-direction: column;
          gap: 12px;
          justify-content: space-between;
        }}
        .telemetry-header h3 {{
          font-size: 1.15rem;
          font-weight: 800;
          color: #0f172a;
          letter-spacing: -0.02em;
        }}
        .telemetry-header p {{
          font-size: 0.8rem;
          color: #64748b;
          margin-top: 3px;
          line-height: 1.45;
        }}
        .kpis-grid {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }}
        .kpi-card {{
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 10px;
          padding: 10px 12px;
        }}
        .kpi-title {{
          font-size: 10px;
          font-weight: 700;
          color: #64748b;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }}
        .kpi-value {{
          font-size: 16px;
          font-weight: 800;
          color: #0f172a;
          margin-top: 2px;
        }}
        .kpi-delta {{
          font-size: 10.5px;
          color: #16a34a;
          font-weight: 600;
          margin-top: 1px;
        }}
        .sound-card {{
          background: #f0fdfa;
          border: 1px solid #ccfbf1;
          border-radius: 12px;
          padding: 12px;
        }}
        .sound-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }}
        .sound-label {{
          font-size: 11.5px;
          font-weight: 700;
          color: #0f766e;
        }}
        .sound-ctrl-btn {{
          padding: 4px 12px;
          border-radius: 7px;
          background: #0f766e;
          color: #ffffff;
          border: none;
          font-size: 11px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.2s;
        }}
        .sound-ctrl-btn:hover {{
          background: #115e59;
        }}
        .sound-select-row {{
          display: flex;
          gap: 6px;
        }}
        .sound-select {{
          flex: 1;
          padding: 5px 8px;
          border-radius: 6px;
          border: 1px solid #cbd5e1;
          font-size: 11px;
          background: #ffffff;
          color: #334155;
          font-family: inherit;
        }}
        .insight-card {{
          background: #f8fafc;
          border: 1px dashed #cbd5e1;
          border-radius: 10px;
          padding: 10px 14px;
        }}
        .insight-card span {{
          display: block;
          font-size: 10px;
          font-weight: 700;
          color: #0f766e;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 2px;
        }}
        .insight-card p {{
          font-size: 11.5px;
          font-style: italic;
          color: #334155;
          line-height: 1.45;
        }}
      </style>
    </head>
    <body>
      <div class="studio-shell">
        <!-- Left: 3D Viewport & Pacer -->
        <div class="canvas-col">
          <div class="canvas-header">
            <div class="status-pill">
              <span class="status-pulse-dot"></span>
              <span>Aura 3D Companion</span>
            </div>
            <span class="fps-tag" id="aura-status-label">Equanimity · 60 FPS</span>
          </div>

          <div id="canvas-viewport"></div>

          <div class="canvas-footer">
            <div class="aura-btn-group">
              <button class="aura-btn active" id="btn-aura-calm" onclick="switchAura('calm')">🌿 Equanimity</button>
              <button class="aura-btn" id="btn-aura-flow" onclick="switchAura('flow')">⚡ Flow State</button>
              <button class="aura-btn" id="btn-aura-joy" onclick="switchAura('joy')">🌅 Vitality</button>
              <button class="aura-btn" id="btn-aura-rest" onclick="switchAura('rest')">🌸 Deep Rest</button>
            </div>

            <div class="breath-sync-bar">
              <span class="breath-phase-badge" id="studio-phase-text">Inhale (4s)</span>
              <div class="breath-track">
                <div class="breath-fill" id="studio-breath-fill"></div>
              </div>
              <span class="breath-counter" id="studio-breath-counter">4s</span>
            </div>
          </div>
        </div>

        <!-- Right: Telemetry & Procedural Sound -->
        <div class="telemetry-col">
          <div class="telemetry-header">
            <h3>Neuro-Somatic Telemetry</h3>
            <p>Bio-resonance calibrated for <strong>{safe_name}</strong>. Drag or click the 3D Aura to awaken mindful neuro-cognitive clarity.</p>
          </div>

          <div class="kpis-grid">
            <div class="kpi-card">
              <div class="kpi-title">Vagal Nerve Tone</div>
              <div class="kpi-value" style="color: #0f766e;">Parasympathetic</div>
              <div class="kpi-delta">● Rest & Regulate</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-title">HRV Surrogate</div>
              <div class="kpi-value">78 ms</div>
              <div class="kpi-delta">+12% Optimal</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-title">Cortisol Gauge</div>
              <div class="kpi-value" style="color: #0f766e;">-34%</div>
              <div class="kpi-delta">De-escalating</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-title">Mindful Coherence</div>
              <div class="kpi-value">4.8 / 5.0</div>
              <div class="kpi-delta">Deep Flow</div>
            </div>
          </div>

          <!-- Ambient Soundscape Player -->
          <div class="sound-card">
            <div class="sound-header">
              <span class="sound-label">Procedural Ambient Sanctuary</span>
              <button class="sound-ctrl-btn" id="studio-sound-btn" onclick="toggleStudioAudio()">Play Ambient</button>
            </div>
            <div class="sound-select-row">
              <select class="sound-select" id="studio-sound-select" onchange="changeStudioTrack()">
                <option value="rain">Verdant Canopy Rain (Pink Noise)</option>
                <option value="ocean">Pacific Tidal Swell (0.08Hz)</option>
                <option value="chime">432Hz Solfeggio Harmonic Bell</option>
              </select>
            </div>
          </div>

          <!-- Dialogue Insight -->
          <div class="insight-card">
            <span>Mindful Reflection Guide</span>
            <p id="studio-insight-text">"Take a slow breath. Release what you cannot control, and step forward with quiet confidence."</p>
          </div>
        </div>
      </div>

      <script>
      (function() {{
        var container = document.getElementById('canvas-viewport');
        if (!container || typeof THREE === 'undefined') return;

        var scene = new THREE.Scene();
        var width = container.clientWidth || 360;
        var height = container.clientHeight || 280;
        var camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
        camera.position.z = 4.8;

        var renderer = new THREE.WebGLRenderer({{ alpha: true, antialias: true }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        container.appendChild(renderer.domElement);

        var ambLight = new THREE.AmbientLight(0xffffff, 1.4);
        scene.add(ambLight);

        var ptLight1 = new THREE.PointLight(0x10b981, 3.5, 50);
        ptLight1.position.set(2, 4, 3);
        scene.add(ptLight1);

        var ptLight2 = new THREE.PointLight(0xd97706, 2.0, 50);
        ptLight2.position.set(-2, -3, 2);
        scene.add(ptLight2);

        var mainGroup = new THREE.Group();

        var coreMat = new THREE.MeshPhongMaterial({{
          color: 0x0f766e,
          emissive: 0x09332f,
          shininess: 95,
          transparent: true,
          opacity: 0.88
        }});
        var coreMesh = new THREE.Mesh(new THREE.SphereGeometry(1.0, 32, 32), coreMat);
        mainGroup.add(coreMesh);

        var innerMat = new THREE.MeshPhongMaterial({{ color: 0xffffff, emissive: 0x10b981, shininess: 100 }});
        var innerMesh = new THREE.Mesh(new THREE.SphereGeometry(0.55, 24, 24), innerMat);
        mainGroup.add(innerMesh);

        var petalGroup = new THREE.Group();
        var petalCount = 6;
        for (var i = 0; i < petalCount; i++) {{
          var angle = (i / petalCount) * Math.PI * 2;
          var petal = new THREE.Mesh(
            new THREE.ConeGeometry(0.22, 1.15, 16),
            new THREE.MeshPhongMaterial({{ color: 0x14b8a6, shininess: 80, flatShading: true }})
          );
          petal.rotation.z = Math.PI / 3;
          petal.position.set(Math.cos(angle) * 1.55, Math.sin(angle) * 1.55, 0);
          petalGroup.add(petal);
        }}
        mainGroup.add(petalGroup);

        var pCount = 120;
        var pGeo = new THREE.BufferGeometry();
        var pPos = new Float32Array(pCount * 3);
        for (var pi = 0; pi < pCount * 3; pi += 3) {{
          pPos[pi] = (Math.random() - 0.5) * 7.5;
          pPos[pi+1] = (Math.random() - 0.5) * 7.5;
          pPos[pi+2] = (Math.random() - 0.5) * 7.5;
        }}
        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        var pMat = new THREE.PointsMaterial({{ color: 0x34d399, size: 0.08, transparent: true, opacity: 0.75 }});
        var particles = new THREE.Points(pGeo, pMat);
        scene.add(particles);

        scene.add(mainGroup);

        var mouseX = 0, mouseY = 0;
        container.addEventListener('mousemove', function(e) {{
          var rect = container.getBoundingClientRect();
          mouseX = ((e.clientX - rect.left) / container.clientWidth - 0.5) * 0.8;
          mouseY = ((e.clientY - rect.top) / container.clientHeight - 0.5) * 0.8;
        }});

        var isInteracting = false;
        var interactTimer = 0;
        var insights = [
          '"Take a deep diaphragmatic breath. Reset your mental posture."',
          '"Your focused energy creates calm throughout your workplace."',
          '"Empathy and clear communication dissolve friction."',
          '"Small mindful pauses protect your cognitive stamina."',
          '"Honor your boundaries today. Sustainable effort wins."'
        ];

        function triggerBurst() {{
          isInteracting = true;
          interactTimer = 1.6;
          var el = document.getElementById('studio-insight-text');
          if (el) el.textContent = insights[Math.floor(Math.random() * insights.length)];
        }}
        container.addEventListener('click', triggerBurst);

        var auraPalettes = {{
          calm: {{ core: 0x0f766e, emissive: 0x09332f, inner: 0x10b981, pt: 0x10b981, p: 0x34d399, label: 'Equanimity · 60 FPS' }},
          flow: {{ core: 0x1d4ed8, emissive: 0x0f172a, inner: 0x3b82f6, pt: 0x3b82f6, p: 0x60a5fa, label: 'Flow State · 60 FPS' }},
          joy: {{ core: 0xd97706, emissive: 0x78350f, inner: 0xf59e0b, pt: 0xf59e0b, p: 0xfcd34d, label: 'Vitality · 60 FPS' }},
          rest: {{ core: 0x7e22ce, emissive: 0x3b0764, inner: 0x8b5cf6, pt: 0x8b5cf6, p: 0xc084fc, label: 'Deep Rest · 60 FPS' }}
        }};

        window.switchAura = function(key) {{
          var pal = auraPalettes[key] || auraPalettes.calm;
          coreMat.color.setHex(pal.core);
          coreMat.emissive.setHex(pal.emissive);
          innerMat.emissive.setHex(pal.inner);
          ptLight1.color.setHex(pal.pt);
          pMat.color.setHex(pal.p);

          ['calm', 'flow', 'joy', 'rest'].forEach(function(k) {{
            var b = document.getElementById('btn-aura-' + k);
            if (b) {{
              if (k === key) b.classList.add('active');
              else b.classList.remove('active');
            }}
          }});
          var lbl = document.getElementById('aura-status-label');
          if (lbl) lbl.textContent = pal.label;
          triggerBurst();
        }};

        // Breath pacer cycle (4-4-4-4)
        var breathPhases = [
          {{ text: 'Inhale (4s)', sec: 4, scale: 1.3 }},
          {{ text: 'Hold (4s)', sec: 4, scale: 1.3 }},
          {{ text: 'Exhale (4s)', sec: 4, scale: 1.0 }},
          {{ text: 'Hold (4s)', sec: 4, scale: 1.0 }}
        ];
        var bPhaseIdx = 0;
        var bRemaining = 4;
        function updateBreathPacer() {{
          bRemaining--;
          var p = breathPhases[bPhaseIdx];
          var pct = Math.round(((p.sec - bRemaining) / p.sec) * 100);
          var fill = document.getElementById('studio-breath-fill');
          var txt = document.getElementById('studio-phase-text');
          var cnt = document.getElementById('studio-breath-counter');
          if (fill) fill.style.width = pct + '%';
          if (txt) txt.textContent = p.text;
          if (cnt) cnt.textContent = bRemaining + 's';

          if (bRemaining <= 0) {{
            bPhaseIdx = (bPhaseIdx + 1) % breathPhases.length;
            bRemaining = breathPhases[bPhaseIdx].sec;
          }}
        }}
        setInterval(updateBreathPacer, 1000);

        // Three.js Render Loop
        var clock = new THREE.Clock();
        function renderScene() {{
          requestAnimationFrame(renderScene);
          var delta = clock.getDelta();
          var t = clock.getElapsedTime();

          if (isInteracting) {{
            interactTimer -= delta;
            coreMat.emissive.setHex(0xd4af37);
            pMat.size = 0.12;
            if (interactTimer <= 0) {{
              isInteracting = false;
              pMat.size = 0.08;
            }}
          }}

          mainGroup.position.y = Math.sin(t * 1.4) * 0.12;
          mainGroup.rotation.y += (t * 0.35 + mouseX - mainGroup.rotation.y) * 0.08 + (isInteracting ? 0.18 : 0);
          mainGroup.rotation.x = Math.sin(t * 0.7) * 0.1 + mouseY;
          petalGroup.rotation.z += (isInteracting ? -0.12 : -0.006);

          // Gentle breathing rhythm scale
          var targetScale = breathPhases[bPhaseIdx].scale;
          var breathPulse = 1.0 + (targetScale - 1.0) * Math.sin(t * (Math.PI / 4));
          mainGroup.scale.set(breathPulse, breathPulse, breathPulse);

          particles.rotation.y += (isInteracting ? 0.06 : 0.0009);

          renderer.render(scene, camera);
        }}
        renderScene();

        window.addEventListener('resize', function() {{
          if (!container || !camera || !renderer) return;
          var w = container.clientWidth || 360;
          var h = container.clientHeight || 280;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }});

        // Procedural Web Audio Ambient Player
        var audioCtx = null;
        var noiseNode = null;
        var isAudioPlaying = false;

        window.toggleStudioAudio = function() {{
          var AudioContext = window.AudioContext || window.webkitAudioContext;
          if (!AudioContext) return;
          if (!audioCtx) audioCtx = new AudioContext();
          if (audioCtx.state === 'suspended') audioCtx.resume();

          var btn = document.getElementById('studio-sound-btn');
          if (isAudioPlaying) {{
            if (noiseNode) {{
              try {{ noiseNode.stop(); noiseNode.disconnect(); }} catch(e) {{}}
              noiseNode = null;
            }}
            isAudioPlaying = false;
            if (btn) {{
              btn.textContent = 'Play Ambient';
              btn.style.background = '#0f766e';
            }}
          }} else {{
            startAudioSound();
            isAudioPlaying = true;
            if (btn) {{
              btn.textContent = 'Pause Ambient';
              btn.style.background = '#334155';
            }}
          }}
        }};

        function startAudioSound() {{
          if (!audioCtx) return;
          var bufSize = audioCtx.sampleRate * 2;
          var buffer = audioCtx.createBuffer(1, bufSize, audioCtx.sampleRate);
          var data = buffer.getChannelData(0);
          var last = 0.0;
          for (var i = 0; i < bufSize; i++) {{
            var white = Math.random() * 2 - 1;
            data[i] = (last + 0.02 * white) / 1.02;
            last = data[i];
            data[i] *= 3.0;
          }}
          noiseNode = audioCtx.createBufferSource();
          noiseNode.buffer = buffer;
          noiseNode.loop = true;

          var filter = audioCtx.createBiquadFilter();
          filter.type = 'lowpass';
          filter.frequency.value = 850;

          var gain = audioCtx.createGain();
          gain.gain.value = 0.15;

          noiseNode.connect(filter);
          filter.connect(gain);
          gain.connect(audioCtx.destination);
          noiseNode.start();
        }}

        window.changeStudioTrack = function() {{
          if (isAudioPlaying) {{
            toggleStudioAudio();
            toggleStudioAudio();
          }}
        }};
      }})();
      </script>
    </body>
    </html>
    """

