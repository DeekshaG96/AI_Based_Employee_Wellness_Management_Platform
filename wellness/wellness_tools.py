"""Interactive wellness micro-interventions and somatic regulation tools.

Evidence-based exercises designed for autonomic nervous system regulation,
stress reduction, and cognitive reframing in workplace settings.
"""

from __future__ import annotations

from typing import Any


def get_guided_exercises() -> list[dict[str, Any]]:
    """Return catalog of evidence-based somatic and cognitive micro-interventions."""
    return [
        {
            "id": "box_breathing",
            "name": "Box Breathing (4-4-4-4)",
            "purpose": "Autonomic nervous system balance & acute stress decompression",
            "clinical_evidence": "Navy SEAL and clinical biofeedback standard for lowering cortisol and heart rate variability recovery.",
            "duration": "2–4 minutes",
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
            "duration": "60–90 seconds (3–5 cycles)",
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
    """Generate self-contained HTML/CSS animated box breathing widget."""
    return """
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:24px; background:linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); border-radius:16px; border:1px solid #bbf7d0; margin:16px 0;">
        <h4 style="margin:0 0 8px 0; color:#166534; font-family:system-ui,sans-serif;">Box Breathing Visual Pacer (4-4-4-4)</h4>
        <p style="margin:0 0 20px 0; font-size:0.85rem; color:#15803d; text-align:center;">Follow the circle: Inhale as it expands, hold, exhale as it contracts, hold.</p>
        
        <div style="position:relative; width:180px; height:180px; display:flex; align-items:center; justify-content:center;">
            <div style="
                width: 90px;
                height: 90px;
                border-radius: 50%;
                background: radial-gradient(circle, #34d399 0%, #059669 100%);
                box-shadow: 0 0 24px rgba(16, 185, 129, 0.45);
                animation: boxBreath 16s infinite ease-in-out;
            "></div>
            <div style="position:absolute; font-weight:700; color:#064e3b; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; animation: textCycle 16s infinite ease-in-out;">
            </div>
        </div>
        
        <style>
            @keyframes boxBreath {
                0%   { transform: scale(1); opacity: 0.7; }
                25%  { transform: scale(1.9); opacity: 1; }
                50%  { transform: scale(1.9); opacity: 0.9; }
                75%  { transform: scale(1); opacity: 0.7; }
                100% { transform: scale(1); opacity: 0.7; }
            }
            @keyframes textCycle {
                0%, 20%   { content: "Inhale (4s)"; }
                25%, 45%  { content: "Hold (4s)"; }
                50%, 70%  { content: "Exhale (4s)"; }
                75%, 95%  { content: "Hold (4s)"; }
                100%      { content: "Inhale (4s)"; }
            }
        </style>
        <div style="display:flex; justify-content:space-between; width:220px; margin-top:20px; font-size:0.75rem; color:#166534; font-weight:600;">
            <span>Inhale 4s</span>
            <span>Hold 4s</span>
            <span>Exhale 4s</span>
            <span>Hold 4s</span>
        </div>
    </div>
    """

