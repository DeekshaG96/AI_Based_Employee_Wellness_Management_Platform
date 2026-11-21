"""Therapeutic Yoga Poses & Asana Remedies for Affective & Somatic Regulation.

Evidence-based yoga postures mapped to specific emotional, cognitive, and physical
workplace strain states (stress, burnout, anxiety, anger, posture slump, mental fatigue).
Includes both full floor asanas and desk/chair modifications for enterprise environments.
"""

from __future__ import annotations

from typing import Any

YOGA_REMEDIES: list[dict[str, Any]] = [
    {
        "id": "balasana",
        "sanskrit_name": "Balasana",
        "english_name": "Child's Pose",
        "category": "Stress & Anxiety Relief",
        "primary_emotions": ["fear", "nervousness", "stress", "anxiety", "overwhelm"],
        "duration": "2 - 3 minutes",
        "difficulty": "Gentle / Restorative",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Stimulates the ventral vagal parasympathetic pathway by gently compressing the abdomen, "
            "supporting adrenal down-regulation and lowering systolic blood pressure. Promotes sensory withdrawal (Pratyahara)."
        ),
        "steps_floor": [
            "Kneel on a mat with knees hip-width or wide apart, big toes touching.",
            "Sink your hips back toward your heels and extend your torso forward.",
            "Rest your forehead gently on the mat (or a folded blanket) and extend arms forward or alongside your body.",
            "Breathe deeply into your posterior ribs and lower back, feeling the expansion with every inhale.",
            "Hold for 8 to 12 slow, rhythmic diaphragmatic breaths.",
        ],
        "office_variation": (
            "**Desk Seated Variation:** Sit toward the front edge of your office chair with feet flat and wide. "
            "Rest your elbows or folded arms on your desk, lowering your forehead onto your arms. "
            "Soften your neck and let your spine decompress forward for 90 seconds while breathing smoothly."
        ),
        "precautions": "Avoid full floor knee flexion if managing acute knee injuries; use the seated desk variation instead.",
        "icon": "🧘",
    },
    {
        "id": "viparita_karani",
        "sanskrit_name": "Viparita Karani",
        "english_name": "Legs-Up-The-Wall Pose",
        "category": "Deep Recovery & Burnout",
        "primary_emotions": ["fatigue", "burnout", "exhaustion", "sadness", "insomnia"],
        "duration": "3 - 5 minutes",
        "difficulty": "Restorative",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Mild restorative inversion that facilitates venous blood return to the heart, reduces sympathetic hyperarousal, "
            "drains lymphatic fluid from sedentary legs, and signals deep somatic safety to the autonomic nervous system."
        ),
        "steps_floor": [
            "Sit sideways close to an open wall with hips against the baseboard.",
            "In one fluid movement, pivot your torso onto the floor and swing your legs straight up against the wall.",
            "Rest your arms out to the sides in a soft cactus or 'T' shape, palms facing upward.",
            "Allow your pelvis and lower back to sink heavily into the ground; close your eyes.",
            "Rest here for 3 to 5 minutes, allowing gravity to facilitate circulation without cardiac strain.",
        ],
        "office_variation": (
            "**Chair Elevation Variation:** In your office, find a quiet spot or couch. "
            "Lie flat on a rug or mat with your calves elevated and resting comfortably horizontally across an office chair seat. "
            "This achieves the exact same hemodynamic restoration without needing an open wall."
        ),
        "precautions": "Avoid during acute neck injuries or uncontrolled severe hypertension without a medical clearance.",
        "icon": "🌊",
    },
    {
        "id": "marjaryasana_bitilasana",
        "sanskrit_name": "Marjaryasana-Bitilasana",
        "english_name": "Cat-Cow Dynamic Flow",
        "category": "Frustration & Anger Decompression",
        "primary_emotions": ["anger", "annoyance", "irritation", "frustration", "stiffness"],
        "duration": "2 minutes (8 - 10 breath cycles)",
        "difficulty": "Gentle",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Coordinates breath with dynamic spinal articulation, mobilizing the cervical, thoracic, and lumbar vertebrae. "
            "Discharges sympathetic fight-or-flight muscular armoring accumulated along the spine and trapezius muscles."
        ),
        "steps_floor": [
            "Start on all fours with wrists directly under shoulders and knees under hips (tabletop position).",
            "**Inhale (Cow Pose):** Drop your belly toward the mat, draw chest forward, and lift gaze and tailbone gently.",
            "**Exhale (Cat Pose):** Press floor firmly away, draw navel into spine, arch your back like a cat, and release neck.",
            "Flow smoothly between Cow on inhale and Cat on exhale for 10 complete, unhurried breath cycles.",
        ],
        "office_variation": (
            "**Seated Chair Cat-Cow:** Sit upright away from your chair back with hands on your knees. "
            "**Inhale:** Arch your spine, pull shoulder blades together, and open your chest forward. "
            "**Exhale:** Round your entire spine, pull belly in, roll shoulders forward, and tuck your chin toward your sternum. "
            "Repeat for 8 to 10 breath cycles right at your desk."
        ),
        "precautions": "Keep neck movements gentle; avoid hyperextending the cervical spine during the inhale.",
        "icon": "🐈",
    },
    {
        "id": "bhujangasana",
        "sanskrit_name": "Salamba Bhujangasana",
        "english_name": "Supported Sphinx / Gentle Cobra",
        "category": "Posture Reset & Lethargy",
        "primary_emotions": ["fatigue", "lethargy", "sadness", "slump", "low_energy"],
        "duration": "1 - 2 minutes",
        "difficulty": "Gentle",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Directly counteracts the kyphotic 'desk hunch' (forward head, rounded shoulders). "
            "Expands the anterior ribcage and diaphragm, improving vital lung capacity, oxygen saturation, and mental alertness."
        ),
        "steps_floor": [
            "Lie on your belly with legs straight and tops of the feet pressing into the mat.",
            "Place elbows under shoulders with forearms parallel on the floor (Sphinx pose).",
            "Press palms down and gently draw your chest forward through your upper arms without crunching the lower back.",
            "Broaden across your collarbones, relax glutes, and gaze softly forward and slightly down.",
            "Hold for 1 to 2 minutes with slow, full diaphragmatic breaths.",
        ],
        "office_variation": (
            "**Desk Chest Opener (Standing or Seated):** Stand or sit tall. Interlace your fingers behind your back "
            "(or hold a lanyard/towel if shoulders are tight). Gently straighten arms and draw shoulder blades together. "
            "Lift your sternum toward the ceiling while keeping your chin level. Hold for 5 deep breaths to reverse desk slouch."
        ),
        "precautions": "Avoid deep backbends during acute lumbar herniation or severe lower back pain; keep the lift mild.",
        "icon": "✨",
    },
    {
        "id": "ardha_matsyendrasana",
        "sanskrit_name": "Ardha Matsyendrasana",
        "english_name": "Seated Spinal Twist (Desk Modification)",
        "category": "Cognitive Reset & Digestion",
        "primary_emotions": ["stress", "annoyance", "overwhelm", "mental_block"],
        "duration": "1 minute per side",
        "difficulty": "Gentle",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Rotational movement decompresses the intervertebral discs, stimulates gastrointestinal motility, "
            "and relieves tension in the obliques, quadratus lumborum, and mid-back rhomboids."
        ),
        "steps_floor": [
            "Sit on the floor with legs extended. Bend right knee and place right foot on the outside of left thigh.",
            "Place right hand on the floor behind your sacrum for support.",
            "Inhale to lengthen spine tall; exhale to hug right knee with left arm or hook left elbow outside right knee.",
            "Gaze gently over the right shoulder. Hold for 5 breaths, then repeat on the opposite side.",
        ],
        "office_variation": (
            "**Chair Desk Twist:** Sit tall in your office chair with feet flat on the floor. "
            "Inhale to lengthen your spine upward. Exhale to rotate your torso to the right, placing your left hand on your right outer knee "
            "and right hand on the chair armrest or back. Breathe deeply for 5 cycles, then gently switch sides."
        ),
        "precautions": "Twist from the thoracic spine and core, never force rotation through the cervical neck.",
        "icon": "🔄",
    },
    {
        "id": "uttanasana",
        "sanskrit_name": "Uttanasana",
        "english_name": "Standing or Seated Forward Fold",
        "category": "Headache & High Anxiety",
        "primary_emotions": ["fear", "anxiety", "nervousness", "headache", "overstimulation"],
        "duration": "1 - 2 minutes",
        "difficulty": "Gentle",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Inversion of the head below the heart slows the heart rate via carotid baroreceptor stimulation. "
            "Stretches the entire posterior fascial chain (hamstrings, glutes, erector spinae) to relieve chronic physical bracing."
        ),
        "steps_floor": [
            "Stand tall with feet hip-distance apart and a micro-bend in your knees.",
            "Exhale and hinge forward from your hip creases, letting your torso drape down over your thighs.",
            "Clasp opposite elbows with your hands (ragdoll pose) and let gravity traction your neck and spine.",
            "Keep weight balanced between balls and heels of feet; shake head gently 'yes' and 'no' to release neck tension.",
            "Hold for 1 to 2 minutes, then slowly roll up vertebra by vertebra on an inhale.",
        ],
        "office_variation": (
            "**Seated Chair Forward Fold:** Sit in your office chair with knees wide and feet flat. "
            "Exhale and fold forward between your thighs, letting your hands dangle toward the floor and your head hang heavy. "
            "Take 6 deep, slow breaths into your back ribs to release tension accumulated from computer screens."
        ),
        "precautions": "Keep knees generously bent if hamstrings or lower back feel tight; avoid if dealing with retinal detachment.",
        "icon": "🌿",
    },
    {
        "id": "tadasana_vrikshasana",
        "sanskrit_name": "Tadasana & Vrikshasana",
        "english_name": "Mountain Pose & Tree Pose",
        "category": "Grounding & Focus Recovery",
        "primary_emotions": ["confusion", "imposter_syndrome", "scattered", "doubt", "neutral"],
        "duration": "1 - 2 minutes",
        "difficulty": "Moderate / Balancing",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Proprioceptive balance poses activate the cerebellar vestibular network and stimulate bilateral cerebral integration. "
            "Grounding through the soles of the feet centers attention, reducing cognitive dispersion and self-doubt."
        ),
        "steps_floor": [
            "**Tadasana (Mountain):** Stand with feet parallel, grounding firmly through all 4 corners of each foot. Engage quadriceps gently.",
            "Align pelvis, lengthen spine, drop shoulders down and back, arms resting with palms forward. Breathe steadily.",
            "**Vrikshasana (Tree):** Shift weight to left foot. Place right sole against inner left calf or inner thigh (never on the knee).",
            "Bring palms together at heart center (Anjali Mudra). Fix your gaze on a non-moving focal point (Drishti).",
            "Hold for 5 steady breaths, then gently lower and repeat on the opposite leg.",
        ],
        "office_variation": (
            "**Desk Supported Tree Pose:** Stand beside your desk and lightly place one hand on the desk surface for balance. "
            "Place the opposite foot on your lower calf. Focus your gaze on a calm point on the horizon. "
            "This provides the full vestibular centering benefit while keeping you safe and steady in an office setting."
        ),
        "precautions": "Never place the raised foot directly against the side of the supporting knee joint.",
        "icon": "🌳",
    },
    {
        "id": "setu_bandhasana",
        "sanskrit_name": "Setu Bandhasana",
        "english_name": "Supported Bridge Pose",
        "category": "Hip Opening & Stamina",
        "primary_emotions": ["fatigue", "grief", "slump", "discouragement"],
        "duration": "2 minutes",
        "difficulty": "Gentle / Moderate",
        "desk_friendly": True,
        "somatic_mechanism": (
            "Opens the anterior hip flexors (iliopsoas) that become chronically shortened from 8+ hours of daily chair sitting. "
            "Elevates the heart slightly above the head, providing an energizing tonic to the endocrine system."
        ),
        "steps_floor": [
            "Lie on your back with knees bent, feet hip-width apart and flat on the floor, heels close to fingertips.",
            "Press into your feet and arms to lift your hips toward the ceiling until thighs and torso form a straight line.",
            "Roll shoulders underneath you and interlace fingers below pelvis (or slide a yoga block under sacrum for restorative version).",
            "Keep knees tracking forward over your ankles; hold for 6 to 8 slow, expansive breaths.",
            "Release arms and slowly lower your spine to the mat on an exhale.",
        ],
        "office_variation": (
            "**Standing Desk Hip & Psoas Stretch:** Stand tall and place hands on your desk. "
            "Step your left foot back into a gentle lunge with your left heel elevated. Tuck your pelvis slightly under "
            "until you feel a soothing opening along the front of your left hip. Hold for 30 seconds per side."
        ),
        "precautions": "Do not turn your head from side to side while in bridge pose to protect the cervical vertebrae.",
        "icon": "🌉",
    },
]


EMOTION_TO_YOGA_MAPPING: dict[str, str] = {
    # Stress & Anxiety
    "fear": "balasana",
    "nervousness": "balasana",
    "anxiety": "balasana",
    "worry": "balasana",
    "panic": "uttanasana",
    # Burnout & Fatigue
    "fatigue": "viparita_karani",
    "burnout": "viparita_karani",
    "exhaustion": "viparita_karani",
    "sadness": "bhujangasana",
    "grief": "setu_bandhasana",
    "slump": "bhujangasana",
    # Frustration & Anger
    "anger": "marjaryasana_bitilasana",
    "annoyance": "marjaryasana_bitilasana",
    "frustration": "marjaryasana_bitilasana",
    "disgust": "ardha_matsyendrasana",
    # Scattered & Focus
    "confusion": "tadasana_vrikshasana",
    "doubt": "tadasana_vrikshasana",
    "neutral": "tadasana_vrikshasana",
    "joy": "bhujangasana",
}


def get_all_yoga_remedies() -> list[dict[str, Any]]:
    """Return catalog of all curated therapeutic yoga remedies."""
    return YOGA_REMEDIES


def get_yoga_remedy_for_emotion(emotion: str) -> dict[str, Any]:
    """Retrieve the targeted yoga remedy for a detected emotion or stress state."""
    clean_emo = str(emotion).lower().strip()
    remedy_id = EMOTION_TO_YOGA_MAPPING.get(clean_emo, "balasana")
    for remedy in YOGA_REMEDIES:
        if remedy["id"] == remedy_id:
            return remedy
    return YOGA_REMEDIES[0]


def get_yoga_categories() -> list[str]:
    """Return distinct categories of yoga remedies."""
    categories: list[str] = []
    for r in YOGA_REMEDIES:
        cat = r.get("category", "General")
        if cat not in categories:
            categories.append(cat)
    return categories
