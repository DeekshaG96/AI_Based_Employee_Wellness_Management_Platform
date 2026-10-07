"""
Personalized Wellness Recommendation Engine.
Matches employee emotional state & sentiment to targeted wellness recommendations.
"""

from typing import List, Dict, Any

class WellnessRecommender:
    """Generates personalized action plans and resources for employee wellness."""

    def __init__(self):
        self.catalog = {
            "Burnout / Exhaustion": [
                {
                    "title": "Immediate Cognitive Reset",
                    "action": "Take an uninterrupted 15-minute screen-free break with deep breathing.",
                    "tag": "Urgent Wellness",
                    "resource": "Employee Assistance Program (EAP) 24/7 Helpline"
                },
                {
                    "title": "Workload Realignment",
                    "action": "Flag pending blockers with your sprint manager and request task re-prioritization.",
                    "tag": "Work-Life Balance",
                    "resource": "Sprint Health & Capacity Guide"
                }
            ],
            "High Stress / Anxiety": [
                {
                    "title": "Box Breathing & Grounding",
                    "action": "Practice 4x4 box breathing (4s inhale, 4s hold, 4s exhale, 4s hold).",
                    "tag": "Stress Relief",
                    "resource": "Mindfulness Audio Session (10 min)"
                },
                {
                    "title": "Schedule Focus Shield",
                    "action": "Turn on 'Do Not Disturb' on Teams/Slack for a 2-hour deep focus block.",
                    "tag": "Time Management",
                    "resource": "Deep Work Playbook"
                }
            ],
            "Frustration / Friction": [
                {
                    "title": "Constructive 1-on-1 Sync",
                    "action": "Book a 15-minute alignment check-in with your team lead to clarify requirements.",
                    "tag": "Team Dynamics",
                    "resource": "Effective Feedback Framework"
                }
            ],
            "High Morale / Joy": [
                {
                    "title": "Peer Recognition & Kudos",
                    "action": "Give a shoutout or peer appreciation badge to teammates who supported you.",
                    "tag": "Positive Culture",
                    "resource": "Infosys Peer Recognition Portal"
                },
                {
                    "title": "Knowledge Sharing",
                    "action": "Document what went well in the sprint retro to share best practices.",
                    "tag": "Growth",
                    "resource": "Engineering Playbook Wiki"
                }
            ],
            "Satisfaction / Calm": [
                {
                    "title": "Sustained Rhythm Maintenance",
                    "action": "Maintain healthy boundary habits and hydrate regularly throughout the day.",
                    "tag": "Habit Building",
                    "resource": "Daily Wellness Checklist"
                }
            ],
            "Neutral / Routine": [
                {
                    "title": "Micro-Movement & Ergonomics Check",
                    "action": "Stand up, stretch shoulder & neck muscles, and check screen eye distance.",
                    "tag": "Ergonomics",
                    "resource": "Workplace Posture Guide"
                },
                {
                    "title": "Continuous Learning",
                    "action": "Explore a bite-sized 15-minute upskilling module on Infosys Springboard.",
                    "tag": "Upskilling",
                    "resource": "Infosys Springboard Learning Catalog"
                }
            ]
        }

    def recommend(self, sentiment: str, emotion_tone: str) -> List[Dict[str, str]]:
        """Returns personalized recommendations based on sentiment and emotion tone."""
        if emotion_tone in self.catalog:
            return self.catalog[emotion_tone]

        if sentiment == "Negative":
            return self.catalog["High Stress / Anxiety"]
        elif sentiment == "Positive":
            return self.catalog["High Morale / Joy"]
        else:
            return self.catalog["Neutral / Routine"]
