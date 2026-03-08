# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# learning_analytics.py
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict

class LearningAnalytics:
    """Track and visualize learning progress"""
    
    def __init__(self, analytics_file="learning_analytics.json"):
        self.analytics_file = analytics_file
        self.data = self.load_analytics()
    
    def load_analytics(self) -> Dict:
        """Load analytics data"""
        try:
            with open(self.analytics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "sessions": {},
                "vni_performance": {},
                "learning_progress": {},
                "feedback_history": []
            }
    
    def save_analytics(self):
        """Save analytics data"""
        with open(self.analytics_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_interaction(self, session_id: str, vni_type: str, confidence: float, feedback: str = None):
        """Record an interaction for analytics"""
        if session_id not in self.data["sessions"]:
            self.data["sessions"][session_id] = {
                "start_time": datetime.now().isoformat(),
                "interactions": [],
                "vni_usage": {}
            }
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "vni_type": vni_type,
            "confidence": confidence,
            "feedback": feedback
        }
        
        self.data["sessions"][session_id]["interactions"].append(interaction)
        
        # Update VNI performance
        if vni_type not in self.data["vni_performance"]:
            self.data["vni_performance"][vni_type] = {
                "total_interactions": 0,
                "total_confidence": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            }
        
        vni_data = self.data["vni_performance"][vni_type]
        vni_data["total_interactions"] += 1
        vni_data["total_confidence"] += confidence
        
        if feedback == "positive":
            vni_data["positive_feedback"] += 1
        elif feedback == "negative":
            vni_data["negative_feedback"] += 1
        
        self.save_analytics()
    
    def get_vni_performance_metrics(self) -> Dict:
        """Get performance metrics for all VNIs"""
        metrics = {}
        for vni_type, data in self.data["vni_performance"].items():
            if data["total_interactions"] > 0:
                metrics[vni_type] = {
                    "average_confidence": data["total_confidence"] / data["total_interactions"],
                    "positive_feedback_rate": data["positive_feedback"] / data["total_interactions"],
                    "total_interactions": data["total_interactions"]
                }
        return metrics
    
    def generate_learning_report(self) -> str:
        """Generate a text report of learning progress"""
        metrics = self.get_vni_performance_metrics()
        
        report = "📊 BabyBIONN Learning Analytics Report\n"
        report += "=" * 50 + "\n\n"
        
        for vni_type, metric in metrics.items():
            report += f"🧠 {vni_type.upper()} VNI:\n"
            report += f"   • Average Confidence: {metric['average_confidence']:.1%}\n"
            report += f"   • Positive Feedback Rate: {metric['positive_feedback_rate']:.1%}\n"
            report += f"   • Total Interactions: {metric['total_interactions']}\n\n"
        
        total_sessions = len(self.data["sessions"])
        total_interactions = sum(len(session["interactions"]) for session in self.data["sessions"].values())
        
        report += f"📈 Overall Statistics:\n"
        report += f"   • Total Sessions: {total_sessions}\n"
        report += f"   • Total Interactions: {total_interactions}\n"
        
        return report
    
    def plot_confidence_trend(self):
        """Plot confidence trend over time"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for vni_type in ['medical', 'legal', 'technical']:
            if vni_type in self.data["vni_performance"]:
                data = self.data["vni_performance"][vni_type]
                if data["total_interactions"] > 0:
                    avg_confidence = data["total_confidence"] / data["total_interactions"]
                    ax.bar(vni_type, avg_confidence, label=vni_type, alpha=0.7)
        
        ax.set_ylabel('Average Confidence')
        ax.set_title('VNI Performance by Confidence')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('vni_confidence_trend.png', dpi=150, bbox_inches='tight')
        plt.close() 
