"""
🚀 Launch Script for Streamlit Demo
===================================

Quick launcher for the anomaly detection demo.
Perfect for staff panel presentations!
"""

import subprocess
import sys
import os

def launch_demo():
    """Launch the Streamlit demo"""
    print("🚀 Launching Anomaly Detection Demo...")
    print("📍 Starting Streamlit server...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(script_dir, "streamlit_demo.py")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", demo_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    launch_demo()