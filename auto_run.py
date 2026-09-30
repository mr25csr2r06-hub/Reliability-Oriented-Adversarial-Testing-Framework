"""
Auto-run script to test the corrected experiment
"""
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main framework
from main import ReliabilityTestingFramework

def main():
    """Run the experiment automatically with Phase 1."""
    framework = ReliabilityTestingFramework()
    
    # Manually trigger Phase 1 execution
    print("Starting Phase 1 Quick Test...")
    try:
        framework._execute_experiment(tasks_per_dataset=3)
        print("\nExperiment completed successfully!")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()