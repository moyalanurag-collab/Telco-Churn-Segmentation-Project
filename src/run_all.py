"""Run the full pipeline in order."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
for script in ["step1_eda.py", "step2_segmentation.py", "step2b_segment_profiles.py",
               "step3_model.py", "step4_export.py", "step5_report.py", "step6_dashboard.py"]:
    print(f"\n>>> {script}")
    subprocess.run([sys.executable, str(here / script)], cwd=here, check=True)
print("\nDone. See outputs/ and report/.")
