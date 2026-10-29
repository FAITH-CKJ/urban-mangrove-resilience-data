
from pathlib import Path
import subprocess,sys
subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).with_name('reproduce_figure_story.py')),*sys.argv[1:],'--figures','6'],check=True)
