import sys
from pathlib import Path

# 저장소 최상위(postprocess 패키지)를 import 경로에 넣는다
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
