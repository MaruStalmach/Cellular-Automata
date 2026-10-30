import pstats
from pathlib import Path

p = pstats.Stats(Path(__file__).with_name("out.profile"))
p.strip_dirs().sort_stats("cumtime").print_stats(20)
