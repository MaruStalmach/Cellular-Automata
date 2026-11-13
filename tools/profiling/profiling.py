import pstats
import sys

p = pstats.Stats(sys.argv[1] if len(sys.argv) > 1 else "output_files/out.profile")
p.strip_dirs().sort_stats("cumtime").print_stats(20)
