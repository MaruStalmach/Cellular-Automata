import pstats

p = pstats.Stats("monitoring/out.profile")
p.strip_dirs().sort_stats("cumtime").print_stats(20)