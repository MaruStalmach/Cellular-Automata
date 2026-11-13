# Profiling tools

Run everything from the repository root.

## Resource monitoring

`monitoring_sim.py` launches `examples/sim_no_render.py` for a range of grid sizes and records CPU and RAM usage.

```bash
python tools/profiling/monitoring_sim.py
```

## Function-level profiling

1. Profile a script with cProfile:

   ```bash
   python -m cProfile -o output_files/out.profile examples/sim_no_render.py -x 20 -y 20 -z 20 -s 100
   ```

2. Print the 20 most expensive functions (defaults to `output_files/out.profile`):

   ```bash
   python tools/profiling/profiling.py [path/to/out.profile]
   ```

3. Optionally visualise it with `snakeviz output_files/out.profile`.
