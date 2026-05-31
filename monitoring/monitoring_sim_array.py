import logging
import psutil
import shlex
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MEM_UNITS = ['B', 'KB', 'MB', 'GB', 'TB']


def monitor_application(pid, interval=5, duration=10):
    process = psutil.Process(pid)
    bytes = []
    start = time.time()
    while (time.time() - start) < duration:
        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()

            unit = 0
            mem = memory_info.rss
            bytes.append(mem)
            while mem > 1024:
                mem = mem / 1024
                unit += 1

            logger.info(f"PID {pid} - CPU: {cpu_percent}% - Memory: {mem:.2f} {MEM_UNITS[unit]}")

            time.sleep(interval)
        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} no longer exists")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            break
    print('KILLING PROCESS WITH PID =', pid)
    process.kill()
    return max(bytes) if bytes else 0


if __name__ == '__main__':
    command = 'python -m sim_no_render_array -x 10 -y 10 -z 10'
    args = shlex.split(command)
    print("\nLAUNCHING NEW PROCESS")
    subproc = subprocess.Popen(args)
    print('process pid =', subproc.pid)

    monitor_application(subproc.pid, interval=1, duration=5)
