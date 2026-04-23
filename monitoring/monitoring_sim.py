import psutil, shlex, subprocess
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)









MEM_UNITS = [
    'B',
    'KB',
    'MB',
    'GB',
    'TB'
]

def monitor_application(pid, interval=5, duration = 10):
    """Monitor a Python process and log its resource usage."""
    process = psutil.Process(pid)
    start = time.time()
    while (time.time() - start) < duration:
        try:
            # Get CPU and memory usage
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            
            unit=0
            mem = memory_info.rss
            while mem > 1024:
                mem = mem/1024
                unit+=1
            
            
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

# Example usage: monitor_application(your_app_pid)

if __name__=='__main__':
    
    sizes = [
        (1,1,1),
        (10,10,10),
        (100,100,15),
        (100,100,100),
        (1000,1000,100)
    ]
    
    for size in [sizes[0]]:
    
        command = f'python -m sim_no_render -x {size[0]} -y {size[1]} -z {size[2]}'
        args = shlex.split(command)
        print("\nLAUNCHING NEW PROCESS")
        print("grid size =",size)
        subproc = subprocess.Popen(args)
        print('process pid =', subproc.pid)
        
        monitor_application(subproc.pid, interval=3,duration=30)