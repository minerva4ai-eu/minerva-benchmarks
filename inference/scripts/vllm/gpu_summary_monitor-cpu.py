import signal
import time
import sys

poll_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

stop = False

def stop_monitor(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, stop_monitor)
signal.signal(signal.SIGTERM, stop_monitor)


try:
    while not stop:
        # Your monitoring code here
        time.sleep(poll_interval)
finally:
    print("Ending 'monitor'")