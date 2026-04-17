try:
    import winsound
except ImportError:
    winsound = None
import threading
import time
from datetime import datetime
import os

class AlertSystem:
    def __init__(self, log_file="drowsiness_log.txt"):
        self.alarming = False
        self.log_file = log_file
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("=== Drowsiness Detection Log ===\n")

    def _play_alarm(self):
        """Plays a sound loop until alarming is set to False."""
        while self.alarming:
            # Using Windows default beep 
            if winsound:
                winsound.Beep(2500, 500) # 2500Hz frequency, 500ms duration
            else:
                # Fallback for non-windows (just print or log)
                pass
            time.sleep(0.1)

    def start_alarm(self):
        """Starts the alarm thread if not already running."""
        if not self.alarming:
            self.alarming = True
            t = threading.Thread(target=self._play_alarm)
            t.daemon = True
            t.start()
            self.log_event("SLEEP DETECTED - ALARM TRIGGERED")

    def stop_alarm(self):
        """Stops the alarm thread."""
        if self.alarming:
            self.alarming = False
            self.log_event("ALARM STOPPED - USER AWAKE")

    def log_event(self, event_type):
        """Logs an event with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {event_type}\n")
        print(f"Logged: [{timestamp}] {event_type}")
