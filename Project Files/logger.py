import csv
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from pathlib import Path

    


class logger():

    def __init__(self):
        self.startTime = datetime.now().strftime("%H-%M-%S")
        self.logs = pd.DataFrame([(datetime.now().strftime("%H:%M:%S"), "Status", "Logging started")])

    def recordLog(self, type, message):
        now = datetime.now()
        log = pd.DataFrame([(now.strftime("%H:%M:%S"), type, message)])
        self.logs.append(log)


    
    def save(self):
        filepath = Path('Logs/MusselDetection' + self.startTime + '.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.logs.to_csv(filepath, index=False)

