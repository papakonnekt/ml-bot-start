```python
import csv
import time
import pyautogui

def CollectMouseMovementData(file_path):
    """
    Collects a large dataset of human mouse movement data while playing OSRS. 
    It records mouse coordinates at regular intervals and captures mouse click events. 
    The collected data is stored in CSV format, including timestamp, x-coordinate, y-coordinate, and click events.
    """
    with open(file_path, mode='w', newline='') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['timestamp', 'x-coordinate', 'y-coordinate', 'click event'])
        while True:
            x, y = pyautogui.position()
            click = pyautogui.mouseDown(button='left')
            data_writer.writerow([time.time(), x, y, click])
            time.sleep(0.1)
```