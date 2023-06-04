```python
import pandas as pd
import numpy as np

def LoadMouseMovementData(file_path):
    """
    Loads the CSV mouse movement data into the bot program.
    """
    data = pd.read_csv(file_path)
    return data

def NormalizeMouseCoordinates(data, screen_resolution):
    """
    Normalizes the mouse coordinates to ensure consistent scaling across different screen resolutions.
    """
    data['x-coordinate'] = data['x-coordinate'] / screen_resolution[0]
    data['y-coordinate'] = data['y-coordinate'] / screen_resolution[1]
    return data

def ConvertClickEvents(data):
    """
    Converts the mouse click events into binary values (e.g., 0 for no click, 1 for a click).
    """
    data['click-event'] = np.where(data['click-event'] == 'click', 1, 0)
    return data
```