# Telegram User Activity Analyzer

A Python tool for analyzing and visualizing user activity patterns from Telegram chat exports.

## Overview

This tool analyzes JSON exports from Telegram chats to generate visualizations of messaging activity patterns for specific users. It breaks down message frequency by day and time intervals, allowing you to identify when users are most active.

## Features

- Filter messages by specific username
- Group messages into customizable time intervals (default: 6-hour blocks)
- Generate bar charts showing message distribution across days and time periods

## Requirements

- Python 3.10+
- Dependencies:
  - pandas
  - matplotlib
  - numpy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MRossa157/tg-user-activity-analyzer.git
   cd tg-user-activity-analyzer
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```


## Usage

### Step 1: Export Telegram Chat History

1. Open Telegram desktop client
2. Select the chat you want to analyze
3. Click on the three dots menu (⋮) in the top-right corner
4. Select "Export chat history"
5. In the export settings:
   - Set format to "JSON"
   - You can deselect media files to make the export smaller
   - Click "Export"

### Step 2: Run the Analysis

Import the module and call the function:

```python
from main import analyze_telegram_messages
import matplotlib.pyplot as plt

# Analyze messages from a specific user
fig = analyze_telegram_messages(
    json_file='path/to/your/telegram_export.json',
    user_name='UserNameToAnalyze'
)

# Display the chart
plt.show()

# Or save the chart
fig.savefig('activity_analysis.png', dpi=300)
```

### Customizing Time Intervals

You can change the time interval grouping (default is 6 hours):

```python
# For 3-hour intervals (8 bars per day)
fig = analyze_telegram_messages(
    json_file='path/to/your/telegram_export.json',
    user_name='UserNameToAnalyze',
    hour_interval=3
)
```

## Example Output

The generated chart shows:
- X-axis: Dates
- Y-axis: Message count
- Colored bars: Different time intervals throughout the day
- Legend: Time interval mappings

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.