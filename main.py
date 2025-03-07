import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
from numpy import arange, ndarray


def load_telegram_data(json_file: str) -> dict[str, Any]:
    """Load Telegram data from a JSON file.

    Args:
        json_file (str): Path to the Telegram chat export JSON file.

    Returns:
        dict[str, Any]: Dictionary containing the parsed JSON data.
    """
    with open(json_file, encoding='utf-8') as file:
        return json.load(file)


def filter_user_messages(
    data: dict[str, Any],
    user_name: str,
) -> list[dict[str, Any]]:
    """Filter messages by the specified user.

    Args:
        data (dict[str, Any]): Telegram chat data dictionary.
        user_name (str): Name of the user to filter messages for.

    Returns:
        list[dict[str, Any]]: list of messages from the specified user.
    """
    messages = data.get('messages', [])
    return [msg for msg in messages if msg.get('from') == user_name]


def extract_timestamps(messages: list[dict[str, Any]]) -> list[datetime]:
    """Extract datetime objects from message timestamps.

    Args:
        messages (list[dict[str, Any]]): list of message dictionaries.

    Returns:
        list[datetime]: list of datetime objects extracted from messages.
    """
    timestamps = []
    for msg in messages:
        if date_str := msg.get('date'):
            try:
                dt = datetime.fromisoformat(date_str)
                timestamps.append(dt)
            except ValueError:
                continue

    if not timestamps:
        raise ValueError('Message not found or date format is incorrect')

    return timestamps


def create_interval_labels(hour_interval: int) -> dict[int, str]:
    """Create time interval labels based on the specified hour interval.

    Args:
        hour_interval (int): The interval in hours to group messages by.

    Returns:
        dict[int, str]: Dictionary mapping hour intervals to their labels.
    """
    interval_labels = {}
    for i in range(0, 24, hour_interval):
        end_hour = i + hour_interval - 1
        if end_hour >= 24:
            end_hour = 23
        interval_labels[i] = f'{i:02d}-{end_hour:02d}'

    # Special case for hour_interval=1
    if hour_interval == 1:
        interval_labels = {}
        for i in range(0, 24):
            interval_labels[i] = f'{i:02d}'

    return interval_labels


def aggregate_message_data(
    timestamps: list[datetime],
    hour_interval: int,
) -> tuple[dict, list, dict[int, str]]:
    """Aggregate message data by date and time interval without using pandas.

    Args:
        timestamps (list[datetime]): list of message datetime objects.
        hour_interval (int): The interval in hours to group messages by.

    Returns:
        tuple[dict, list, dict[int, str]]:
            A tuple containing the message counts by date and interval,
            list of unique dates, and interval labels.
    """
    message_counts = defaultdict(lambda: defaultdict(int))
    dates = set()

    for dt in timestamps:
        date = dt.date()
        hour = dt.hour
        hour_interval_val = (hour // hour_interval) * hour_interval

        message_counts[date][hour_interval_val] += 1
        dates.add(date)

    unique_dates = sorted(dates)
    interval_labels = create_interval_labels(hour_interval)

    return message_counts, unique_dates, interval_labels


def find_last_day_hour_intervals(
    timestamps: list[datetime],
    hour_interval: int,
) -> list[int]:
    """Find the available hour intervals in the last day of data.

    Args:
        timestamps (list[datetime]): list of message datetime objects.
        hour_interval (int): The interval in hours to group messages by.

    Returns:
        list[int]: list of hour intervals available in the last day.
    """
    if not timestamps:
        return list(range(0, 24, hour_interval))

    max_date = max(dt.date() for dt in timestamps)

    max_hour = 0
    for dt in timestamps:
        if dt.date() == max_date and dt.hour > max_hour:
            max_hour = dt.hour

    max_interval = (max_hour // hour_interval) * hour_interval

    intervals = list(range(0, max_interval + hour_interval, hour_interval))

    return intervals


def create_pivot_data(
    message_counts: dict,
    unique_dates: list,
    interval_labels: dict[int, str],
    hour_interval: int,
) -> tuple[dict, list]:
    """Create pivot data structure similar to pandas pivot table.

    Args:
        message_counts (dict): Dictionary containing message counts by date and
            interval.
        unique_dates (list): list of unique dates.
        interval_labels (dict[int, str]): Dictionary mapping hour intervals to
            their labels.
        hour_interval (int): The interval in hours to group messages by.

    Returns:
        tuple[dict, list]: A tuple containing the pivot data and a list of
            interval columns.
    """
    all_intervals = list(range(0, 24, hour_interval))
    interval_columns = [interval_labels[interval] for interval in all_intervals]

    pivot_data = {}
    for date in unique_dates:
        pivot_data[date] = {}
        for interval in all_intervals:
            pivot_data[date][interval_labels[interval]] = message_counts[date][
                interval
            ]

    return pivot_data, interval_columns


def add_value_labels(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    """Add value labels to the top of each bar.

    Args:
        ax (plt.Axes): The axes to add labels to.
        bars (list[plt.Rectangle]): list of bar plot rectangles.
    """
    for rect in bars:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + 0.5,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                rotation=90,
            )


def configure_chart_appearance(
    ax: plt.Axes,
    positions: ndarray,
    unique_dates: list[datetime],
    user_name: str,
    interval_columns: list,
    num_intervals: int,
    bar_width: float,
) -> None:
    """Configure the appearance of the chart.

    Args:
        ax (plt.Axes): The axes to configure.
        positions (np.ndarray): Array of x-positions for bars.
        unique_dates (list): list of unique dates.
        user_name (str): Name of the user being analyzed.
        interval_columns (list): list of interval column labels.
        num_intervals (int): Number of time intervals.
        bar_width (float): Width of each bar.
    """
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [date.strftime('%Y-%m-%d') for date in unique_dates],

    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of messages', fontsize=12)
    ax.set_title(
        f'Activity of user *{user_name}* by time intervals',
        fontsize=14,
    )

    bottom_offset = -0.7
    for i, interval in enumerate(interval_columns):
        offset = (i - num_intervals / 2 + 0.5) * bar_width
        for pos in positions:
            ax.text(
                pos + offset,
                bottom_offset,
                interval,
                ha='center',
                va='top',
                fontsize=8,
            )

    if len(positions) > 1:
        for i in range(1, len(positions)):
            ax.axvline(
                x=(positions[i - 1] + positions[i]) / 2,
                color='black',
                linestyle='-',
                linewidth=2,
                alpha=0.7,
            )

    if ax.get_legend():
        ax.get_legend().remove()

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(bottom=0.2)


def plot_message_chart(
    pivot_data: dict,
    interval_columns: list,
    unique_dates: list,
    user_name: str,
    hour_interval: int,
    color_theme: str = 'viridis',
) -> plt.Figure:
    """Create a bar chart visualization of message activity.

    Args:
        pivot_data (dict): Dictionary with pivot table data.
        interval_columns (list): list of interval column labels.
        unique_dates (list): list of unique dates.
        user_name (str): Name of the user being analyzed.
        hour_interval (int): The interval in hours messages are grouped by.
        color_theme (str, optional): Color theme for the chart.
            Defaults to 'viridis'.

    Returns:
        plt.Figure: Figure object containing the chart.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    num_intervals = 24 // hour_interval

    num_dates = len(unique_dates)
    bar_width = 0.8 / num_intervals

    positions = arange(num_dates)
    colors = plt.get_cmap(color_theme)

    # Find maximum value for color normalization
    max_value = 0
    for date in unique_dates:
        for interval in interval_columns:
            max_value = max(max_value, pivot_data[date].get(interval, 0))

    bars = []
    for i, interval in enumerate(interval_columns):
        offset = (i - num_intervals / 2 + 0.5) * bar_width

        column_bars = []
        for j, date in enumerate(unique_dates):
            value = pivot_data[date].get(interval, 0)

            # Normalize value to determine color intensity
            color_intensity = value / max_value if max_value > 0 else 0
            bar = ax.bar(
                positions[j] + offset,
                value,
                width=bar_width,
                color=colors(color_intensity),
                label=f'{interval}_{j}' if j == 0 else '',
            )
            column_bars.extend(bar)

        bars.append(column_bars)
        add_value_labels(ax, column_bars)

    configure_chart_appearance(
        ax,
        positions,
        unique_dates,
        user_name,
        interval_columns,
        num_intervals,
        bar_width,
    )

    plt.tight_layout()
    return fig


def analyze_telegram_messages(
    json_file: str,
    user_name: str,
    hour_interval: int = 6,
    color_theme: str = 'viridis',
) -> plt.Figure:
    """Analyze Telegram messages and create a visualization of user activity by
    time.

    Args:
        json_file (str): Path to the Telegram chat export JSON file.
        user_name (str): Name of the user to analyze messages for.
        hour_interval (int, optional):
            Time interval in hours to group messages by. Defaults to 6.
        color_theme (str, optional): Color theme for the visualization.
            Defaults to 'viridis'.

    Returns:
        plt.Figure: Figure object containing the visualization chart.
    """
    data = load_telegram_data(json_file)
    user_messages = filter_user_messages(data, user_name)
    timestamps = extract_timestamps(user_messages)

    message_counts, unique_dates, interval_labels = aggregate_message_data(
        timestamps,
        hour_interval,
    )

    pivot_data, interval_columns = create_pivot_data(
        message_counts,
        unique_dates,
        interval_labels,
        hour_interval,
    )

    fig = plot_message_chart(
        pivot_data,
        interval_columns,
        unique_dates,
        user_name,
        hour_interval,
        color_theme,
    )

    return fig


if __name__ == '__main__':
    fig = analyze_telegram_messages(
        'result.json',
        'MRossa',
        hour_interval=8,
        color_theme='ocean_r',
    )
    plt.show()
