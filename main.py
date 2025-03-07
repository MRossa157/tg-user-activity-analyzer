import json
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
from numpy import arange, ndarray
from pandas import DataFrame, pivot_table


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
        date_str = msg.get('date')
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                timestamps.append(dt)
            except ValueError:
                continue

    if not timestamps:
        raise ValueError('Сообщения не найдены или формат даты некорректен')

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


def create_message_dataframe(
    timestamps: list[datetime],
    hour_interval: int,
) -> tuple[DataFrame, dict[int, str]]:
    """Create a DataFrame from timestamps and add time interval information.

    Args:
        timestamps (list[datetime]): list of message datetime objects.
        hour_interval (int): The interval in hours to group messages by.

    Returns:
        tuple[DataFrame, dict[int, str]]:
            A tuple containing the message DataFrame and interval labels.
    """
    df = DataFrame({'datetime': timestamps})
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['hour_interval'] = (df['hour'] // hour_interval) * hour_interval

    interval_labels = create_interval_labels(hour_interval)
    df['interval_label'] = df['hour_interval'].map(interval_labels)

    return df, interval_labels


def aggregate_message_data(
    df: DataFrame,
    interval_labels: dict[int, str],
    hour_interval: int,
) -> DataFrame:
    """Aggregate message data by date and time interval.

    Args:
        df (DataFrame): DataFrame containing message data.
        interval_labels (dict[int, str]):
            Dictionary mapping hour intervals to their labels.
        hour_interval (int): The interval in hours to group messages by.

    Returns:
        DataFrame: A pivot table with aggregated message counts.
    """
    message_counts = (
        df.groupby(['date', 'hour_interval']).size().reset_index(name='count')
    )
    unique_dates = sorted(df['date'].unique())

    pivot_data = pivot_table(
        message_counts,
        index='date',
        columns='hour_interval',
        values='count',
        fill_value=0,
    ).reindex(unique_dates)

    try:
        pivot_data.columns = [interval_labels[col] for col in pivot_data.columns]
    except KeyError:
        for col in pivot_data.columns:
            if col not in interval_labels:
                if hour_interval == 1:
                    interval_labels[col] = f'{col:02d}'
                else:
                    end_hour = col + hour_interval - 1
                    if end_hour >= 24:
                        end_hour = 23
                    interval_labels[col] = f'{col:02d}-{end_hour:02d}'
        pivot_data.columns = [interval_labels[col] for col in pivot_data.columns]

    return pivot_data


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
    pivot_data: DataFrame,
    user_name: str,
    num_intervals: int,
    bar_width: float,
) -> None:
    """Configure the appearance of the chart.

    Args:
        ax (plt.Axes): The axes to configure.
        positions (np.ndarray): Array of x-positions for bars.
        pivot_data (DataFrame): DataFrame with pivot table data.
        user_name (str): Name of the user being analyzed.
        num_intervals (int): Number of time intervals.
        bar_width (float): Width of each bar.
    """
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [date.strftime('%Y-%m-%d') for date in pivot_data.index],
        rotation=45,
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of messages', fontsize=12)
    ax.set_title(
        f'Activity of user {user_name} by time intervals',
        fontsize=14,
        fontweight='bold',
    )

    bottom_offset = -3
    for i, interval in enumerate(pivot_data.columns):
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

    if ax.get_legend():
        ax.get_legend().remove()

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(bottom=0.2)


def plot_message_chart(
    pivot_data: DataFrame,
    user_name: str,
    hour_interval: int,
    color_theme: str = 'viridis',
) -> plt.Figure:
    """Create a bar chart visualization of message activity.

    Args:
        pivot_data (DataFrame): DataFrame with pivot table data.
        user_name (str): Name of the user being analyzed.
        hour_interval (int): The interval in hours messages are grouped by.
        color_theme (str, optional): Color theme for the chart.
            Defaults to 'viridis'.

    Returns:
        plt.Figure: Figure object containing the chart.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    num_intervals = 24 // hour_interval
    num_dates = len(pivot_data.index)
    bar_width = 0.8 / num_intervals

    positions = arange(num_dates)
    colors = plt.get_cmap(color_theme)

    # Find maximum value for color normalization
    max_value = pivot_data.max().max()

    bars = []
    for i, interval in enumerate(pivot_data.columns):
        offset = (i - num_intervals / 2 + 0.5) * bar_width

        column_values = pivot_data[interval].values
        column_bars = []

        for j, value in enumerate(column_values):
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
        pivot_data,
        user_name,
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

    df, interval_labels = create_message_dataframe(timestamps, hour_interval)
    pivot_data = aggregate_message_data(df, interval_labels, hour_interval)

    fig = plot_message_chart(pivot_data, user_name, hour_interval, color_theme)

    return fig


if __name__ == '__main__':
    fig = analyze_telegram_messages(
        'result.json',
        'MRossa',
        hour_interval=1,
        color_theme='ocean',
    )
    plt.show()
