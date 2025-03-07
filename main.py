import json
from datetime import datetime

import matplotlib.pyplot as plt
from numpy import arange
from pandas import DataFrame


def analyze_telegram_messages(json_file, user_name, hour_interval=6):
    """
    Анализирует экспорт чата Telegram и строит график сообщений пользователя по временным интервалам.

    Parameters:
    -----------
    json_file : str
        Путь к JSON-файлу экспорта чата Telegram
    user_name : str
        Имя пользователя, для которого строится анализ
    hour_interval : int, optional
        Интервал разбивки часов (по умолчанию 6 часов)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Объект рисунка с графиком
    """
    # Загрузка данных из JSON-файла
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Извлечение сообщений
    messages = data.get('messages', [])

    # Фильтрация сообщений по пользователю
    user_messages = [msg for msg in messages if msg.get('from') == user_name]

    # Преобразование временных меток
    timestamps = []
    for msg in user_messages:
        date_str = msg.get('date')
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                timestamps.append(dt)
            except ValueError:
                continue

    if not timestamps:
        raise ValueError(f'Сообщения от пользователя {user_name} не найдены')

    df = DataFrame({'datetime': timestamps})

    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    df['hour_interval'] = (df['hour'] // hour_interval) * hour_interval

    interval_labels = {}
    for i in range(0, 24, hour_interval):
        end_hour = (i + hour_interval - 1) % 24
        interval_labels[i] = f'{i:02d}-{end_hour:02d}'

    df['interval_label'] = df['hour_interval'].map(interval_labels)

    message_counts = (
        df.groupby(['date', 'hour_interval']).size().reset_index(name='count')
    )

    unique_dates = sorted(df['date'].unique())

    pivot_data = message_counts.pivot_table(
        index='date',
        columns='hour_interval',
        values='count',
        fill_value=0,
    ).reindex(unique_dates)

    pivot_data.columns = [interval_labels[col] for col in pivot_data.columns]

    fig, ax = plt.subplots(figsize=(12, 8))

    num_intervals = 24 // hour_interval
    num_dates = len(unique_dates)
    bar_width = 0.8 / num_intervals

    positions = arange(num_dates)

    for i, interval in enumerate(pivot_data.columns):
        offset = (i - num_intervals / 2 + 0.5) * bar_width
        ax.bar(
            positions + offset,
            pivot_data[interval],
            width=bar_width,
            label=interval,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [date.strftime('%Y-%m-%d') for date in unique_dates],
        rotation=45,
    )

    ax.set_xlabel('Дата')
    ax.set_ylabel('Количество сообщений')
    ax.set_title(
        f'Активность пользователя {user_name} '
        f'по интервалам времени ({hour_interval} ч)',
    )
    ax.legend(title='Интервал времени (ч)')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = analyze_telegram_messages('result.json', 'MRossa')
    plt.show()
