"""
Calculate and graph various statistics about reddit comments.
"""

import datetime
import statistics

import numpy
import pandas

import plotly.express

from typing import Optional

from dataclasses import dataclass


@dataclass
class StatisticsCommentInfo:
    """
    This class represents a single statistical data point
    for the statistics_analyze method.

    Representation Invariants:
        - -1 <= self.sentiment <= 1

    Instance Attributes:
        - sentiment: The sentiment of the comment from -1 (negative) through +1 (positive).
        - date: The date the comment was posted.
    """

    sentiment: float
    date: datetime.datetime


@dataclass
class StatisticsPoint:
    """
    This class represents a raw data point on the graph. For internal use.

    Instance Attributes:
        - x: The x coordinate of the point.
        - y: The y coordinate of the point.
    """

    x: float
    y: float


@dataclass
class StatisticsNormalizeResult:
    """
    This class represents comment data after initial processing.
    Holds range values about the data and a list of raw points. For internal use.

    Instance Attributes:
        - min_x: The smallest x value in points. Usually 0.
        - max_x: The greatest x value in points. Usually 1.
        - min_y: The smallest y value in points.
        - max_y: The greatest y value in points.
        - points: A list of pairs of x, y values. To be analyzed or graphed.
        - start_date: The earliest date in the input data. For user text in graph_raw.
        - end_date: The latest date in the input data. For user text in graph_raw.

    Representation Invariants:
        - self.points != []
    """
    min_x: float
    max_x: float

    min_y: float
    max_y: float

    points: list[StatisticsPoint]

    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None


@dataclass
class StatisticsAnalysisResult:
    """
    This class represents the resulting statistical analysis data
    returned from the statistics_analyze method.

    Instance Attributes:
        - mean_x: Average x value.
        - median_x: Median x value.
        - mode_x: Mode x value (most frequently taken).
        - mean_y: Average y value.
        - median_y: Median y value.
        - mode_y: Mode y value (most frequently taken).
        - correlation: Pearson correlation coefficient. Higher magnitude is stronger.
        - fit: 2nd degree polynomial fit. In the form fit[0] x^2 + fit[1] x + fit[2].

    Representation Invariants:
        - -1 <= self.correlation <= +1
    """

    # We don't really care about these variables, but I will leave them here.
    mean_x: float
    median_x: float
    mode_x: float

    mean_y: float
    median_y: float
    mode_y: float

    correlation: float

    fit: tuple[float, float, float]


def statistics_normalize(comments: list[StatisticsCommentInfo],
                         start_time: datetime.datetime,
                         end_time: datetime.datetime) -> StatisticsNormalizeResult:
    """
    Performs initial processing on `comments`.
    Ignores any comments that are not within the `start_time`, `end_time` range.

    Preconditions:
        - comments != []
    """

    earliest_date = None
    latest_date = None

    for comment in comments:
        if not (start_time <= comment.date <= end_time):
            continue

        if earliest_date is None or earliest_date > comment.date:
            earliest_date = comment.date

        if latest_date is None or latest_date < comment.date:
            latest_date = comment.date

    min_x, max_x = None, None
    min_y, max_y = None, None

    points = []

    magnitude = (latest_date - earliest_date).total_seconds()

    for comment in comments:
        if not (start_time <= comment.date <= end_time):
            continue

        start_seconds = (comment.date - earliest_date).total_seconds()

        # Not sure how great this is ...
        # Normalizing x between 0 and 1.
        x = start_seconds / magnitude
        y = comment.sentiment

        if min_x is None or x < min_x:
            min_x = x
        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y
        if max_y is None or y > max_y:
            max_y = y

        points.append(StatisticsPoint(x=x, y=y))

    return StatisticsNormalizeResult(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,

        points=points,

        start_date=earliest_date,
        end_date=latest_date,
    )


def statistics_normalize_all(comments: list[StatisticsCommentInfo]) -> StatisticsNormalizeResult:
    """
    Convenience method for statistics_normalize. Processes `comments`.
    Does not filter any comments based on date.

    Preconditions:
        - comments != []
    """
    # I dont want to bother with Optional date time values so...
    start_time = datetime.datetime(year=1, month=1, day=1)
    end_time = datetime.datetime.now()

    # A better way would be to invoke this with a method that filters it out.
    # Too lazy...
    return statistics_normalize(comments, start_time, end_time)


def statistics_analyze_raw(normalized_data: StatisticsNormalizeResult) -> StatisticsAnalysisResult:
    """
    Performs analysis on the data points in `normalized_data`.
    `normalized_data` is typically a structure returned from statistics_normalize.
    """

    x_values = [point.x for point in normalized_data.points]
    y_values = [point.y for point in normalized_data.points]

    # I got lazy here and just invoked statistics instead of pandas.
    mean_x = statistics.mean(x_values)
    median_x = statistics.median(x_values)
    mode_x = statistics.mode(x_values)

    mean_y = statistics.mean(y_values)
    median_y = statistics.median(y_values)
    mode_y = statistics.mode(y_values)

    data_frame = pandas.DataFrame(data=normalized_data.points)
    correlation = data_frame.corr().loc['x']['y']

    fit = numpy.polyfit(data_frame['x'], data_frame['y'], 2)

    # Seems I can use python methods for this... A bit disappointing.
    return StatisticsAnalysisResult(
        mean_x=mean_x,
        median_x=median_x,
        mode_x=mode_x,

        mean_y=mean_y,
        median_y=median_y,
        mode_y=mode_y,

        fit=(float(fit[0]), float(fit[1]), float(fit[2])),
        correlation=correlation
    )


def statistics_analyze(comments: list[StatisticsCommentInfo],
                       start_time: datetime.datetime,
                       end_time: datetime.datetime) -> StatisticsAnalysisResult:
    """
    Convenience method to analyze comments in `comments` with a date posted between
    `start_time` and `end_time`.

    Returns various statistical information.

    Preconditions:
        - comments != []
    """

    return statistics_analyze_raw(statistics_normalize(comments, start_time, end_time))


def statistics_analyze_all(comments: list[StatisticsCommentInfo]) -> StatisticsAnalysisResult:
    """
    Convenience method to analyze all comments in `comments`.

    Returns various statistical information.

    Preconditions:
        - comments != []
    """

    return statistics_analyze_raw(statistics_normalize_all(comments))


def graph_raw(points: StatisticsNormalizeResult, title: Optional[str] = None) -> None:
    """
    Opens a new window with a graph for the points in `points`.
    `points` is typically a structure returned from statistics_normalize.

    The graph title will be `title` if specified, otherwise it will be generated in method.
    """
    analysis = statistics_analyze_raw(points)

    if title is None:
        title = f'Comment Sentiment Over Time'

    if points.start_date is not None and points.end_date is not None:
        correlation = '[r = {:.3f}]'.format(analysis.correlation)
        title += f', From {str(points.start_date.date())} ' \
                 f'Through {str(points.end_date.date())} {correlation}'

    fit_x = []
    fit_y = []

    resolution = 1000

    for i in range(resolution + 1):
        x = 1.0 / resolution * i
        # hmm... order of fit here?
        y = x * x * analysis.fit[0] + x * analysis.fit[1] + analysis.fit[2]

        fit_x.append(x)
        fit_y.append(y)

    figure = plotly.express.line(
        title=title,
        x=[point.x for point in points.points],
        y=[point.y for point in points.points]
    )

    fit_name = "{:.1f}x^2 + {:.1f}x + {:.1f}"\
        .format(analysis.fit[0], analysis.fit[1], analysis.fit[2])

    figure.add_scatter(x=fit_x, y=fit_y, name=fit_name)

    figure.show()


def graph(comments: list[StatisticsCommentInfo],
          start_time: datetime.datetime,
          end_time: datetime.datetime,
          title: Optional[str] = None) -> None:
    """
    Convenience method for graph.
    Opens a new window with a graph graphing all comments in `comments`.
    Comments that do not fall in the start_time, end_time range will not be graphed.

    Representation Invariants:
        - comments != []
    """

    return graph_raw(statistics_normalize(comments, start_time, end_time), title)


def graph_all(comments: list[StatisticsCommentInfo], title: Optional[str] = None) -> None:
    """
    Convenience method for graph.
    Opens a new window with a graph graphing all comments in `comments`.

    Representation Invariants:
        - comments != []
    """

    return graph_raw(statistics_normalize_all(comments), title)


def process_average_comments(comments: list[StatisticsCommentInfo]) -> list[StatisticsCommentInfo]:
    """
    AA
    """

    normalized = statistics_normalize_all(comments)

    assert normalized.start_date is not None
    assert normalized.end_date is not None

    start = normalized.start_date
    end = normalized.end_date

    days_count = (end - start).days + 1

    # array [total_sentiment, total_days]
    days = []

    # Can't do [[0, 0]] * days_count, would share reference of arrays...
    for _ in range(days_count):
        days.append([0, 0])

    for comment in comments:
        day = (comment.date - start).days

        days[day][0] += comment.sentiment
        days[day][1] += 1

    result = []

    for i in range(days_count):
        day = start + datetime.timedelta(days=i)
        sentiment = 0

        if days[i][1] != 0:
            sentiment = days[i][0] / days[i][1]

        result.append(StatisticsCommentInfo(date=day, sentiment=sentiment))

    return result

if __name__ == '__main__':
    data = [
        StatisticsCommentInfo(sentiment=-0.4, date=datetime.datetime(year=2020, month=1, day=1)),
        StatisticsCommentInfo(sentiment=-0.3, date=datetime.datetime(year=2020, month=2, day=1)),
        StatisticsCommentInfo(sentiment=-0.2, date=datetime.datetime(year=2020, month=3, day=1)),
        StatisticsCommentInfo(sentiment=-0.1, date=datetime.datetime(year=2020, month=4, day=1)),
        StatisticsCommentInfo(sentiment=-0.0, date=datetime.datetime(year=2020, month=5, day=1)),
        StatisticsCommentInfo(sentiment=+0.1, date=datetime.datetime(year=2020, month=6, day=1)),
        StatisticsCommentInfo(sentiment=+0.2, date=datetime.datetime(year=2020, month=7, day=1)),
        StatisticsCommentInfo(sentiment=+0.3, date=datetime.datetime(year=2020, month=8, day=1)),
        StatisticsCommentInfo(sentiment=+0.4, date=datetime.datetime(year=2021, month=10, day=1)),
        StatisticsCommentInfo(sentiment=+0.5, date=datetime.datetime(year=2021, month=9, day=1)),
        StatisticsCommentInfo(sentiment=+0.6, date=datetime.datetime(year=2021, month=8, day=1)),
        StatisticsCommentInfo(sentiment=+0.7, date=datetime.datetime(year=2021, month=7, day=1)),
        StatisticsCommentInfo(sentiment=+0.8, date=datetime.datetime(year=2021, month=6, day=1)),
        StatisticsCommentInfo(sentiment=+0.89, date=datetime.datetime(year=2021, month=5, day=1)),
        StatisticsCommentInfo(sentiment=+0.9, date=datetime.datetime(year=2021, month=4, day=1)),
        StatisticsCommentInfo(sentiment=+1.0, date=datetime.datetime(year=2021, month=3, day=1)),
    ]

    print("====== ALL COMMENTS ======")
    print(statistics_analyze_all(data))

    print("====== BEFORE COVID ======")
    print(statistics_analyze(data,
                             start_time=datetime.datetime(year=2019, month=1, day=1),
                             end_time=datetime.datetime(year=2020, month=3, day=1)))

    print("====== AFTER COVID ======")
    print(statistics_analyze(data,
                             start_time=datetime.datetime(year=2020, month=3, day=1),
                             end_time=datetime.datetime.now()))

    print("====== DURING 2021 ======")
    print(statistics_analyze(data,
                             start_time=datetime.datetime(year=2021, month=1, day=1),
                             end_time=datetime.datetime.now()))

    # graph(statistics_normalize(data,
    #                            start_time=datetime.datetime(year=2019, month=1, day=1),
    #                            end_time=datetime.datetime(year=2020, month=3, day=1)))

    graph_all(data)
