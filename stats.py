"""
Calculate and graph various statistics about reddit comments.
Copyright Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""

import datetime
import statistics

from typing import Optional
from dataclasses import dataclass

import numpy
import pandas

import plotly.express


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

    >>> david = StatisticsCommentInfo(0, datetime.datetime.now())
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

    >>> StatisticsPoint(0.3, -0.2)
    """

    x: float
    y: float


@dataclass
class StatisticsNormalizeResult:
    """
    This class represents comment data after initial processing.
    Holds range values about the data and a list of raw points. For internal use.

    Instance Attributes:
        - min_y: The smallest y value in points.
        - max_y: The greatest y value in points.
        - points: A list of pairs of x, y values. To be analyzed or graphed.
        - line_graph: Whether or not the data should be displayed as a line graph.
        - start_date: The earliest date in the input data. For user text in graph_raw.
        - end_date: The latest date in the input data. For user text in graph_raw.

    Representation Invariants:
        - self.points != []

    >>> StatisticsNormalizeResult(0.4, 0.8, [StatisticsPoint(0, 0.4), StatisticsPoint(1, 0.8)])
    """

    min_y: float
    max_y: float

    points: list[StatisticsPoint]

    line_graph: bool = False

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
        - correlation: Pearson correlation coefficient. Higher magnitude is stronger.
        - fit: 2nd degree polynomial fit. In the form fit[0] x^2 + fit[1] x + fit[2].

    Representation Invariants:
        - -1 <= self.correlation <= +1

    >>> StatisticsAnalysisResult(0.5, 0.5, 0.5, 0.1, 0.1, 1.0, (1.0, 0.0, 2.0))
    """

    # We don't really care about these variables, but I will leave them here.
    mean_x: float
    median_x: float
    mode_x: float

    mean_y: float
    median_y: float

    correlation: float

    fit: tuple[float, float, float]


def statistics_normalize(comments: list[StatisticsCommentInfo]) -> StatisticsNormalizeResult:
    """
    Performs initial processing on `comments`.

    Preconditions:
        - comments != []

    >>> comments = [
    ...     StatisticsCommentInfo(-0.4, datetime.datetime(year=2020, month=1, day=1)),
    ...     StatisticsCommentInfo(-0.3, datetime.datetime(year=2020, month=2, day=1))
    ... ]
    >>> expected = StatisticsNormalizeResult(
    ...     min_y=-0.4, max_y=-0.3,
    ...     points=[StatisticsPoint(x=0.0, y=-0.4), StatisticsPoint(x=1.0, y=-0.3)],
    ...     line_graph=False,
    ...     start_date=datetime.datetime(2020, 1, 1, 0, 0),
    ...     end_date=datetime.datetime(2020, 2, 1, 0, 0)
    ... )
    >>> statistics_normalize(comments) == expected
    True
    """

    earliest_date = None
    latest_date = None

    for comment in comments:
        if earliest_date is None or earliest_date > comment.date:
            earliest_date = comment.date

        if latest_date is None or latest_date < comment.date:
            latest_date = comment.date

    min_y, max_y = None, None

    points = []

    magnitude = (latest_date - earliest_date).total_seconds()

    for comment in comments:
        start_seconds = (comment.date - earliest_date).total_seconds()

        # Not sure how great this is ...
        # Normalizing x between 0 and 1.
        x = start_seconds / magnitude
        y = comment.sentiment

        if min_y is None or y < min_y:
            min_y = y
        if max_y is None or y > max_y:
            max_y = y

        points.append(StatisticsPoint(x=x, y=y))

    return StatisticsNormalizeResult(
        min_y=min_y,
        max_y=max_y,

        points=points,

        start_date=earliest_date,
        end_date=latest_date,
    )


def statistics_analyze_raw(normalized_data: StatisticsNormalizeResult) -> StatisticsAnalysisResult:
    """
    Performs analysis on the data points in `normalized_data`.
    `normalized_data` is typically a structure returned from statistics_normalize.

    >>> data = StatisticsNormalizeResult(
    ...     min_y=-0.4, max_y=-0.3,
    ...     points=[
    ...         StatisticsPoint(x=0.0, y=-0.4), StatisticsPoint(x=1.0, y=-0.3),
    ...         StatisticsPoint(x=0.5, y=-0.2), StatisticsPoint(x=0.6, y=-0.1)
    ...     ],
    ...     line_graph=False,
    ...     start_date=datetime.datetime(2020, 1, 1, 0, 0),
    ...     end_date=datetime.datetime(2020, 2, 1, 0, 0)
    ... )
    >>> analysis = statistics_analyze_raw(data)
    >>> import math
    >>> math.isclose(analysis.mean_x, 0.525) and math.isclose(analysis.median_x, 0.55)
    True
    >>> math.isclose(analysis.mean_y, -0.25) and math.isclose(analysis.median_y, -0.25)
    True
    """

    x_values = [point.x for point in normalized_data.points]
    y_values = [point.y for point in normalized_data.points]

    # I got lazy here and just invoked statistics instead of pandas.
    mean_x = statistics.mean(x_values)
    median_x = statistics.median(x_values)
    mode_x = statistics.mode(x_values)

    mean_y = statistics.mean(y_values)
    median_y = statistics.median(y_values)

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

        fit=(float(fit[0]), float(fit[1]), float(fit[2])),
        correlation=correlation
    )


def statistics_analyze(comments: list[StatisticsCommentInfo]) -> StatisticsAnalysisResult:
    """
    Convenience method to analyze all comments in `comments`.

    Returns various statistical information.

    Preconditions:
        - comments != []

    >>> comments = [
    ...     StatisticsCommentInfo(1.0, datetime.datetime(2011, 1, 1)),
    ...     StatisticsCommentInfo(0.6, datetime.datetime(2011, 1, 2)),
    ...     StatisticsCommentInfo(0.0, datetime.datetime(2011, 1, 1)),
    ...     StatisticsCommentInfo(1.0, datetime.datetime(2011, 1, 3)),
    ... ]
    >>> analysis = statistics_analyze(comments)
    >>> import math
    >>> math.isclose(analysis.mean_x, 0.375) and math.isclose(analysis.median_x, 0.25)
    True
    >>> math.isclose(analysis.mean_y, 0.65) and math.isclose(analysis.median_y, 0.8)
    True
    """

    return statistics_analyze_raw(statistics_normalize(comments))


def graph_raw(points: StatisticsNormalizeResult, title: Optional[str] = None) -> None:
    """
    Opens a new window with a graph for the points in `points`.
    `points` is typically a structure returned from statistics_normalize.

    The graph title will be `title` if specified, otherwise it will be generated in method.

    This function is not pure.
    """
    analysis = statistics_analyze_raw(points)

    if title is None:
        title = 'Comment Sentiment Over Time'

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

    if points.line_graph:
        figure = plotly.express.line(
            title=title,
            x=[point.x for point in points.points],
            y=[point.y for point in points.points]
        )
    else:
        figure = plotly.express.scatter(
            title=title,
            x=[point.x for point in points.points],
            y=[point.y for point in points.points]
        )

    fit_name = "{:.1f}x^2 + {:.1f}x + {:.1f}"\
        .format(analysis.fit[0], analysis.fit[1], analysis.fit[2])

    figure.add_scatter(x=fit_x, y=fit_y, name=fit_name)

    figure.show()


def graph(comments: list[StatisticsCommentInfo], title: Optional[str] = None) -> None:
    """
    Convenience method for graph.
    Opens a new window with a graph graphing all comments in `comments`.

    This function is not pure.

    Representation Invariants:
        - comments != []
    """

    return graph_raw(statistics_normalize(comments), title)


def filter_comments(comments: list[StatisticsCommentInfo],
                    start_time: datetime.datetime,
                    end_time: datetime.datetime) -> list[StatisticsCommentInfo]:
    """
    Returns a new list from `comments` containing all comments
    that occur between `start_time` and `end_time` inclusive.

    >>> comments = [
    ...     StatisticsCommentInfo(1.0, datetime.datetime(2011, 1, 1)),
    ...     StatisticsCommentInfo(0.6, datetime.datetime(2011, 1, 2)),
    ...     StatisticsCommentInfo(0.0, datetime.datetime(2011, 1, 1))
    ... ]
    >>> expected = [
    ...     StatisticsCommentInfo(0.6, datetime.datetime(2011, 1, 2))
    ... ]
    >>> filter_comments(comments, datetime.datetime(2011, 1, 2), datetime.datetime(2011, 1, 2)) == expected
    True
    """

    return [
        comment
        for comment in comments
        if start_time <= comment.date <= end_time
    ]


def process_average_comments(comments: list[StatisticsCommentInfo]) -> StatisticsNormalizeResult:
    """
    Takes a list of `comments` and returns a StatisticsNormalizeResult
    containing one data point for each day, where the y value is the
    average sentiment of all comments on that day.

    Preconditions:
        - comments != []

    >>> comments = [
    ...     StatisticsCommentInfo(1.0, datetime.datetime(2011, 1, 1)),
    ...     StatisticsCommentInfo(0.6, datetime.datetime(2011, 1, 2)),
    ...     StatisticsCommentInfo(0.0, datetime.datetime(2011, 1, 1))
    ... ]
    >>> normalized = process_average_comments(comments)
    >>> normalized.points[0] == StatisticsPoint(0.0, 0.5)
    True
    >>> normalized.points[1] == StatisticsPoint(1.0, 0.6)
    True
    """

    normalized = statistics_normalize(comments)

    assert normalized.start_date is not None
    assert normalized.end_date is not None

    start = normalized.start_date
    end = normalized.end_date

    days_count = (end - start).days + 1

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

    normalized = statistics_normalize(result)
    normalized.line_graph = True  # since x values are unique...

    return normalized


if __name__ == '__main__':
    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'numpy', 'pandas', 'plotly.express',
            'math', 'datetime', 'statistics', 'doctest', 'python_ta'
        ],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
