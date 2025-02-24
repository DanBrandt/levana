# This module contains miscellaneous helper functions for general use.

#### Top-level imports ####
import numpy as np
import pandas as pd
from datetime import datetime

##### Helper Functions ####
def linspace_datetime(dateStart, dateEnd):
    """
    Make an array of datetimes between two dates, inclusive. Assumes a cadence of 1 day.
    :param dateStart: str
        Starting date in YYYY-MM-DD.
    :param dateEnd: str
        Ending date in YYYY-MM-DD.
    :return dateTimeArray: numpy.ndarray
        An array of datetimes.
    """
    start = pd.Timestamp(dateStart)
    end = pd.Timestamp(dateEnd)
    t = np.linspace(start.value, end.value, (end-start).days+1)
    t = pd.to_datetime(t)
    t_dt = np.asarray([element.to_pydatetime() for element in t])
    return t_dt

def dayNumber(datetime_val):
    """
    Given a datetime object, return the day-of-the-year in the format 'DOY'.
    :param datetime_val: datetime
        An arbitrary datetime object.
    :return doyStr: str
        The day-of-the-year in 'DOY' format (i.e. '001' or '075' or '365').
    """
    day_of_year = datetime_val.timetuple().tm_yday
    if len(str(day_of_year)) == 1:
        doyStr = '00'+str(day_of_year)
    elif len(str(day_of_year)) == 2:
        doyStr = '0'+str(day_of_year)
    else:
        doyStr = str(day_of_year)
    return doyStr
