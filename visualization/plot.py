import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sgp4.io import twoline2rv
import matplotlib.pyplot as plt
from sgp4.earth_gravity import wgs72

def get_single_coordinate(line1, line2, year, month, day, hour, minute, second):
    """ returns sgp4 coordinate for TLE lines and time """

    satellite = twoline2rv(line1, line2, wgs72)
    position, velocity = satellite.propagate(year, month, day, hour, minute, second)

    return position

def get_coordinates(line1, line2, timepoints):
    """ returns sgp4 coordinates for TLE lines and list of times
        as tuple (year, month, day, hour, minute, second) """

    coordinates = []
    for (year, month, day, hour, minute, second) in timepoints:

        position = get_single_coordinate(line1, line2, year, month, day, hour, minute, second)

        coordinates.append(position)

    return coordinates

def generate_timepoints_for_month(year, month, from_day, until_day, x_minutes = 2):
    """ generate list of timepoints for a maximum of one month from from_day to
        until_day and samples one timepoint every x_minutes """
    days = 30
    if month in [1,3,5,7,8,10,12]:
        days = 31
    elif month == 2:
        days = 28

    return [(year, month, d, h, m, 0) for d in range(from_day, min(days, until_day)) for h in range(24) for m in range(0,60,x_minutes)]


def plot_single_orbit_2d(line1, line2, year, month, color, from_day = 0, until_day = 31):
    """ plots one songle orbit using only the x-coordinate and y-coordinate
        for the given timeframe """

    timepoints = generate_timepoints_for_month(year, month, from_day, until_day, x_minutes = 2)
    coordinates = get_coordinates(line1, line2, timepoints)

    x = [coordinates[i][0] for i in range(len(coordinates))]
    y = [coordinates[i][1] for i in range(len(coordinates))]
    z = [coordinates[i][2] for i in range(len(coordinates))]

    #cmap = cm.binary
    for i in range(len(coordinates)-1):
        plt.plot([x[i],x[i+1]], [y[i],y[i+1]], color=color)

def plot_orbits_2d(tle_list, year, month, from_day = 0, until_day = 31):
    """ plots a list of orbits in tle-format using only the x-coordinate and y-coordinate
        tle_list should be a tuple of strings (line1, line2) for the sgp4 propagator """

    fig = plt.figure(dpi=600)

    ax = plt.axes()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

    cmap = cm.binary
    norm = Normalize(vmin=0, vmax=len(tle_list))

    for i, (line1, line2) in enumerate(tle_list):
        color = cmap(norm(i))
        plot_single_orbit_2d(line1, line2, year, month, color, from_day, until_day)
