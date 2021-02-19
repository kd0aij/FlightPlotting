
from flightdata import Flight
from io import StringIO, BytesIO
from os.path import splitext
from flightanalysis import Section, State, FlightLine


class PlotDat():
    def __init__(self):
        self.fl = None
        self.sq = None


def read_log(file, contents):
    filename, file_extension = splitext(file)
    flight = read_csv(contents) if file_extension == '.csv' else read_bin(contents)
    return flight   


def generate_section(flight, meth=0):
    if meth == 0:
        return Section.from_flight(flight, FlightLine.from_covariance(flight))
    elif meth == 1:
        return Section.from_flight(flight, FlightLine.from_initial_position(flight))


def read_csv(contents):
    return Flight.from_csv(BytesIO(contents))


def read_bin(contents):
    with open("temp.BIN", "wb") as fp:
        fp.write(contents)
    return Flight.from_log("temp.BIN")
    

