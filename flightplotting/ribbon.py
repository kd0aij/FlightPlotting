
from flightanalysis import Section
from geometry import Point, Points
import numpy as np



def vec_ribbon(sec: Section, span: float):
    """Vectorised version of ribbon.
        minor mod - 2 triangles per pair of points:  
            current pair to next left
            current right to next pair
    """

    left = sec.body_to_world(Point(0, span/2, 0))
    right = sec.body_to_world(Point(0, -span/2, 0))

    points = np.empty((2*left.count), dtype=left.dtype)
    points[0::2] = left
    points[1::2] = right

    