import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import get_sun, AltAz, EarthLocation
from astropy.time import Time
import numpy as np
from astropy import units as u

def getSunPos():
    lat = 47.3721678
    lon = 8.5259803
    time = Time("2022-10-25 14:51:00.000", format='iso') - 2 * u.hour

    loc = EarthLocation(lon=lon*u.deg, lat=lat*u.deg)
    altaz = AltAz(obstime=time, location=loc)
    
    sunPos = get_sun(time).transform_to(altaz)

    return time, sunPos.az.degree, sunPos.alt.degree

print(getSunPos())
