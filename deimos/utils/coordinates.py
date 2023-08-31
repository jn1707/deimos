#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoordTransform: A class for coordinate transformations and path length calculations used for sidereal LIV.

Created on Tue Jul 18 13:35:29 2023
Author: janni
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle, ICRS, get_sun
from astropy.time import Time

class CoordTransform(object):
    """
    CoordTransform class provides methods for coordinate transformations and path length calculations used for sidereal LIV.

    Parameters:
        detector_lat (str or float): Latitude of the detector in degrees or degrees-minutes-seconds format.
        detector_long (str or float): Longitude of the detector in degrees or degrees-minutes-seconds format.
        detector_height_m (float): Height of the detector above sea level in meters.
    """

    def __init__(self, detector_lat, detector_long, detector_height_m):
        # Convert latitude and longitude to degrees
        if isinstance(detector_lat, str):
            detector_lat = Angle(detector_lat)
            detector_lat = detector_lat.deg
        # latitude in range [-90,90]

        if isinstance(detector_long, str):
            detector_long = Angle(detector_long)
            detector_long = detector_long.deg
        # longitude in range (-180,180)

        self.detector_location = EarthLocation(lat=detector_lat*u.deg, lon=detector_long*u.deg, height=detector_height_m*u.m)
        self.observer_frame = None
        self.datetime_obj = None

    def parse_date_string(self, date_str):
        """
        Parse the date string and create an Astropy Time object.

        Parameters:
            date_str (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            astropy.time.Time: Time object representing the provided date string.
        """

        # Define the format of the input date string with seconds
        date_format = "%B %d, %Y, %H:%M:%S"
        # Try parsing the date string with seconds
        try:
            datetime_obj = Time.strptime(date_str, date_format)
        except ValueError:
            # If parsing with seconds fails, try parsing without seconds
            date_format_no_seconds = "%B %d, %Y, %H:%M"
            try:
                datetime_obj = Time.strptime(date_str, date_format_no_seconds)
            except ValueError:
                raise ValueError("Invalid date format. Please provide a date in the form 'July 15, 2023, 14:30' or 'July 15, 2023, 14:30:25'.")
                
        self.datetime_obj = datetime_obj
        
        return datetime_obj
    

    def get_coszen_altitude_and_azimuth(self, ra_deg, dec_deg, date_str):
        """
        Get the cosine of the zenith angle and the azimuth corresponding to the given 
        declination and right ascension (at the specified date/time). Depends on the detector location.

        Parameters:
            ra_deg (float): Declination in degrees.
            dec_deg (float): Right ascension in degrees.
            date_str (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            tuple: Tuple containing coszen (cosine of the zenith angle) and azimuth in degrees.
        """

        # Create observation time object
        if self.datetime_obj is None:
            date_obj = self.parse_date_string(date_str)
        else:
            date_obj = self.datetime_obj
        
        # Create neutrino direction
        # TODO: should this be the opposite direction (e.g., travel direction vs origin direction?)
        nu_dir = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        # SkyCoord object that is per default in the icrs frame
        
        # Convert to local coordinate system (zenith, azimuth)
        # Coordinate transformation to the horizontal coordinate system. Azimuth is oriented East of North (i.e., N=0, E=90 degrees) 
        # Altitude is the angle above the horizon
        frame = AltAz(obstime=date_obj, location=self.detector_location)
        nu_dir_alt_az = nu_dir.transform_to(frame)

        # Get zenith angle and convert to cos(zen)
        coszen = np.cos(nu_dir_alt_az.zen.to(u.rad).value)
        
        # Get altitude angle
        altitude = nu_dir_alt_az.alt.to(u.rad).value
        
        # Get azimuthal angle
        azimuth = nu_dir_alt_az.az.to(u.rad).value
        
        return coszen, altitude, azimuth
    
    
    def get_right_ascension_and_declination(self, coszen, azimuth, date_str):
        """
        Get the right ascension and declination corresponding to the given 
        cosine of the zenith angle and azimuth (at the specified date/time). Depends on the detector location.

        Parameters:
            coszen (float): Cosine of the zenith angle.
            azimuth (float): Azimuth angle in degrees.
            date_str (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            tuple: Tuple containing right ascension and declination in degrees.
        """
        # Create observation time object
        if self.datetime_obj is None:
            date_obj = self.parse_date_string(date_str)
        else:
            date_obj = self.datetime_obj

        # Create neutrino direction in AltAz frame
        nu_dir_alt_az = AltAz(az=azimuth*u.deg, alt=(np.arccos(coszen)-np.pi/2)*u.rad, obstime=date_obj, location=self.detector_location)
        
        icrs_frame_observed = ICRS()
        # Transform back to ICRS frame (right ascension, declination)
        nu_dir_icrs = nu_dir_alt_az.transform_to(icrs_frame_observed)
    
        # Get right ascension and declination in degrees
        right_ascension = nu_dir_icrs.ra.degree
        declination = nu_dir_icrs.dec.degree
    
        return right_ascension, declination


    def get_observer_frame(self,date_str):
        
        # Create observation time object
        if self.datetime_obj is None:
            date_obj = self.parse_date_string(date_str)
        else:
            date_obj = self.datetime_obj
            
        self.observer_frame = AltAz(obstime=date_obj, location=self.detector_location)
        
        return self.observer_frame
    
    
    def rotate_vector_from_ICRS_to_local_AltAz_frame(self,date_str,vec):
        
        if self.observer_frame is None:
            observer_frame = self.get_observer_frame(date_str)
        else:
            observer_frame = self.observer_frame
            
        # Define the Cartesian coordinates for the unit vectors in the ICRS frame
        # The x, y, and z coordinates are simply 1 for the respective directions
        x_unit_vector = 1 * u.dimensionless_unscaled
        y_unit_vector = 1 * u.dimensionless_unscaled
        z_unit_vector = 1 * u.dimensionless_unscaled
        
        # Initialize the SkyCoord objects for the unit vectors
        # Use representation_type='cartesian' to specify Cartesian coordinates
        x_vector_icrs = SkyCoord(x=x_unit_vector, y=0*u.dimensionless_unscaled, z=0*u.dimensionless_unscaled, frame='icrs', representation_type='cartesian')
        y_vector_icrs = SkyCoord(x=0*u.dimensionless_unscaled, y=y_unit_vector, z=0*u.dimensionless_unscaled, frame='icrs', representation_type='cartesian')
        z_vector_icrs = SkyCoord(x=0*u.dimensionless_unscaled, y=0*u.dimensionless_unscaled, z=z_unit_vector, frame='icrs', representation_type='cartesian')
        
        # Transform unit vectors to the topocentric frame specified by zenith angle and azimuth
        x_vector_topo = x_vector_icrs.transform_to(observer_frame)
        y_vector_topo = y_vector_icrs.transform_to(observer_frame)
        z_vector_topo = z_vector_icrs.transform_to(observer_frame)
        
        new_vec = np.zeros(vec.shape)
        
        new_vec[0] = vec[0] * x_vector_topo.cartesian.xyz[0] + vec[1] * y_vector_topo.cartesian.xyz[0]
        new_vec[1] = vec[0] * x_vector_topo.cartesian.xyz[1] + vec[1] * y_vector_topo.cartesian.xyz[1]
        new_vec[2] = vec[0] * x_vector_topo.cartesian.xyz[2] + vec[1] * y_vector_topo.cartesian.xyz[2]
        
        return new_vec
    
    def get_colatitude(self):
        # Get the latitude of the detector in radians
        lat_rad = self.detector_location.lat.to(u.rad).value
        
        # Calculate the colatitude (90 degrees minus the latitude)
        colatitude_deg = np.pi/2 - lat_rad
        
        return colatitude_deg
    
    
    def get_local_sidereal_time(self, date_str):
        """
        Get the local sidereal time corresponding to the given date/time.

        Parameters:
            date_str (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            float: Local sidereal time in degrees.
        """
        # Create observation time object
        date_obj = self.parse_date_string(date_str)
        # Calculate local sidereal time
        lst = date_obj.sidereal_time('apparent', longitude=self.detector_location.lon)
        return lst.to(u.deg).value

    def get_angle_to_aries(self, date_str):
        """
        Get the angle between the lines from Earth and the first point of Aries to the Sun at the specified date/time.

        Parameters:
            date_str (str): Date string in the format "Month day, Year, Hour:Minute(:Second)" (e.g., "July 15, 2023, 14:30").
        
        Returns:
            float: Angle to the first point of Aries in degrees.
        """
        # Get the position of the Sun at the specified date/time
        date_obj = self.parse_date_string(date_str)
        sun_position = get_sun(date_obj)
    
        # Calculate the angle between the lines from Earth and the first point of Aries to the Sun
        aries = SkyCoord(ra=0*u.deg, dec=0*u.deg, frame='icrs')
        angle_to_aries = sun_position.separation(aries)
    
        return angle_to_aries.to(u.deg).value
        
    def path_length(self, coszen, prodHeight, detectorDepth):
        """
        Calculate the path length of neutrinos from the production height to the detector.

        Parameters:
            coszen (array-like): Cosine of the zenith angle(s) of the neutrinos.
            prodHeight (float): Height at which the neutrinos are produced above the Earth's surface (in km).
            detectorDepth (float): Depth at which the detector lies below the Earth's surface (in km).

        Returns:
            numpy.ndarray: An array containing the path lengths of neutrinos in the same shape as coszen.       
        """
        
        # Approximate Earth's radius in km
        rEarth = 6371
        
        #Create array of the same form as coszen variable to store pathlengths
        path_lengths = np.zeros_like(coszen)
        
        #Calculate pathlength
        expression = (rEarth - detectorDepth)**2 * (-1 + coszen**2) + (rEarth + prodHeight)**2
        # Check for non-negative values
        mask = expression >= 0  
        path_lengths = -rEarth * coszen + np.sqrt(expression[mask])
        
        return path_lengths
