import os

from checkGaps import checkGaps
from dater import dater
from joiner import joiner
from mfilterer import mfilterer
from normalizer import normalizer

# Define the data we want to study
files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]

varNames = [i[0:-4] for i in files] # Extract the names of the variables, e.g.: pH_901.txt (4 intervals already appended following temporal sequence)
stations = [980] # Define with stations to process

# Define the time frame we want to use (a: months, b: weeks, c: days)
timeFrame = 'b'
timeStep = '15 min'


if __name__ == '__main__':

    for varName in varNames:
        # Fill in the gaps in the time series
        checkGaps(File=f'{varName}.txt', timestep=timeStep)
        print('[INFO] checkGaps() DONE')

        # Add time-related columns to the data. See columner.py for details
        dater(File=f'{varName}_full.csv', timestep=timeStep)
        print('[INFO] normalizer() DONE')
        
    
    for station in stations:
        # Join the databases by station number
        joiner(station=station)
        print(f'[INFO] joiner() DONE | station: {station}')

        # Filter out those months, weeks or days (depending on the desired
        # time unit) with too many NaN in several variables and iterate on the rest
        # mfilterer(File='joined.csv', timeframe=timeFrame, timestep=timeStep)
        # print('[INFO] filterer() DONE')

        # Normalize the results
        normalizer(station=station)
        print(f'[INFO] normalizer() DONE | station: {station}')