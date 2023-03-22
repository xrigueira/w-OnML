
import os
import numpy as np
import pandas as pd
from datetime import datetime

"""There are two functions in this file. findMonday() returns the index of the first Monday in the data base,
and dater() takes the filled database and adds a series of time-related columns to make that information
more accessible"""

# Function to get the index of the firts Monday
def findMonday(Dataframe):
    
    for i in range(len(Dataframe)):

        d = datetime(Dataframe['year'][i], Dataframe['month'][i], Dataframe['day'][i])
        
        if d.weekday() == 0:
            print('First Monday index: {} | {}'.format(i, d))
            break
    
    return i

def dater(File, timestep):

    if timestep == '15 min':
    
        fileName, fileExtension = os.path.splitext(File)
        df = pd.read_csv(f'data/{fileName}.csv', delimiter=';', parse_dates=['date'], index_col=['date'])

        # Add the needed columns (year, month, day, hour, min, sec)
        year = [i for i in df.index.year]
        month = [i for i in df.index.month]
        day = [i for i in df.index.day]
        hour = [i for i in df.index.hour]
        minute = [i for i in df.index.minute]
        second = [i for i in df.index.second]

        df['year'] = year
        df['month'] = month
        df['day'] = day
        df['hour'] = hour
        df['minute'] = minute
        df['second'] = second

        # Store the index of the first Monday
        mondayIndex = findMonday(df)

        # Add three columns with the week number, start date, and end date, respectively
        weekIndex = []
        weekNumber = 0
        for i in range(mondayIndex):
            if i < mondayIndex:
                weekIndex.append(0)
                
        for i in range(len(df) - mondayIndex):
            
            if i % 672 == 0:
                weekNumber += 1
            weekIndex.append(weekNumber)

        df['week'] = weekIndex

        # Get the week order within every month
        weekOrder = []
        weekPosition = 0
        for i, e in enumerate(df['week']):
            
            if e == 0:
                weekOrder.append(0)
            
            else:
                if df['week'][i] == df['week'][i-1]:
                    weekOrder.append(weekPosition)
                elif df['week'][i] != df['week'][i-1]:
                    weekPosition += 1
                    if weekPosition == 5:
                        weekOrder.append(1)
                    else:
                        weekOrder.append(weekPosition)
                    if weekPosition == 5:
                        weekPosition = 1

        df['weekOrder'] = weekOrder

        # Save the new new file
        df.index.name = 'date'
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName[0:-5]}_col.csv', sep=';', encoding='utf-8', index=True, header=cols)

    elif timestep == '1 day':
        
        fileName, fileExtension = os.path.splitext(File)
        df = pd.read_csv(f'data/{fileName}.csv', delimiter=';', parse_dates=['date'], index_col=['date'])
        
        # Add the needed columns (year, month, day)
        year = [i for i in df.index.year]
        month = [i for i in df.index.month]
        day = [i for i in df.index.day]

        df['year'] = year
        df['month'] = month
        df['day'] = day
        
        # Store the index of the first Monday
        mondayIndex = findMonday(df)
        
        # Add three columns with the week number, start date, and end date, respectively
        weekIndex = []
        weekNumber = 0
        for i in range(mondayIndex):
            if i < mondayIndex:
                weekIndex.append(0)
                
        for i in range(len(df) - mondayIndex):
            
            if i % 7 == 0:
                weekNumber += 1
            weekIndex.append(weekNumber)

        df['week'] = weekIndex
        
        # Get the week order within every month
        weekOrder = []
        weekPosition = 0
        for i, e in enumerate(df['week']):
            
            if e == 0:
                weekOrder.append(0)
            
            else:
                if df['week'][i] == df['week'][i-1]:
                    weekOrder.append(weekPosition)
                elif df['week'][i] != df['week'][i-1]:
                    weekPosition += 1
                    if weekPosition == 5:
                        weekOrder.append(1)
                    else:
                        weekOrder.append(weekPosition)
                    if weekPosition == 5:
                        weekPosition = 1

        df['weekOrder'] = weekOrder

        # Save the new new file
        df.index.name = 'date'
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName[0:-5]}_col.csv', sep=';', encoding='utf-8', index=False, header=cols)
