# w-OnML
 Online learning for watar quality prediction and anomaly detection.

## How to process the data?
1. Get the concatenated files for each station desired from the (data repo)[https://github.com/xrigueira/data] and place them in the /data directory. For example: 'ammonium_901.txt', 'ph_901.txt', ... No other files should be in this directory when preprocessing.

2. The function checkGaps() will fill in the gaps in each data file.

3. The function dater() will add time-related columns to the data. See dater.py for details. This function is not needed anymore as the method should not see these columns containing time information.

4. The function joiner() will joined each univariate file into a multivariate data base and save it as 'merged_{station number}.csv'.

5. Run the lableler.py file to label the data.