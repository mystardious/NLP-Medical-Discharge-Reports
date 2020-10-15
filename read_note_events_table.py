import pandas as pd

data = pd.read_csv('data/NOTEEVENTS.csv.gz', nrows=100, compression='gzip', error_bad_lines=False)

print(data.TEXT)

with open('data/discharge_reports_100samples.txt', 'w') as f:
    for item in data.TEXT:
        f.write("%s\n|||\n" % item)