import pandas as pd
from streamz.dataframe import DataFrame
from streamz import Stream

# stream = Stream()
example = pd.read_csv('loginData-2018-11-22-12-00-06.csv')
# sdf = DataFrame(stream, example=example)
#
# def printOut(str):
#     print(str)
#
# print(sdf.groupby(sdf.USERNAME).count())

def agg_test(series):
    print(series)
    return len(series) - series.count()

print(example.groupby('USERNAME').rolling(5).agg(agg_test))