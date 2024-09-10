import pandas as pd
from ydata_profiling import ProfileReport
import os

#column_names = ['sepal length','sepal width', 'petal length','petal width','y']
df = pd.read_csv('./data/bezdekIris.csv')
profile = ProfileReport(df, title="Profiling Report")

profile.to_file('./report.html')
