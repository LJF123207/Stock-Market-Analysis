import pyarrow.parquet as pq

table = pq.read_table("/z5s/morph/home/ljf/qlib/data_stocks/IndexAndNews/801125.SI.pqt", columns=None)
df = table.to_pandas().head()
print(df)
