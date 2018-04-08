
# coding: utf-8

# In[ ]:


RATE = 8

NCORES = 8
NTASKS_PER_CORE = 1

NTASKS = -1


# In[ ]:


import sys
sys.version


# In[ ]:


import datetime


# In[ ]:


tstart = datetime.datetime.now()
print(tstart.isoformat())


# In[ ]:


# %%time

import matplotlib.pyplot as plt

import pandas
print(pandas.__version__)
print(pandas.__path__)

import pyarrow
print(pyarrow.__version__)
print(pyarrow.__path__)

import pyspark
import pyspark.sql
print(pyspark.__version__)
print(pyspark.__path__)


# In[ ]:


# %%time

spark = pyspark.sql.SparkSession                                     .builder                                          .appName('ignition')                              .master('local[%d]' % NCORES)                     .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

print(spark._sc.defaultParallelism)
print(spark._sc.defaultMinPartitions)


# In[ ]:


# %%time

sdf = spark.read.csv('parquet_files.csv', header=True, inferSchema=True)
count = sdf.count()

if NTASKS > 0:
    sdf = sdf.limit(NTASKS)

print('%d out of %d' % (sdf.count(), count))
sdf.show()


# In[ ]:


# %%time

grouped_rdd = sdf.select('record_id', 'file', 'size').rdd.map(lambda row: (row[1], row)).groupByKey()


# In[ ]:

def __extract_features(row, callback):
    try:
        import gc, os, time

        parquet_files_pdf = pandas.DataFrame(list(row), columns=('record_id', 'file', 'size'))

        tstart_input_read = time.clock()
        pdf = pandas.concat(list(parquet_files_pdf['file'].map(lambda parquet_file: pandas.read_parquet(parquet_file))))
        tend_input_read = time.clock()
        input_size = pdf.memory_usage().sum()

    #     print(parquet_files_pdf)
    #     print(len(pdf))
    #     print(pdf.head())

        if len(pdf) > 0:

            tstart_pivot = time.clock()
            pdf = pdf.pivot_table(index='time', columns='parameter', values='value', aggfunc='mean')
            pdf.sort_index(inplace=True)
            tend_pivot = time.clock()
            pivot_size = pdf.memory_usage().sum()

            tstart_resample = time.clock()
            pdf.fillna(method='ffill', inplace=True)
            pdf = pdf.resample('%dus' % int(1000000 / RATE)).mean()
            tend_resample = time.clock()
            resample_size = pdf.memory_usage().sum()

            gc.collect()

            tstart_callback = time.clock()
            result_pdf = callback(pdf)
            tend_callback = time.clock()
            result_size = result_pdf.memory_usage().sum()

            del pdf
            gc.collect()

            tstart_output_write = time.clock()
            output_path = parquet_files_pdf['file'][0][:-8] + '.features.parquet'
            result_pdf.to_parquet(output_path)
            tend_output_write = time.clock()
            output_size = os.path.getsize(output_path)

            return (True,
                    parquet_files_pdf['size'].sum(),
                    tend_input_read-tstart_input_read,
                    input_size,
                    tend_pivot-tstart_pivot,
                    pivot_size,
                    tend_resample-tstart_resample,
                    resample_size,
                    tend_callback-tstart_callback,
                    result_size,
                    tend_output_write-tstart_output_write,
                    output_size,
                    tend_output_write-tstart_input_read,
                    None)

        else:
            return (True,
                    parquet_files_pdf['size'].sum(),
                    tend_input_read-tstart_input_read,
                    input_size,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, None)

    except Exception as e:
        return (False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e)


# In[ ]:


def extract_features(pdf):
    import numpy
    import scipy.signal

    result_pdf = None
    for column in pdf.columns:
        local_mins = scipy.signal.argrelextrema(pdf[column].values, numpy.less)[0]
        local_maxs = scipy.signal.argrelextrema(pdf[column].values, numpy.greater)[0]
        local_minmaxs = numpy.sort(numpy.concatenate((local_mins, local_maxs)))

        if result_pdf is None:
            result_pdf = pandas.DataFrame()
            result_pdf[column] = pdf[column][local_minmaxs]
        else:
            tmp_pdf = pandas.DataFrame()
            tmp_pdf[column] = pdf[column][local_minmaxs]
            result_pdf = pandas.merge(result_pdf, tmp_pdf, left_index=True, right_index=True, how='outer')
    return result_pdf


# In[ ]:


print('Go!')


# In[ ]:


# %%time

print('# tasks: %d' % grouped_rdd.count())

print('# partitions before repartition: %d' % grouped_rdd.getNumPartitions())

grouped_rdd = grouped_rdd.repartition(NTASKS_PER_CORE * NCORES)

print('# partitions after repartition: %d' % grouped_rdd.getNumPartitions())

grouped_rdd = grouped_rdd.map(lambda row: __extract_features(row[1], extract_features))

results = grouped_rdd.collect()
print(len(results))


# In[ ]:

result_pdf = pandas.DataFrame(results, columns=('status',
                                                'input_file_size',
                                                'input_read_duration',
                                                'input_size',
                                                'pivot_duration',
                                                'pivot_size',
                                                'resample_duration',
                                                'resample_size',
                                                'callback_duration',
                                                'result_size',
                                                'output_write_duration',
                                                'output_file_size',
                                                'total_duration',
                                                'exception'))
print(result_pdf.describe())
result_pdf.to_csv('profile.csv')


# In[ ]:

# print(result_pdf.head())


# In[ ]:

print('%d bytes of input read' % result_pdf['input_file_size'].sum())
print('%d bytes of input uncompressed' % result_pdf['input_size'].sum())
print('%d bytes of output uncompressed' % result_pdf['result_size'].sum())
print('%d bytes of output written' % result_pdf['output_file_size'].sum())

print('%fs total processing time' % result_pdf['total_duration'].sum())


# In[ ]:


tend = datetime.datetime.now()
print(tend.isoformat())
print('%fs elapsed' % (tend-tstart).total_seconds())
