
# coding: utf-8

# In[ ]:


import sys
print(sys.version)


# In[ ]:


import datetime
import gc
import glob
import os
import re
import time

from mpi4py import MPI
import wfdb

import pandas

print(pandas.__version__)
print(pandas.__path__)


# In[ ]:


data_files = glob.glob('../../data/physionet/mimic3wdb/matched/p0?/p0?????/p0?????-????-??-??-??-??.hea')
data_files.sort()

# data_files = data_files[:100]

print(len(data_files))
data_files[:10]


# In[ ]:

comm = MPI.COMM_WORLD
N = comm.Get_size()
rank = comm.Get_rank()

start = int(rank * len(data_files) / N)
end = int((rank+1) * len(data_files) / N)
if end > len(data_files):
    end = -1


# In[ ]:

if rank == 0:
    print(len(data_files))


# In[ ]:


# %%time

BLOCKSIZE = 400000000

if rank == 0:
    tstart = time.clock()

for i, data_file in enumerate(data_files[start:end]):

    print('[%d] Processing %s (%d, %d)...' % (os.getpid(), data_file, i, len(data_files)))

    try:

        subfiles = []
        with open(data_file) as f:
            header = f.readline().split(' ')
            if '' in header:
                header.remove('')
            _, _, _, _, _, _date = header

            prefix = f.readline().split('_')[0]

            for line in f.readlines():
                subfiles.append(line.split(' ')[0])

        pattern = re.compile(prefix + r'_\d{4}')
        subfiles = list(filter(lambda subfile: pattern.match(subfile), subfiles))
        subfiles = list(map(lambda subfile: '%s/%s.hea' % (os.path.dirname(data_file), subfile), subfiles))

        output_df = None
        count = 0
        for subfile in subfiles:
            with open(subfile) as f:
                header = f.readline().split(' ')
                if '' in header:
                    header.remove('')
                name, _, fs, length, _time = header

            time_str = '%s %s' % (_date.strip(), _time.strip())

            try:
                dt = datetime.datetime.strptime(time_str, '%d/%m/%Y %H:%M:%S.%f')
            except ValueError:
                dt = datetime.datetime.strptime(time_str, '%d/%m/%Y %H:%M:%S')

            sig, fields = wfdb.srdsamp(subfile[:-4])
            freq = '8ms'
            index = pandas.DatetimeIndex(start=dt, freq=freq, periods=int(length))

            df = pandas.DataFrame()
            df['time'] = index
            df['record'] = name
            for i, signal in enumerate(fields['signame']):
                df[signal] = sig[:,i]

            df.index = [df['time'], df['record']]
            df = df.drop('time', axis=1)
            df = df.drop('record', axis=1)
            df = df.stack().reset_index()
            df.columns = ('time', 'record', 'parameter', 'value')

            if output_df is None:
                output_df = df
            else:
                output_df = pandas.concat((output_df, df))

            threshold = int(BLOCKSIZE / (output_df.memory_usage().sum()/len(output_df)))
            while len(output_df) > threshold:
                output_file = '%s/%s.%04d.parquet' % (os.path.dirname(data_file), prefix, count)
                output_df[:threshold].to_parquet(output_file)

                output_size = os.path.getsize(output_file)
                if output_size > 100000000:
                    print('\n+[%d] output_size > 100000000!: len(output_df) = %d, threhold = %d * %d / %d, output_file = %s\n' % (os.getpid(), len(output_df), BLOCKSIZE, len(output_df), output_df.memory_usage().sum(), output_file))

                output_df = output_df[threshold:]

                output_size = os.path.getsize(output_file)
                if output_size > 100000000:
                    print('\n-[%d] output_size > 100000000!: len(output_df) = %d, threhold = %d * %d / %d, output_file = %s\n' % (os.getpid(), len(output_df), BLOCKSIZE, len(output_df), output_df.memory_usage().sum(), output_file))

                count += 1
                gc.collect()
                sys.stdout.write('w')

            sys.stdout.write('.')
            sys.stdout.flush()
        print()

        output_file = '%s/%s.%04d.parquet' % (os.path.dirname(data_file), prefix, count)
        output_df.to_parquet(output_file)
        sys.stdout.write('w')

    except Exception as e:
        import logging
        print('\n')
        logging.exception('Failed to process %s!', data_file)
        print('\n')

comm.barrier()

if rank == 0:
    tend = time.clock()
    print('%fs elapsed.' % (tend-tstart))
