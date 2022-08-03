import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import multiprocessing as mp
import time
import os

# To make tensorflow and CUDA work with multiprocessing, this article really helped:
# https://sefiks.com/2019/03/20/tips-and-tricks-for-gpu-and-multiprocessing-in-tensorflow/

# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
# https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
# The problem is, that Tensorflow is greedy in allocating all available VRAM. That causes issues when multi processes start using CUDA
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_run_graph(params):
    
    # Must move all tf context initializations inside the child process
    import qsimpy
    from qsimpy.random import Deterministic
    from qsimpy_aqm.codel import CodelQueue

    from arrivals import HeavyTailGamma

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = qsimpy.Model(name=f"Codel AQM benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate = 0.095,
        seed = params['arrival_seed'],
        dtype = 'float64',
    )
    source = qsimpy.TimedSource(
        name='start-node',
        arrival_rp=arrival,
        task_type='0',
        delay_bound=params['target_delay'],
    )
    model.add_entity(source)

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed = params['service_seed'],
        gamma_concentration = 5,
        gamma_rate = 0.5,
        gpd_concentration = 0.1,
        threshold_qnt = 0.8,
        dtype = 'float64',
        batch_size = params['arrivals_number'],
    )
    queue = CodelQueue(
        name='queue',
        service_rp= service,
        interval = params['codel_interval'],
        target = params['codel_target'],
    )
    model.add_entity(queue)

    # Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
    sink = qsimpy.PolarSink(
        name='sink',
        batch_size = 10000,
    )
    # define postprocess function: the name must be 'user_fn'

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df['end2end_delay'] = df['end_time']-df['start_time']
        df['service_delay'] = df['end_time']-df['service_time']
        df['queue_delay'] = df['service_time']-df['queue_time']
        # process time in service
        df['time_in_service'] = df.apply(
                                lambda row: (row.start_time-row.last_service_time) if row.queue_is_busy else None,
                                axis=1,
                            ).astype('float64')
        # process longer_delay_prob here for benchmark purposes
        df['longer_delay_prob'] = np.float64(1.00) - service.cdf(
            y = df['time_in_service'].to_numpy(),
        )
        df['longer_delay_prob'] = df['longer_delay_prob'].fillna(np.float64(0.00))
        del df['last_service_time'], df['queue_is_busy']
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # Wire start-node, queue, end-node, and sink together
    source.out = queue.name
    queue.out = sink.name
    queue.drop = sink.name

    # Setup task records
    model.set_task_records({
        'timestamps' : {
            source.name : {
                'task_generation':'start_time',
            },
            queue.name : {
                'task_reception':'queue_time',
                'service_start':'service_time',
                'service_end':'end_time',
            },
        },
        'attributes' : {
            source.name : {
                'task_generation' : {
                    queue.name : {
                        'queue_length':'queue_length',
                        'last_service_time':'last_service_time',
                        'is_busy':'queue_is_busy',
                    },
                },
            },
        },
    })

    modeljson = model.json()
    with open(params['records_path']+f"{params['run_number']}_model.json", 'w', encoding='utf-8') as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        print(f"{params['run_number']}: Simulation progress {100.0*float(model.env.now)/float(params['until'])}% done")
    for step in np.arange(0, params['until'], params['until']*params['report_state'], dtype=int):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params['until'])
    end = time.time()
    print("{0}: Run finished in {1} seconds".format(params['run_number'],end - start))

    print("{0}: Source generated {1} tasks".format(params['run_number'],source.get_attribute('tasks_generated')))
    print("{0}: Queue completed {1}, dropped {2}".format(
            params['run_number'],
            queue.get_attribute('tasks_completed'),
            queue.get_attribute('tasks_dropped'),
        )
    )
    print("{0}: Sink received {1} tasks".format(params['run_number'],sink.get_attribute('tasks_received')))

    start = time.time()

    # Process the collected data
    df = sink.received_tasks
    #print(df)

    end = time.time()
    
    df.to_parquet(params['records_path'] + f"{params['run_number']}_records.parquet")

    print("{0}: Data processing finished in {1} seconds".format(params['run_number'],end - start))



if __name__ == "__main__":

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + '/projects/aqm_benchmark/'

    # simulation parameters
    # quantile values of no-aqm model with p1 as gpd_concentration
    # these Codel parameters are tuned, NOTE: target_delay is the queuing delay
    bench_params = { # target_delay
        'p999':{'target_delay':131.054472733289,'codel_target':55,'codel_interval':55*3},
        'p99':{'target_delay':107.70908319205046,'codel_target':50,'codel_interval':50*3}, 
        'p9':{'target_delay':73.76106050610542,'codel_target':40,'codel_interval':40}, 
        'p8':{'target_delay':57.15778886526823,'codel_target':30,'codel_interval':15},
    }

    # important command for multiprocessing
    mp.set_start_method('spawn', force=True)

    # 4 x 4, until 1000000 took 7 hours
    sequential_runs = 4 # 4
    parallel_runs = 4 # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):

            # parameter figure out
            keys = list(bench_params.keys())
            key_this_run = keys[j%len(keys)]

            # create and prepare the results directory
            results_path = project_path + key_this_run  + '_results/'
            records_path = results_path + 'records_codel/'
            os.makedirs(records_path, exist_ok=True)

            params = {
                'records_path' : records_path,
                'arrivals_number' : 100000, #5M #1.5M
                'run_number' : j*parallel_runs + i,
                'arrival_seed' : 100234+i*100101+j*10223,
                'service_seed' : 120034+i*200202+j*20111,
                'until': int(1000000), # 10M timesteps takes 1000 seconds, generates 900k samples
                'report_state' : 0.05, #0.05 # report when 10%, 20%, etc progress reaches
            }

            # add codel params
            params = { **params, **bench_params[key_this_run] }
            
            p = mp.Process(target=create_run_graph, args=(params,))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                p.join()
                exit(0)