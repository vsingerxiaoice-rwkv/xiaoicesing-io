import platform
import re
import traceback
from torch.multiprocessing import Pool, Manager, current_process, get_context

is_main_process = not bool(re.match(r'(Process)|(SyncManager)|(.*PoolWorker)-\d+', current_process().name))


def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def run_and_collect_once(args):
    map_func = args[0]
    map_func_args = args[1]
    result_queue = args[2]
    # noinspection PyBroadException
    try:
        res = map_func(*map_func_args)
        result_queue.put(res)
    except:
        traceback.print_exc()
        result_queue.put(None)


def chunked_multiprocess_run(map_func, args, num_workers, q_max_size=100):
    n_jobs = len(args)
    queue = Manager().Queue(maxsize=q_max_size)
    if platform.system().lower() != 'windows':
        pool_creation_func = get_context('spawn').Pool
    else:
        pool_creation_func = Pool
    with pool_creation_func(processes=num_workers) as pool:
        pool.map_async(run_and_collect_once, [(map_func, i_args, queue) for i_args in args])
        for n_finished in range(n_jobs):
            res = queue.get()
            yield res
        pool.close()
        pool.join()
