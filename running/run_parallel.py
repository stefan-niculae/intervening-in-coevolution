from os import listdir
from sys import argv
from multiprocessing import Process, Queue, current_process, cpu_count
from queue import Empty as queue_empty
import traceback

from main import main as train_main


def get_arg_and_process(args_q):
    while True:
        try:
            arg = args_q.get_nowait()
        except queue_empty:
            # No more tasks
            break

        else:
            # Got a task
            try:
                pid = current_process().pid
                print(f'PID {pid} starting {arg}')
                train_main(arg)
                print(f'PID {pid} finished {arg}')
            except Exception as e:
                print(f'Exception ({e}) while processing {arg}.')
                traceback.print_exc()
    return True


def main():
    try:
        num_processes = int(argv[1])
        configs_dir = argv[2]
    except:
        print('usage: run_parallel.py <num_processes> <configs_dir>')
    config_names = listdir(configs_dir)

    print(f'Running {num_processes} processes (CPU count: {cpu_count()})')
    print(f'Running {len(config_names)} configs from {configs_dir}')

    # Instantiate arguments queue
    args_q = Queue()
    for name in config_names:
        args_q.put(configs_dir + '/' + name)

    # Create and start processes
    processes = []
    for _ in range(num_processes):
        p = Process(target=get_arg_and_process, args=(args_q,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
