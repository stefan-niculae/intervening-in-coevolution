from sys import argv
from multiprocessing import Process, Queue, current_process, cpu_count
import queue  # imported for using queue.Empty exception


import time
import random

# TODO replace with actual args (config path)
DUMMY_ARGS = 'ABCDEFGHIJKLM'


def process(arg: int):
    # TODO replace with actual work (main.py)
    pid = current_process().pid
    print(f'{pid}: start {arg}')
    time.sleep(random.random() * 5)
    print(f'{pid}: finish {arg}')


def get_arg_and_process(args_q):
    while True:
        try:
            arg = args_q.get_nowait()
        except queue.Empty:
            # No more tasks
            break

        else:
            # Got a task
            process(arg)
    return True


def main():
    num_processes = int(argv[1])
    print(f'Running {num_processes} processes (CPU count: {cpu_count()})')

    # Instantiate arguments queue
    args_q = Queue()
    for arg in DUMMY_ARGS:
        args_q.put(arg)

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
