# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as its
import operator as op
from multiprocessing.shared_memory import SharedMemory
from struct import calcsize, pack_into, unpack_from
from threading import TIMEOUT_MAX, Thread, Timer
from time import sleep
from typing import Callable, Iterable, NoReturn, Optional, Tuple, Union
from warnings import warn

import more_itertools as mit
from tqdm.auto import tqdm

__all__ = ['Buffer', 'Parallel']


class Buffer:
    """Buffer for shared memory.

    Buffer for shared memory.

    Args:
        sequence_or_size: Initialize 'Buffer' with given sequence or by the
            given size.
        dtype: A valid type.
    """

    def __init__(self,
                 sequence_or_size: Union[Iterable[any], int],
                 dtype: str = 'q'):

        # Check dtype
        if len(dtype) != 1:
            raise ValueError("'dtype' is not valid.")

        try:
            calcsize(dtype)
        except Exception:
            raise ValueError("'dtype' is not valid.")

        # Get sequence and size
        try:
            sequence = None
            size = int(sequence_or_size)
        except (ValueError, TypeError):
            sequence = tuple(sequence_or_size)
            size = len(sequence)

        # Set type
        self._dtype = dtype

        # Create buffer
        self.shm = SharedMemory(create=True, size=size * calcsize(self._dtype))

        # Initialize it
        if sequence:
            pack_into(self.dtype * size, self.shm.buf, 0, *sequence)

    def __len__(self):
        return self.shm.size // calcsize(self.dtype)

    def __repr__(self):
        return repr(list(self))

    def __getitem__(self, pos):
        if pos >= len(self):
            raise IndexError("list index out of range")
        return unpack_from(self.dtype, self.shm.buf,
                           pos * calcsize(self.dtype))[0]

    def __setitem__(self, pos, value):
        if pos >= len(self):
            raise IndexError("list index out of range")
        pack_into(self.dtype, self.shm.buf, pos * calcsize(self.dtype), value)

    @property
    def dtype(self):
        return self._dtype


def Parallel(core: Callable,
             *args,
             description: Optional[str] = "Processing...",
             text: Optional[str] = "",
             n_jobs: Optional[int] = -1,
             timeout: Optional[float] = None,
             verbose: Optional[int] = False,
             buffers: Optional[Iterable[Tuple[str, str]]] = (),
             refresh_per_second: Optional[float] = None,
             leave: Optional[bool] = False,
             **kwargs) -> NoReturn:
    """Parallelize 'core'.

    'Parallel' can be used to parallelize an arbitrary 'core'.

    Args:
        core: Function to parallelize. The core must have the keyword
            parameters 'idx' (which is used to identify the process), 'status'
            (which is used by the process to inform the main process of its
            status), and 'stop' (which instruct the process to stop if 'True').
            All the arguments 'args' and keyword arguments 'kwargs' are passed
            to the 'core'.
        description: Description to use.
        text: Text to append to the progress bar.
        n_jobs: Number of processes to use. By default, all available cores are
            used. If 'n_jobs' is a positive number, 'n_jobs' processes will be
            used. If 'n_jobs' is negative, 'n_cpus + n_jobs + 1' will be used.
            If 'n_jobs' is zero, it will raise a 'ValueError'. (See:
            'joblib.Parallel')
        timeout: If provided, set the 'stop' variables to all process to 'True'
            after 'timeout' seconds have passed.
        verbose: Verbose output.
        buffers: Buffers used to communicate between processes. Each buffer is
            of the form '(buffer_name, type)', with 'buffer_name' being the
            name of the buffer and 'type' its type (See:
            'tnco.parallel.Buffer'). Buffers are then passed as keyword
            arguments to 'core'.
        refresh_per_second: Number of refresh per seconds.
        leave: If 'True', progress bars are kept at the end of the process.

    Examples:
        def example(x, *, idx, status, stop, log_buffer):
            from time import sleep
            from math import log

            # Exit if stop is already set
            if stop[idx]:
                return 'stopped'

            # Initialize the status. Status must be a number between
            # 0 (0%) and 1 (100%)
            status[idx] = 0

            for n in range(100):
                sleep(0.05)
                x += x
                status[idx] = (n + 1)/100

                # Update buffer
                log_buffer[idx] = log(n + 1)

                # If stop is set, break loop
                if stop[idx]:
                    break

            # Return results
            return x

        Parallel(example,
                 x=range(10),
                 n_jobs=4,
                 description="Example",
                 text="LOG={log_buffer:1.2f}",
                 buffers=[('log_buffer', 'f')],
                 verbose=3,
                 timeout=2)
        > [0,
        >  1099511627776,
        >  2199023255552,
        >  3298534883328,
        >  'stopped',
        >  'stopped',
        >  'stopped',
        >  'stopped',
        >  'stopped',
        >  'stopped']
    """
    # Try to load joblib
    if n_jobs == 1:
        use_joblib = False
    else:
        try:
            from joblib import Parallel, delayed, parallel_config
            use_joblib = True
        except ImportError:
            warn("Cannot load 'joblib'. Falling to sequential.")
            use_joblib = False

    # If code is run from IPython, we can use a larger number of refresh per
    # second. Otherwise, let's use a smaller number of refresh per seconds to
    # avoid visual glitches
    if refresh_per_second is None:
        try:
            get_ipython()
            refresh_per_second = 10
        except NameError:
            refresh_per_second = 0.5

    if not args and not kwargs:
        raise ValueError("Must specify arguments.")

    # Get arguments
    args = list(
        zip(
            *args,
            map(lambda vs: dict(zip(kwargs, vs)), zip(
                *kwargs.values())) if kwargs else its.repeat({})))

    # Get number of total runs
    n_runs = len(args)

    # Initialize status
    status = Buffer([0] * n_runs, 'f')

    # Initialize stop
    stop = Buffer([False] * n_runs, '?')

    # Initialize proc_status:
    # 0: Not started
    # 1: Started
    # 2: Completed
    proc_status = Buffer([0] * n_runs, 'b')

    # Get buffers
    buffers = dict(
        its.starmap(lambda name, dtype: (name, Buffer([0] * n_runs, dtype)),
                    buffers))

    # Initialize timer
    def timer_helper():
        for idx in range(len(stop)):
            stop[idx] = True

    timer = Timer(TIMEOUT_MAX if timeout is None else timeout, timer_helper)
    timer.start()

    # Update progressbar
    def update_progress():
        # Set format for progress bar
        bar_format = "{desc}: {percentage:3.0f}% |{bar}| "
        bar_format += "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        # Initialize map of progress bars
        pbars = {}

        # While there are active procs
        while any(map(lambda x: x < 2, proc_status)):

            # Add a progress bar for all the active procs without it
            for idx in map(
                    op.itemgetter(0),
                    filter(lambda x: x[1] == 1 and x[0] not in pbars,
                           enumerate(proc_status))):
                pbars[idx] = tqdm(total=1,
                                  bar_format=bar_format,
                                  leave=leave,
                                  desc='{} [{}/{}]'.format(
                                      description, idx + 1, len(proc_status)))

            # Update all active progress bars
            for idx, pbar in pbars.items():
                pbar.n = status[idx]
                pbar.postfix = text.format(**dict(
                    its.starmap(lambda k, v: (k, v[idx]), buffers.items())))
                pbar.update(0)

            # Remove all progress bars for procs that are completed
            for idx in map(
                    op.itemgetter(0),
                    filter(lambda x: x[1] == 2 and x[0] in pbars,
                           enumerate(proc_status))):
                pbars[idx].close()
                del pbars[idx]

            # Sleep for a bit
            sleep(1 / refresh_per_second)

        # Perform last full update
        for idx, pbar in pbars.items():
            pbars[idx].n = status[idx]
            pbars[idx].update(0)
            pbars[idx].close()

    # Start progressbar
    if verbose >= 2:
        updater = Thread(target=update_progress, daemon=True)
        updater.start()

    def core_(*args, idx, **kwargs):
        # Set proc_status to 'start'
        proc_status[idx] = 1

        # Get results
        results = core(*args, idx=idx, **kwargs)

        # Set proc_status to 'complete'
        proc_status[idx] = 2

        # Return results
        return results

    # Get results (parallel)
    if use_joblib:
        with parallel_config(backend='loky',
                             n_jobs=n_jobs,
                             verbose=(10 if verbose == 1 else False)):
            results = Parallel()(map(
                lambda idx, xs: delayed(core_)(*xs[:-1],
                                               idx=idx,
                                               status=status,
                                               stop=stop,
                                               **xs[-1],
                                               **buffers), range(n_runs), args))

    # Get results (sequential)
    else:
        results = list(
            map(
                lambda idx, xs: core_(*xs[:-1],
                                      idx=idx,
                                      status=status,
                                      stop=stop,
                                      **xs[-1],
                                      **buffers), range(n_runs), args))

    # Close progressbar
    if verbose >= 2:
        updater.join()

    # Close buffers
    status.shm.unlink()
    proc_status.shm.unlink()
    stop.shm.unlink()
    mit.consume(map(lambda buffer: buffer.shm.unlink(), buffers.values()))

    # Stop timer
    timer.cancel()

    # Return results
    return results
