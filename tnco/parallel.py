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
"""Parallelization utilities."""

import itertools as its
from multiprocessing.shared_memory import SharedMemory
from struct import calcsize, pack_into, unpack_from
from threading import TIMEOUT_MAX, Thread, Timer
from time import sleep
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from warnings import warn

import more_itertools as mit
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn

__all__ = ['Buffer', 'Parallel']


class Buffer:
    """A shared memory buffer.

    Buffer for shared memory.

    Args:
        sequence_or_size: The initial sequence of data or the size of the
            buffer.
        dtype: The data type of the elements in the buffer. Must be a single
            character format string compatible with ``struct`` module.

    Examples:
        >>> from tnco.parallel import Buffer
        >>> b = Buffer([1, 2, 3], dtype='i')
        >>> len(b)
        3
        >>> b[0]
        1
    """

    def __init__(self,
                 sequence_or_size: Union[Iterable[Any], int],
                 dtype: str = 'q') -> None:

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

    def __len__(self) -> int:
        return self.shm.size // calcsize(self.dtype)

    def __repr__(self) -> str:
        return repr(list(self))

    def __getitem__(self, pos: int) -> Any:
        if pos >= len(self):
            raise IndexError("list index out of range")
        return unpack_from(self.dtype, self.shm.buf,
                           pos * calcsize(self.dtype))[0]

    def __setitem__(self, pos: int, value: Any) -> None:
        if pos >= len(self):
            raise IndexError("list index out of range")
        pack_into(self.dtype, self.shm.buf, pos * calcsize(self.dtype), value)

    @property
    def dtype(self) -> str:
        return self._dtype


def Parallel(core: Callable[..., Any],
             *args: Any,
             description: str = "Processing...",
             text: str = "",
             n_jobs: int = -1,
             timeout: Optional[float] = None,
             verbose: int = False,
             buffers: Iterable[Tuple[str, str]] = (),
             refresh_per_second: Optional[float] = None,
             leave: bool = False,
             **kwargs: Any) -> Any:
    """Parallelize ``core``.

    Executes a function in parallel.

    Args:
        core: Function to parallelize. The core must have the keyword
            parameters ``idx`` (which is used to identify the process),
            ``status`` (which is used by the process to inform the main process
            of its status), and ``stop`` (which instruct the process to stop if
            ``True``). All the arguments ``args`` and keyword arguments
            ``kwargs`` are passed to the ``core``.
        description: Description to use for the progress bar.
        text: ``rich.progress.TextColumn`` formatting string for additional
            text.
        n_jobs: Number of processes to use. By default, all available cores are
            used. If ``n_jobs`` is a positive number, ``n_jobs`` processes will
            be used. If ``n_jobs`` is negative, ``n_cpus + n_jobs + 1`` will be
            used. If ``n_jobs`` is zero, it will raise a ``ValueError``. (See:
            ``joblib.Parallel``).
        timeout: If provided, sets the ``stop`` variables for all processes to
            ``True`` after ``timeout`` seconds have passed.
        verbose: If ``True``, prints verbose output.
        buffers: Buffers used to communicate between processes. Each buffer is
            of the form ``(buffer_name, type)``, with ``buffer_name`` being the
            name of the buffer and ``type`` its type (See:
            ``tnco.parallel.Buffer``). Buffers are then passed as keyword
            arguments to ``core``.
        refresh_per_second: Number of refresh updates per second for the
            progress bar.
        leave: If ``True``, progress bars are kept after completion.

    Returns:
        The results of the parallel execution.

    Examples:
        def example(x, *, idx, status, stop):
            from time import sleep

            # Exit if stop is already set
            if stop[idx]:
                return 'stopped'

            # Initialize the status. Status must be a number between
            # 0 (0%) and 1 (100%)
            status[idx] = 0

            for n in range(100):
                sleep(0.05)
                x += x
                status[idx] = n/100

                # If stop is set, break loop
                if stop[idx]:
                    break

            # Return results
            return x

        Parallel(example, x=range(10), n_jobs=4, verbose=3, timeout=2)
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
    stop = Buffer([False] * n_runs, 'b')

    # Initialize completed
    completed = Buffer([0] * n_runs, '?')

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

    with Progress(TextColumn('[blue][{task.fields[idx]}/' + str(len(args)) +
                             ']'),
                  *Progress.get_default_columns(),
                  TimeElapsedColumn(),
                  TextColumn(text),
                  console=Console(stderr=True),
                  disable=(verbose <= 1),
                  auto_refresh=False,
                  transient=not leave) as progress:

        # Update progressbar
        def update_progress():
            # Initialize tasks
            tasks = {}

            def get_task(idx):
                if idx not in tasks:
                    tasks[idx] = progress.add_task(
                        description,
                        total=1,
                        idx=idx + 1,
                        **dict(zip(buffers, its.repeat(float('nan')))))
                return tasks.get(idx)

            def update_task(idx, completed):
                if (task := get_task(idx)) is not None:
                    progress.update(task,
                                    completed=completed,
                                    idx=idx + 1,
                                    **dict(
                                        map(lambda k: (k, buffers[k][idx]),
                                            buffers)))

            def remove_task(idx):
                if not leave and tasks.get(idx) is not None:
                    progress.remove_task(tasks.pop(idx))
                    tasks[idx] = None

            def full_update():
                for idx_, (c_, s_) in enumerate(zip(completed, status)):
                    if s_ > 0:
                        update_task(idx_, status[idx_])
                    if c_:
                        remove_task(idx_)

                # Refresh progress
                progress.refresh()

            # While there are active tasks
            while any(not c_ for c_ in completed):
                # Perform full update
                full_update()

                # Sleep for a bit
                sleep(1 / refresh_per_second)

            # Perform last full update
            full_update()

            # Last checks
            assert not timer.is_alive() or not tasks or (
                len(tasks) == n_runs and sorted(tasks) == list(range(n_runs)))
            assert leave or all(v_ is None for v_ in tasks.values())

        # Start progressbar
        if verbose >= 2:
            updater = Thread(target=update_progress, daemon=True)
            updater.start()

        def core_(*args, idx, **kwargs):
            # Get results
            results = core(*args, idx=idx, **kwargs)

            # Set completed to true
            completed[idx] = True

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
                                                   **buffers), range(n_runs),
                    args))

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
    completed.shm.unlink()
    stop.shm.unlink()
    mit.consume(map(lambda buffer: buffer.shm.unlink(), buffers.values()))

    # Stop timer
    timer.cancel()

    # Return results
    return results
