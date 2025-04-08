import logging
from collections import namedtuple, defaultdict
import time
import contextlib
from typing import Optional, Iterator, Any



class NoOpContext:
    """A dummy context manager that does nothing"""
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class ScopedProfiler:
    Block = namedtuple('Block', ['start_time', 'name'])

    class Stats:
        def __init__(self) -> None:
            self.times = []
            self.calls = 0
            self.children = set()

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        if not self.enabled:
            return
        self._stats: dict[tuple[str], self.Stats] = defaultdict(self.Stats)
        self._stats[()]  # instantiate root
        self._blocks: list[self.Block] = []

    def begin_block(self, name: str) -> None:
        if not self.enabled:
            return

        self._blocks.append(self.Block(start_time=time.time(), name=name))

    def end_block(self, name: str) -> None:
        if not self.enabled:
            return

        #assert len(self._blocks) > 0
        #assert name == self._blocks[-1].name, f'end_block failed: name={name} current_block={self._blocks[-1].name}'
        full_name = tuple(x.name for x in self._blocks)
        self._stats[full_name].times.append(time.time() - self._blocks[-1].start_time)
        self._stats[full_name].calls += 1
        self._blocks.pop()

    def no_op_block(self, name):
        return NoOpContext()  # Returns a dummy context manager

    @contextlib.contextmanager
    def real_block(self, name):
        self.begin_block(name)
        try:
            yield
        finally:
            self.end_block(name)

    # ✅ Dynamically choose the correct method
    def block(self, name: str):
        if not self.enabled:
            return self.no_op_block(name)  # Use dummy context manager
        else:
            return self.real_block(name)  # Use actual profiling context

    def __call__(self, name: str) -> contextlib._GeneratorContextManager:
        return self.block(name)

    def wrap_iterator(self, name: str, it: Iterator[Any]) -> Iterator[Any]:
        while True:
            try:
                with self.block(name):
                    item = next(it)
                yield item
            except StopIteration:
                break

    def print_timings(self, *, only_sum: bool = False) -> None:
        if not self.enabled:
            return

        if not self._stats:
            return

        for k, v in self._stats.items():
            if k:
                parent = k[:-1]
                self._stats[parent].children.add(k)

        output = ['Stats (milliseconds):']

        def print_stats(k: tuple[str], offset: str) -> tuple[float, list[str]]:
            v = self._stats[k]
            total_time = 0
            lines = []

            if k:
                total_time = sum(v.times)
                calls = f' x{v.calls}' if v.calls > 1 else ''
                if only_sum or v.calls == 1:
                    lines.append(f'{offset}{k[-1]}{calls}: sum={total_time:.2f} sec')
                else:
                    v_ms = sorted([x * 1000 for x in v.times])
                    cnt = len(v_ms)
                    p_0 = v_ms[0]
                    p_50 = v_ms[int(cnt * 0.5)]
                    p_95 = v_ms[int(cnt * 0.95)]
                    p_100 = v_ms[-1]

                    lines.append(
                        f'{offset}{k[-1]}{calls}: p0={p_0:.1f} p50={p_50:.1f} p95={p_95:.1f}'
                        f' max={p_100:.1f} sum={total_time:.2f} sec'
                    )

            other_time = total_time
            for child in sorted(v.children, key=lambda x: sum(self._stats[x].times), reverse=True):
                child_time, child_lines = print_stats(child, offset + '| ')
                other_time -= child_time
                lines.extend(child_lines)

            if k and v.children:
                lines.append(f'{offset}| other____: {other_time:.2f} sec')

            return total_time, lines

        _, stats_lines = print_stats((), '')
        output.extend(stats_lines)
        print('\n'.join(output))
