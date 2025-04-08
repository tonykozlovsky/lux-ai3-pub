# Copyright (c) Facebook, Inc. and its affiliates.
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
"""Naive profiling using timeit. (Used in MonoBeast.)"""

import collections
import timeit


class Timings:
    """Not thread-safe."""

    def __init__(self, name):
        self._sums = collections.defaultdict(float)
        self._means = collections.defaultdict(float)
        self._vars = collections.defaultdict(float)
        self._counts = collections.defaultdict(int)
        self.last_time = collections.defaultdict(float)
        self.name = name
        #self.reset()

    def reset(self, name):
        self.last_time[name] = timeit.default_timer()

    def time(self, name):
        """Save an update for event `name`.

        Nerd alarm: We could just store a
            collections.defaultdict(list)
        and compute means and standard deviations at the end. But thanks to the
        clever math in Sutton-Barto
        (http://www.incompleteideas.net/book/first/ebook/node19.html) and
        https://math.stackexchange.com/a/103025/5051 we can update both the
        means and the stds online. O(1) FTW!
        """
        now = timeit.default_timer()
        x = now - self.last_time[name]
        self.last_time[name] = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1
        self._sums[name] =self._means[name] * self._counts[name]

    def means(self):
        return self._means

    def sums(self):
        return self._sums

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        sums = self.sums()
        total = sum(sums.values()) if self.name not in sums.keys() else sums[self.name]

        result = prefix
        result += "\n\n"
        for k in sorted(sums, key=sums.get, reverse=True):
            result += f"\n    {k}:     {1000 * means[k]:.2f}ms ({100 * sums[k] / total:.2f}%)"
        result += f"\nTotal: {1000 * total:.2f}ms\n\n"
        return result
