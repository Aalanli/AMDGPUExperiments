from dataclasses import dataclass
from typing import List, Any, Dict, Tuple, Callable, Optional
from . import do_bench

@dataclass
class BenchData:
    x_vals: List[Any]
    x_name: str
    y_name: str
    kwargs: Dict[str, Any]
    data: Dict[str, Tuple[List[float], List[float], List[float]]]  # [t_min, t_avg, t_max]

    def show_plot(self, show=True, save_path=None, figsize=None, title=None):
        from matplotlib import pyplot as plt

        if all(isinstance(x, (float, int)) for x in self.x_vals):
            x_vals = self.x_vals
        else:
            x_vals = range(1, len(self.x_vals) + 1)

        plt.figure(figsize=figsize)
        ax = plt.subplot()
        for name, (t_min, t_avg, t_max) in self.data.items():
            p = ax.plot(x_vals, t_avg, label=name)
            color = p[0].get_color()
            ax.fill_between(x_vals, t_min, t_max, alpha=0.15, color=color)
        ax.legend()
        ax.set_xlabel(self.x_name)
        ax.set_ylabel(self.y_name)
        if title is not None:
            ax.set_title(title)
        ax.set_xticks(ticks=x_vals, labels=[str(x) for x in self.x_vals])
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        return self

    def to_dataframe(self):
        import pandas as pd

        columns = list(self.data.keys())
        df = pd.DataFrame(columns=columns, index=self.x_vals)
        for n in columns:
            df[n] = self.data[n][1]  # get t_avg
        return df

    def print_data(self):
        print(self.to_dataframe())


class Bench:
    def __init__(self, x_vals: List[Any], x_name: str, **kwargs):
        self.x_vals = x_vals
        self.x_name = x_name
        self.y_name = 'ms'
        self.byte_fn = None

        self.kwargs: Dict[str, Any] = kwargs
        self.bench_fns: List[Tuple[str, Callable]] = []
        self.bench_data: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    def measure_flops(self, byte_fn: Callable[[Any], int]):
        """
        set a function that takes in the config, and the current x_val and returns the number of bytes
        """
        self.byte_fn = byte_fn
        self.y_name = 'TFLOP/s'

    def bench(self, fn: Callable[[Any], Callable[[], Any]], name: Optional[str] = None):
        """
        add a function that takes in the config and int and returns a function to be benchmarked
        to the list of functions to be benchmarked.
        If the name argument is None, the the name for this particular line is fn.__name__
        """
        if name is None:
            if hasattr(fn, '__name__'):
                name = fn.__name__
            else:
                raise ValueError("cannot get name of function")
        self.bench_fns.append((name, fn))
        return self

    def run(self):
        """
        run all the functions that needs to be benchmarked, returning BenchData representing
        the collected results
        """
        for i in self.x_vals:
            for name, fn in self.bench_fns:

                if name not in self.bench_data:
                    self.bench_data[name] = ([], [], [])
                t_min, t_avg, t_max = self.bench_data[name]

                bench_fn = fn(i, **self.kwargs)
                lo, avg, hi = do_bench(bench_fn, quantiles=[0.2, 0.5, 0.8])
                if self.byte_fn is not None:
                    lo = self.byte_fn(i, **self.kwargs) * 1e-12 / (lo * 1e-3)
                    avg = self.byte_fn(i, **self.kwargs) * 1e-12 / (avg * 1e-3)
                    hi = self.byte_fn(i, **self.kwargs) * 1e-12 / (hi * 1e-3)
                t_min.append(lo)
                t_avg.append(avg)
                t_max.append(hi)
        return BenchData(self.x_vals, self.x_name, self.y_name, self.kwargs, self.bench_data)
