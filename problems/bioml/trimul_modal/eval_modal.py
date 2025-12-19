import base64
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any, Optional
import socket

import torch.cuda

from utils import set_seed
try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from reference import check_implementation, generate_input

import pathlib

# module_path = pathlib.Path(os.environ["CUSTOM_KERNEL_MODULE"]).resolve()
module_path = pathlib.Path(os.environ["CUSTOM_KERNEL_MODULE"])

MODAL_REMOTE_DIR = "/root/project"


class PopcornOutput:
    # def __init__(self, fd: int):
    #     self.file = os.fdopen(fd, 'w')
    #     os.set_inheritable(fd, False)
    #     return
    def __init__(self, fname: str):
        self.file = open(fname, 'w')
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)

    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def _combine(a: int, b: int) -> int:
    # combine two integers into one:
    # we need this to generate a secret seed based on the test-level seed and
    # the global secret seed.
    # the test-level seeds are public knowledge, and typically relatively small numbers,
    # so we need to make sure they don't provide any useful info for the full seed.
    # This Cantor construction ensures that if the secret seed is a large number,
    # then so is the overall seed.
    return int(a + (a + b) * (a + b + 1) // 2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                pass

            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)

    return tests


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst))


def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def wrap_check_implementation(data, submission_output):
    # Old version returned just a single string, new version
    # returns (bool, str); this function ensures compatibility with old
    # problem definitions.
    result = check_implementation(data, submission_output)
    if isinstance(result, tuple):
        return result
    else:
        return not bool(result), result


def _run_single_test(test: TestCase):
    """
    Runs a single test case. Do not call directly
    """
    # from submission import custom_kernel
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        module_path.stem, os.path.join(_MODAL_REMOTE_DIR, module_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    custom_kernel = module.custom_kernel

    data = generate_input(**test.args)
    torch.cuda.synchronize()
    submission_output = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()
    return wrap_check_implementation(data, submission_output)


# =========================
# Modal adaptation (only)
# =========================
try:
    import modal
except Exception:
    modal = None

_MODAL_APP = None
_MODAL_IMAGE = None
_MODAL_REMOTE_DIR = None

def _modal_setup():
    """
    Minimal Modal setup. Uses a CUDA base image and mounts the current repo
    so `submission`, `reference`, `utils`, etc. import exactly as-is.
    """
    global _MODAL_APP, _MODAL_IMAGE, _MODAL_REMOTE_DIR
    if _MODAL_APP is not None:
        return

    if modal is None:
        raise RuntimeError("Modal is not available (could not import modal).")

    ## @xh: follow KB
    # cuda_version = "12.8.0"  # should be no greater than host CUDA version
    # flavor = "devel"  #  includes full CUDA toolkit
    # operating_sys = "ubuntu22.04"
    # tag = f"{cuda_version}-{flavor}-{operating_sys}"
    # cuda_image = f"nvidia/cuda:{tag}"
    # _MODAL_APP = modal.App("trimul-test")
    # _MODAL_IMAGE = (
    #     modal.Image.from_registry(cuda_image, add_python="3.10")
    #     .apt_install(
    #         "git",
    #         "gcc-10",
    #         "g++-10",
    #         "clang" # note i skip a step
    #     )
    #     .pip_install_from_requirements("/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/code/xh/KernelBench-fork/requirements.txt")
    #     # Ensure current project code is available in the remote container.
    #     .env({"PYTHONPATH": "/root/project"})
    #     .add_local_dir(".", remote_path="/root/project")
    #     # .add_local_python_source("bioml/trimul_modal")
    # )
    
    ### @xh: follow Discord Manager: https://github.com/gpu-mode/discord-cluster-manager/blob/c3a50838e982f34fceb329cf1d77a29d503333f2/src/runners/modal_runner.py
    _MODAL_REMOTE_DIR = "/root/project"
    _MODAL_APP = modal.App("discord-bot-runner")
    cuda_version = "12.8.0"
    flavor = "devel"
    operating_sys = "ubuntu24.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"

    # Move this to another file later:
    _MODAL_IMAGE = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
        .apt_install(
            "git",
            "gcc-13",
            "g++-13",
            "clang-18",
        )
        .pip_install(
            "ninja~=1.11",
            "wheel~=0.45",
            "requests~=2.32.4",
            "packaging~=25.0",
            "numpy~=2.3",
            "pytest",
            "PyYAML",
        )
        .pip_install(
            "torch>=2.7.0,<2.8.0",
            "torchvision~=0.22",
            "torchaudio>=2.7.0,<2.8.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        # other frameworks
        .pip_install(
            "jax[cuda12]==0.5.3",  # 0.6 want's cudnn 9.8 in conflict with torch 2.7
            "jax2torch==0.0.7",
            "tinygrad~=0.10",
        )
        # nvidia cuda packages
        .pip_install(
            "nvidia-cupynumeric~=25.3",
            "nvidia-cutlass-dsl~=4.0",
            "cuda-core[cu12]~=0.3",
            "cuda-python[all]==12.8",
            # "nvmath-python[cu12]~=0.4",
            # "numba-cuda[cu12]~=0.15",
        )
        # Ensure current project code is available in the remote container.
        
        .env({"PYTHONPATH": _MODAL_REMOTE_DIR})
        .add_local_dir(".", remote_path=_MODAL_REMOTE_DIR)
    )
    

def _modal_gpu_from_env():
    """
    Pick GPU from env with minimal mapping.
    Supported examples: L40S, H100, A100, L4, T4, A10G.
    """
    gpu_type = os.environ.get("MODAL_GPU_TYPE", "H100!")
    return gpu_type

def _modal_timeout_s_from_env(default_s: int = 1800) -> int:
    try:
        return int(os.getenv("MODAL_TIMEOUT_S", str(default_s)))
    except Exception:
        return default_s

# Remote worker functions (keep signatures pickle-friendly)
def _modal_run_single_test(test_args: dict, test_spec: str):
    test = TestCase(args=test_args, spec=test_spec)
    return _run_single_test(test)

def _modal_run_single_benchmark(test_args: dict, test_spec: str, recheck: bool, max_repeats: int, max_time_ns: float):
    test = TestCase(args=test_args, spec=test_spec)
    return _run_single_benchmark(test, recheck, max_repeats, max_time_ns)

# We define the Modal functions lazily so imports work in both local and modal contexts.
_MODAL_FNS_DEFINED = False
def _define_modal_functions():
    global _MODAL_FNS_DEFINED
    if _MODAL_FNS_DEFINED:
        return
    _modal_setup()

    @_MODAL_APP.function(image=_MODAL_IMAGE, gpu=_modal_gpu_from_env(), timeout=_modal_timeout_s_from_env(), serialized=True)
    def modal_run_testing(tests_payload: list[tuple[dict, str]]):
        # returns {"device": str, "passed": bool, "results": [(spec, good, message), ...]}
        hostname = socket.getfqdn()  # socket.gethostname()
        device = 0
        device_name = torch.cuda.get_device_name(device=device)
        results = []
        passed = True
        for (args, spec) in tests_payload:
            good, message = _modal_run_single_test(args, spec)
            results.append((spec, good, message))
            if not good:
                passed = False
        return {"hostname": hostname, "device": device_name, "passed": passed, "results": results}

    @_MODAL_APP.function(image=_MODAL_IMAGE, gpu=_modal_gpu_from_env(), timeout=_modal_timeout_s_from_env(), serialized=True)
    def modal_run_benchmarking(tests_payload: list[tuple[dict, str]]):
        # returns {"device": str, "passed": bool, "results": [(spec, ok, stats_dict_or_error), ...]}
        hostname = socket.getfqdn()  # socket.gethostname()
        device = 0
        device_name = torch.cuda.get_device_name(device=device)

        if len(tests_payload) == 0:
            return {"hostname": hostname, "device": device_name, "passed": True, "results": []}

        # warm up (mirror original behavior)
        first_args, first_spec = tests_payload[0]
        _modal_run_single_benchmark(first_args, first_spec, False, 100, 10e7)

        results = []
        passed = True
        for (args, spec) in tests_payload:
            res = _modal_run_single_benchmark(args, spec, False, 100, 10e9)
            if isinstance(res, Stats):
                stats_dict = {f.name: getattr(res, f.name) for f in dataclasses.fields(Stats)}
                results.append((spec, True, stats_dict))
            else:
                passed = False
                results.append((spec, False, res))
        return {"hostname": hostname, "device": device_name, "passed": passed, "results": results}

    @_MODAL_APP.function(image=_MODAL_IMAGE, gpu=_modal_gpu_from_env(), timeout=_modal_timeout_s_from_env(), serialized=True)
    def modal_run_leaderboard(tests_payload: list[tuple[dict, str]]):
        # returns {"device": str, "passed": bool, "results": [(spec, ok, stats_dict_or_error), ...]}
        hostname = socket.getfqdn()  # socket.gethostname()
        device = 0
        device_name = torch.cuda.get_device_name(device=device)

        if len(tests_payload) == 0:
            return {"hostname": hostname, "device": device_name, "passed": True, "results": []}

        # warmup (mirror original behavior)
        first_args, first_spec = tests_payload[0]
        _modal_run_single_benchmark(first_args, first_spec, False, 100, 1e7)

        results = []
        passed = True
        for (args, spec) in tests_payload:
            res = _modal_run_single_benchmark(args, spec, True, 100, 30e9)
            if isinstance(res, Stats):
                stats_dict = {f.name: getattr(res, f.name) for f in dataclasses.fields(Stats)}
                results.append((spec, True, stats_dict))
            else:
                passed = False
                results.append((spec, False, str(res)))
                break
        return {"hostname": hostname, "device": device_name, "passed": passed, "results": results}

    # stash on module globals for use below
    globals()["modal_run_testing"] = modal_run_testing
    globals()["modal_run_benchmarking"] = modal_run_benchmarking
    globals()["modal_run_leaderboard"] = modal_run_leaderboard
    _MODAL_FNS_DEFINED = True
# =========================
# End Modal adaptation
# =========================


def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    """
    Runs a single test in another process.
    """
    # kept for compatibility (unused after adaptation)
    return pool.apply(_run_single_test, (test,))


def run_testing(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    _define_modal_functions()
    tests_payload = [(t.args, t.spec) for t in tests]

    with _MODAL_APP.run():
        out = modal_run_testing.remote(tests_payload)

    logger.log("device", out["device"])
    logger.log("test-count", len(tests))
    passed = True
    for idx, (spec, good, message) in enumerate(out["results"]):
        logger.log(f"test.{idx}.spec", spec)
        if not good:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", message)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")
            if message:
                logger.log(f"test.{idx}.message", message)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def _run_single_benchmark(test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """
    # from submission import custom_kernel
    global _MODAL_REMOTE_DIR
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        module_path.stem, os.path.join(_MODAL_REMOTE_DIR, module_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    custom_kernel = module.custom_kernel

    durations = []
    # generate input data once
    data = generate_input(**test.args)
    check_copy = _clone_data(data)
    #  first, one obligatory correctness check
    output = custom_kernel(data)
    good, message = wrap_check_implementation(check_copy, output)
    if not good:
        return message

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            # ensure we use a different seed for every benchmark
            if "seed" in test.args:
                test.args["seed"] += 13

            data = generate_input(**test.args)
            check_copy = _clone_data(data)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) * 1e6  # Convert ms to ns

        if recheck:
            good, message = check_implementation(check_copy, output)
            if not good:
                return message

        del output
        durations.append(duration)

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start_time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if stats.err / stats.mean < 0.001 or stats.mean * stats.runs > max_time_ns or total_bm_duration > 120e9:
                break

    return calculate_stats(durations)


def run_single_benchmark(pool: multiprocessing.Pool, test: TestCase, recheck: bool, max_repeats: int,
                         max_time_ns: float):
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param pool: Process on which the benchmark will be launched.
    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    # kept for compatibility (unused after adaptation)
    return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns))


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    _define_modal_functions()
    tests_payload = [(t.args, t.spec) for t in tests]

    with _MODAL_APP.run():
        out = modal_run_benchmarking.remote(tests_payload)

    passed = True
    logger.log("device", out["device"])
    logger.log("benchmark-count", len(tests))
    for idx, (spec, ok, payload) in enumerate(out["results"]):
        logger.log(f"benchmark.{idx}.spec", spec)
        if ok:
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", payload[field.name])
        else:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", payload)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112



def main():
    # fd = os.getenv("POPCORN_FD")
    # if not fd:
    #     return 111

    # if len(sys.argv) < 3:
    #     return 2
    # mode = sys.argv[1]
    # seed = os.getenv("POPCORN_SEED")
    # mode = "test"

    # mode = "test"
    mode = "leaderboard"

    seed = None
    os.unsetenv("POPCORN_SEED")
    seed = int(seed) if seed else None
    set_seed(seed or 42)
    # tests = get_test_cases(sys.argv[2], seed)
    
    # out_dir = "out_best/test3"
    out_dir = sys.argv[1]
    
    os.makedirs(out_dir, exist_ok=True)
    fd = f"{out_dir}/{mode}.out"
    tests = get_test_cases(f"inputs/{mode}_cases.txt", seed) if mode != "leaderboard" else get_test_cases(
        f"inputs/benchmark_cases.txt", seed
    )

    with PopcornOutput(fd) as logger:
        import multiprocessing
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)

            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)

            if mode == "leaderboard":
                _define_modal_functions()
                tests_payload = [(t.args, t.spec) for t in tests]
                with _MODAL_APP.run():
                    out = modal_run_leaderboard.remote(tests_payload)

                logger.log("hostname", out["hostname"])                
                logger.log("device", out["device"])
                logger.log("benchmark-count", len(tests))
                passed = True
                for i, (spec, ok, payload) in enumerate(out["results"]):
                    logger.log(f"benchmark.{i}.spec", spec)
                    if ok:
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{i}.{field.name}", payload[field.name])
                    else:
                        passed = False
                        logger.log(f"benchmark.{i}.status", "fail")
                        logger.log(f"benchmark.{i}.error", payload)
                        break

                logger.log("check", "pass" if passed else "fail")

            elif mode == "profile":
                # run_profiling(logger, tests)
                raise NotImplementedError

            else:
                # TODO: Implement script mode
                return 2


if __name__ == "__main__":
    sys.exit(main())
