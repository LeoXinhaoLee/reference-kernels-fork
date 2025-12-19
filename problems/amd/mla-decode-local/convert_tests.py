#!/usr/bin/env python3
"""
Simple script to convert tests and benchmarks from task.yml into txt files
that comply with the format expected by eval.py (lines 74-102).
"""

import yaml
from pathlib import Path


def convert_dict_to_line(test_dict: dict) -> str:
    """
    Convert a test dictionary to a line in the expected format.
    Format: key1:value1;key2:value2;key3:value3
    """
    parts = []
    for key, value in test_dict.items():
        parts.append(f"{key}:{value}")
    return ";".join(parts)


def main():
    # Read task.yml
    task_yml_path = Path(__file__).parent / "task.yml"
    with open(task_yml_path, "r") as f:
        task_data = yaml.safe_load(f)
    
    # Create inputs directory if it doesn't exist
    inputs_dir = Path(__file__).parent / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    
    # Convert tests
    if "tests" in task_data:
        test_lines = []
        for test_dict in task_data["tests"]:
            line = convert_dict_to_line(test_dict)
            test_lines.append(line)
        
        test_file = inputs_dir / "test_cases.txt"
        with open(test_file, "w") as f:
            f.write("\n".join(test_lines) + "\n")
        print(f"Created {test_file} with {len(test_lines)} test cases")
    
    # Convert benchmarks
    if "benchmarks" in task_data:
        benchmark_lines = []
        for benchmark_dict in task_data["benchmarks"]:
            line = convert_dict_to_line(benchmark_dict)
            benchmark_lines.append(line)
        
        benchmark_file = inputs_dir / "benchmark_cases.txt"
        with open(benchmark_file, "w") as f:
            f.write("\n".join(benchmark_lines) + "\n")
        print(f"Created {benchmark_file} with {len(benchmark_lines)} benchmark cases")


if __name__ == "__main__":
    main()
