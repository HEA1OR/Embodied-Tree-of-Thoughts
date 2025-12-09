#!/usr/bin/env python3
# entry.py
import sys
import subprocess
from pathlib import Path

TASK_MAP = {
    "0": "disturbance",
    "1": "short_1",
    "2": "short_2",
    "3": "short_3",
    "4": "short_4",
    "5": "long_1",
    "6": "long_2",
    "7": "long_3",
}

DESC_MAP = {   
    "0": "Pick up a tennis ball.",
    "1": "Open the door of the microwave oven.",
    "2": "Reorient the pen and drop it into a holder.",
    "3": "Pick up the holder horizontally or vertically.",
    "4": "Close the drawer.",
    "5": "Reorient the pen and drop it into a holder.",
    "6": "Put the apple and the pen holder on the drawer, and the apple should be placed in the pen holder.",
    "7": "Put the apple and the tennis ball in either the drawer or the pen holder, together or separately.",
}

PLAN_MEANING = {
    "disturbance": {0: "pick up tennis 1 before disturbance", 1: "pick up tennis 2 before disturbance", 2: "pick up tennis 1 after disturbance", 3: "pick up tennis 2 after disturbance"},
    "short_1": {0: "(feasible) tennis>>safe, open the door", 1: "only open the door"},
    "short_2": {0: "pen>>holder_1", 1: "(feasible) pen>>holder_2"},
    "short_3": {0: "(feasible)vertical", 1: "horizontal"},
    "short_4": {0: "close drawer", 1: "(feasible)toy>>safe and close drawer"},
    "long_1": {0:"pen>>holder_1;", 1:"pen>>holder_2", 2:"(feasible) apple>>desk; pen>>holder_2"},
    "long_2": {0:"(feasible) holder_1>>drawer,apple>>holder_1;", 1:"apple>>holder_1,holder_1>>drawer"},
    "long_3": {0:"(feasible) apple>>holder_1, tennis1>>safe, open drawer, tennis1>>drawer", 1:"#1 apple>>holder_1,tennis1>>holder_1", 2: "apple>>holder_1,tennis1>>drawer", 3: "apple>>drawer"},
}


def choose_task() -> str:
    print("Tasks:")
    for k, v in TASK_MAP.items():
        print(f"  {k}  ->  {v}   ({DESC_MAP[k]})")
    while True:
        sel = input("Please input task number (0/1/2/3/4/5/6/7): ").strip()
        if sel in TASK_MAP:
            return TASK_MAP[sel]
        print("Invalid input, please try again!")


def choose_plan(task_key: str) -> int:
    meanings = PLAN_MEANING[task_key]
    print(f"\nThe plan meanings of task {task_key}: ")
    for p, desc in meanings.items():
        print(f"  plan {p}  ->  {desc}")
    while True:
        try:
            p = int(input(f"Please choose plan number ({'/'.join(map(str, meanings.keys()))}) for task {task_key}: ").strip())
            if p in meanings:
                return p
        except ValueError:
            pass
        print("Invalid input, please try again!")


def main():
    task_key = choose_task()
    plan = choose_plan(task_key)

    script_path = Path(__file__).parent / f"{task_key}.py"
    if not script_path.exists():
        print(f"Script {script_path} not found, please check the directory structure!")
        sys.exit(1)

    cmd = [sys.executable, str(script_path), "--plan", str(plan)]
    print(f"\nLaunch: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()