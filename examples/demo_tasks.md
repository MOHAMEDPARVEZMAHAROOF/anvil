# Demo tasks for Anvil

Copy any of these into the web UI or `scripts/demo.py`. Each is chosen to show
the full loop: plan → write → run → fix → green.

## 1. Stats module (quick, ~1 min)
Write a Python module `stats.py` with `mean`, `median`, and `mode` functions,
plus `pytest` tests covering empty input, single values, and even/odd-length lists.

## 2. CSV-to-JSON CLI (~2 min)
Build a tiny CLI `csv2json.py` using `argparse` that converts a CSV file to JSON.
Include `pytest` tests for valid input, a missing file, and malformed CSV.

## 3. LRU cache (~2 min)
Implement an `LRUCache` class in Python with `get`/`put` and capacity eviction,
plus `pytest` tests for eviction order, key updates, and capacity of 1.

## 4. Rate limiter (~3 min, shows a fix iteration)
Write a token-bucket rate limiter in Python with `pytest` tests covering burst
capacity, token refill over time, and thread safety.

## 5. Bug fix from a seed file (shows the diff view)
Seed the sandbox with `buggy.py` containing an off-by-one error in a binary
search, then ask: "Find and fix the bug in buggy.py and add a regression test."
