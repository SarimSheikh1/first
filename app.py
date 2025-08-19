import random
import datetime as dt
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable

# =========================
# Calendar helpers
# =========================
MONTH_LENGTHS_COMMON = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_LENGTHS_LEAP   = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def _month_lengths(include_leap: bool) -> List[int]:
    return MONTH_LENGTHS_LEAP if include_leap else MONTH_LENGTHS_COMMON

# =========================
# Birthday generators
# =========================
def random_birthday(year: int = 2001, include_leap: bool = False) -> dt.date:
    """
    Return a random date from that calendar year.
    If include_leap is False, Feb 29 is excluded (365-day model).
    """
    months = _month_lengths(include_leap)
    # choose a month weighted by its length, then a day in that month
    month = random.choices(range(1, 13), weights=months, k=1)[0]
    day = random.randint(1, months[month - 1])
    return dt.date(year, month, day)

def generate_group(n: int, include_leap: bool = False) -> List[dt.date]:
    """Return a list of n random birthdays."""
    return [random_birthday(include_leap=include_leap) for _ in range(n)]

# =========================
# Duplicate detection
# =========================
def _md_pairs(dates: Iterable[dt.date]) -> List[Tuple[int, int]]:
    return [(d.month, d.day) for d in dates]

def has_shared_birthday(dates: Iterable[dt.date]) -> bool:
    """True if any two birthdays (month, day) match in the list."""
    md = _md_pairs(dates)
    return len(md) != len(set(md))

def find_shared_birthdays(dates: Iterable[dt.date]) -> Dict[Tuple[int, int], int]:
    """Return a dict of {(month, day): count} for days that appear more than once."""
    counts = Counter(_md_pairs(dates))
    return {k: v for k, v in counts.items() if v > 1}

# =========================
# Probability calculations
# =========================
def exact_probability(n: int, days: int = 365) -> float:
    """
    Exact probability that at least two people share a birthday
    in a 'days'-day year. Uses: 1 - prod((days - i)/days for i=0..n-1)
    """
    if n <= 1:
        return 0.0
    if n > days:
        return 1.0
    p_all_diff = 1.0
    for i in range(n):
        p_all_diff *= (days - i) / days
    return 1.0 - p_all_diff

def simulated_probability(n: int, trials: int = 100_000, include_leap: bool = False) -> float:
    """Estimate probability by Monte Carlo simulation."""
    hits = 0
    for _ in range(trials):
        if has_shared_birthday(generate_group(n, include_leap)):
            hits += 1
    return hits / trials

def smallest_group_for(target: float = 0.5, days: int = 365) -> int:
    """Smallest n with probability >= target."""
    n = 1
    while exact_probability(n, days) < target:
        n += 1
    return n

def expected_matching_pairs(n: int, days: int = 365) -> float:
    """E[# of matching pairs] = C(n, 2) / days."""
    return n * (n - 1) / (2 * days)

# =========================
# Exploration utilities
# =========================
def probability_table(max_n: int = 50, trials: int = 100_000,
                      include_leap: bool = False, days: int | None = None) -> None:
    """
    Print a table of exact and simulated probabilities for n = 2..max_n.
    To use the leap-year model, set include_leap=True (days will be 366).
    """
    if days is None:
        days = 366 if include_leap else 365

    print(f"Birthday Paradox (days={days}, simulated with {trials:,} trials)")
    print(" n  |  exact %   sim %   E[pairs]")
    print("----------------------------------")
    for n in range(2, max_n + 1):
        exact = exact_probability(n, days) * 100
        sim = simulated_probability(n, trials, include_leap) * 100
        exp_pairs = expected_matching_pairs(n, days)
        print(f"{n:>3} | {exact:7.2f}  {sim:7.2f}   {exp_pairs:7.3f}")

def collision_histogram(n: int, trials: int = 20_000, include_leap: bool = False) -> dict[str, int]:
    """
    Count how many different days collide in a group:
    - "0 days with duplicates"
    - "1 day with duplicates"
    - "2+ days with duplicates"
    """
    buckets: defaultdict[str, int] = defaultdict(int)
    for _ in range(trials):
        dups = find_shared_birthdays(generate_group(n, include_leap))
        k = len(dups)
        if k == 0:
            buckets["0 days with duplicates"] += 1
        elif k == 1:
            buckets["1 day with duplicates"] += 1
        else:
            buckets["2+ days with duplicates"] += 1
    return dict(buckets)

def demo_group(n: int, include_leap: bool = False) -> None:
    """Generate one group and print any shared birthdays."""
    group = generate_group(n, include_leap)
    print(f"Group of {n}: {[f'{d.month:02d}-{d.day:02d}' for d in group]}")
    duplicates = find_shared_birthdays(group)
    if duplicates:
        pretty = [f"{m:02d}-{d:02d} ×{c}" for (m, d), c in duplicates.items()]
        print("Shared birthdays:", ", ".join(pretty))
    else:
        print("No shared birthdays.")

# =========================
# Run section
# =========================
if __name__ == "__main__":
    INCLUDE_LEAP = False                     # set True to include Feb 29
    DAYS = 366 if INCLUDE_LEAP else 365
    TRIALS = 50_000
    N = 23

    print("=== Quick facts ===")
    print(f"n = {N}, days = {DAYS}")
    print(f"Exact probability:      {exact_probability(N, DAYS)*100:.2f}%")
    print(f"Simulated probability:  {simulated_probability(N, TRIALS, INCLUDE_LEAP)*100:.2f}%")
    print(f"Expected matching pairs: {expected_matching_pairs(N, DAYS):.3f}")
    print(f"Smallest n for ≥50%:     {smallest_group_for(0.50, DAYS)}")
    print(f"Smallest n for ≥99%:     {smallest_group_for(0.99, DAYS)}")
    print()

    probability_table(max_n=30, trials=20_000, include_leap=INCLUDE_LEAP, days=DAYS)
    print()

    print("=== Collision histogram (how many different days collide) ===")
    hist = collision_histogram(n=N, trials=20_000, include_leap=INCLUDE_LEAP)
    for k, v in hist.items():
        print(f"{k}: {v}")
    print()

    print("=== One random demo group ===")
    demo_group(N, include_leap=INCLUDE_LEAP)
