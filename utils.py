import time


def show_progress_estimate(iterable, desc: str = ''):
    # simple wrapper around tqdm if you want consistent progress bars
    from tqdm import tqdm
    return tqdm(iterable, desc=desc, unit='item')