import numpy as np
import importlib.util
from pathlib import Path

# Load shuffle module without importing cremerl package
shuffle_path = Path(__file__).resolve().parents[1] / "cremerl" / "shuffle.py"
shuffle_spec = importlib.util.spec_from_file_location("shuffle", shuffle_path)
shuffle = importlib.util.module_from_spec(shuffle_spec)
shuffle_spec.loader.exec_module(shuffle)

# Extract get_batch_score from utils.py without importing heavy deps
utils_path = Path(__file__).resolve().parents[1] / "cremerl" / "utils.py"
utils_source = utils_path.read_text()
start = utils_source.index("def get_batch_score")
end = utils_source.index("def extend_sequence")
ns = {"np": np}
exec(utils_source[start:end], ns)
get_batch_score = ns["get_batch_score"]

def dinuc_counts(tokens):
    counts = {}
    for i in range(len(tokens)-1):
        pair = (int(tokens[i]), int(tokens[i+1]))
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def test_get_batch_score_vector():
    pred = np.array([10, 20, 30, 40])
    assert get_batch_score(pred, trials=2) == -10.0

def test_get_batch_score_matrix():
    pred = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert get_batch_score(pred, trials=2) == -4.0

def test_dinuc_shuffle_preserves_shape_and_counts():
    rng = np.random.RandomState(0)
    tokens = rng.randint(0, 4, size=10)
    seq = np.identity(4)[tokens].T
    shuffled = shuffle.dinuc_shuffle(seq, rng=np.random.RandomState(1))
    assert shuffled.shape == seq.shape

    orig = shuffle.one_hot_to_tokens(seq)
    shuf = shuffle.one_hot_to_tokens(shuffled)
    assert dinuc_counts(orig) == dinuc_counts(shuf)
