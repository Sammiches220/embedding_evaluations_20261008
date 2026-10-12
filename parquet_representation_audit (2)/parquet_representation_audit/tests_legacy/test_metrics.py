import numpy as np
from embedding_feature_audit.neighborhood import categorical_purity_lift


def test_separated_clusters_have_high_lift():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-5, scale=0.1, size=(30, 2))
    b = rng.normal(loc=5, scale=0.1, size=(30, 2))
    z = np.vstack([a, b])
    y = np.array(["a"] * 30 + ["b"] * 30)
    summary, _ = categorical_purity_lift(z, y, [5])
    assert summary.loc[0, "micro_purity"] > 0.95
    assert summary.loc[0, "micro_lift"] > 1.8
