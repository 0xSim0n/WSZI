"""Szybkie testy poprawnosci (bez pytest):  python tests/test_basic.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

from cfd.metrics import delta_grid, robustness_radius
from cfd.models import Classifier, TrainConfig
from cfd.streams import DriftAssumption, make_stream


def test_sea_label_and_drift():
    st = make_stream("sea")
    X = np.array([[0.30, 0.40, 0.9], [0.50, 0.50, 0.1]])          # sumy 7 i 10 (jedn. oryg.)
    assert list(st.label(X, 0.0)) == [1, 0]
    assert list(st.label(X, 2.0)) == [1, 1]                       # theta 8 -> 10
    assert st.disagreement(0.0) == 0.0 and st.disagreement(st.default_bound) > 0.05


def test_drift_is_bounded():
    for name in ("sea", "hyperplane"):
        st = make_stream(name)
        assert max(abs(st.delta_at(t)) for t in range(1000)) <= st.default_bound + 1e-12


def test_robustness_radius_sea_analytic():
    st = make_stream("sea")
    x = np.array([[0.2, 0.4, 0.5]])                               # suma 6 -> klasa 1 dopoki theta >= 6
    Rp, Rm, _, _ = robustness_radius(st, x, 1, delta_grid(st))
    assert Rp[0] == delta_grid(st)[-1]                            # theta rosnie: zawsze poprawny
    assert abs(Rm[0] - 2.0) < 0.11                                # theta = 8 - r >= 6  =>  r <= 2


def test_models_train_and_shapes():
    st = make_stream("sea")
    rng = np.random.default_rng(0)
    X, y = st.sample(1500, rng, 0.0, noise=0.05)
    a = DriftAssumption.symmetric(st.default_bound)
    for kind in ("standard", "cbd", "dual"):
        c = Classifier(kind, st.d, TrainConfig(epochs=40), a if kind != "standard" else None).fit(X, y, st)
        assert c.predict_proba(X[:5]).shape == (5,)
        assert c.target_prob(X[:5], 1).shape == (5,)
        assert (c.predict(X) == st.label(X, 0.0)).mean() > 0.9, kind


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("OK", k)
