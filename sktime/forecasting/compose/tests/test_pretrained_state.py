#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Compositors must retain pretrained state of inner forecasters.

Regression for issue #11526: wrapping a pretrained estimator in a native
forecasting compositor and calling ``fit`` must not drop pretrained
attributes such as ``global_mean_``.
"""

__author__ = ["AndrewPrakash1"]

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.compose import (
    ForecastingPipeline,
    ForecastX,
    MultiplexForecaster,
    Permute,
    TransformedTargetForecaster,
)
from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.exponent import ExponentTransformer
from sktime.utils._testing.hierarchical import _make_hierarchical


class _PretrainedRequiredDummy(DummyGlobalForecaster):
    """Dummy that fails if pretrained state was dropped before ``fit``."""

    def _fit(self, y, X=None, fh=None):
        if not hasattr(self, "global_mean_"):
            raise RuntimeError("pretrained state discarded before fit")
        return super()._fit(y, X, fh)


def _pretrain_dummy():
    """Return a pretrained dummy and the mean it learned."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(3,),
        min_timepoints=15,
        max_timepoints=15,
        n_columns=1,
    )
    dummy = _PretrainedRequiredDummy(strategy="mean")
    dummy.pretrain(y_panel)
    return dummy, dummy.global_mean_


def _make_compositor(name, dummy):
    """Wrap ``dummy`` in the named forecasting compositor."""
    if name == "ForecastingPipeline":
        return ForecastingPipeline([("forecaster", dummy)])
    if name == "TransformedTargetForecaster":
        return TransformedTargetForecaster(
            [
                ("scale", ExponentTransformer(1)),
                ("forecaster", dummy),
            ]
        )
    if name == "ForecastX":
        return ForecastX(forecaster_y=dummy)
    if name == "Permute":
        pipe = TransformedTargetForecaster(
            [
                ("scale", ExponentTransformer(1)),
                ("forecaster", dummy),
            ]
        )
        return Permute(pipe, permutation=["scale", "forecaster"])
    if name == "MultiplexForecaster":
        return MultiplexForecaster(
            forecasters=[("dummy", dummy)],
            selected_forecaster="dummy",
        )
    raise ValueError(f"unknown compositor {name!r}")


def _inner_fitted_forecaster(name, estimator):
    """Return the inner fitted dummy wrapped by ``estimator``."""
    if name == "ForecastingPipeline":
        return estimator.forecaster_
    if name == "TransformedTargetForecaster":
        return estimator.forecaster_
    if name == "ForecastX":
        return estimator.forecaster_y_
    if name == "Permute":
        return estimator.estimator_.forecaster_
    if name == "MultiplexForecaster":
        return estimator.forecaster_
    raise ValueError(f"unknown compositor {name!r}")


_COMPOSITORS = [
    "ForecastingPipeline",
    "TransformedTargetForecaster",
    "ForecastX",
    "Permute",
    "MultiplexForecaster",
]

_COMPOSITOR_CLASSES = [
    ForecastingPipeline,
    TransformedTargetForecaster,
    ForecastX,
    Permute,
    MultiplexForecaster,
    DummyGlobalForecaster,
    ExponentTransformer,
]


@pytest.mark.skipif(
    not run_test_for_class(_COMPOSITOR_CLASSES),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("compositor", _COMPOSITORS)
def test_compositor_preserves_pretrained_state(compositor):
    """Inner fitted forecaster must keep pretrained ``global_mean_``.

    ``clone()`` keeps pretrained attributes via ``_PretrainedCloner``, but
    ``set_params`` calls ``reset()`` and drops them. Compositors should use
    ``_clone_at(..., "pretrained")`` / ``_set_params_at(..., "pretrained", ...)``
    so wrapping and fitting does not discard pretraining. Regression for #11526.
    """
    dummy, pretrain_mean = _pretrain_dummy()
    estimator = _make_compositor(compositor, dummy)
    estimator.fit(load_airline()[:36], fh=[1, 2, 3])

    inner = _inner_fitted_forecaster(compositor, estimator)
    assert hasattr(inner, "global_mean_")
    np.testing.assert_almost_equal(inner.global_mean_, pretrain_mean)
