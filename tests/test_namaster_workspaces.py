from __future__ import annotations

import types
import warnings


from comet.namaster_utils import _workspace_from_fields


class _ArgsWorkspace:
    """Stub that warns when instantiated with constructor arguments."""

    def __init__(self, field_1=None, field_2=None, bins=None):
        if field_1 is None or field_2 is None or bins is None:
            raise ValueError("missing constructor arguments")
        warnings.warn(
            "The bare constructor for `NmtWorkspace` objects is deprecated and will be removed soon",
            DeprecationWarning,
        )
        self._fields = (field_1, field_2, bins)


class _BareWorkspace:
    """Stub that only supports the legacy bare constructor."""

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("bare constructor accepts no arguments")
        warnings.warn(
            "The bare constructor for `NmtWorkspace` objects is deprecated and will be removed soon",
            DeprecationWarning,
        )
        self._computed_with: tuple[object, object, object] | None = None

    def compute_coupling_matrix(self, field_1, field_2, bins):
        self._computed_with = (field_1, field_2, bins)


def test_workspace_helper_suppresses_constructor_warning_when_args_supported() -> None:
    module = types.SimpleNamespace(NmtWorkspace=_ArgsWorkspace)
    with warnings.catch_warnings(record=True) as caught:
        workspace = _workspace_from_fields(module, "a", "b", "c")

    assert not caught
    assert isinstance(workspace, _ArgsWorkspace)


def test_workspace_helper_suppresses_bare_constructor_warning_and_computes_matrix() -> None:
    module = types.SimpleNamespace(NmtWorkspace=_BareWorkspace)
    with warnings.catch_warnings(record=True) as caught:
        workspace = _workspace_from_fields(module, "x", "y", "z")

    assert not caught
    assert isinstance(workspace, _BareWorkspace)
    assert workspace._computed_with == ("x", "y", "z")
