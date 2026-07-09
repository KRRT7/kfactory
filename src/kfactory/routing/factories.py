"""Internal factory adapters for routing hot paths."""

from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from ..kcell import KCell, ProtoTKCell

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..factories import StraightFactoryDBU, StraightFactoryUM
    from ..layout import KCLayout


__all__ = [
    "get_polygon_materialization_cross_section",
    "normalize_routing_straight_factory_dbu",
    "normalize_routing_straight_factory_um",
    "supports_polygon_materialization_factory",
    "supports_routing_fast_factory",
]


class _HasRoutingFastFactory(Protocol):
    routing_fast_factory: StraightFactoryDBU


class _RoutingFastParameterFactoryDBU(Protocol):
    def __call__(
        self, width: int, length: int, routing_fast: bool = False
    ) -> ProtoTKCell[Any]: ...


class _RoutingFastParameterFactoryUM(Protocol):
    def __call__(
        self, width: float, length: float, routing_fast: bool = False
    ) -> ProtoTKCell[Any]: ...


class _RoutingStraightFactoryDBU(Protocol):
    supports_routing_fast: bool
    supports_polygon_materialization: bool

    def __call__(
        self, width: int, length: int, routing_fast: bool = False
    ) -> KCell: ...


class _HasPolygonMaterializationCrossSection(Protocol):
    def polygon_materialization_cross_section(self, width: int) -> Any | None: ...


class _CachedRoutingStraightFactoryDBU:
    __slots__ = (
        "_cache",
        "_cross_section_cache",
        "_make_cell",
        "supports_polygon_materialization",
        "supports_routing_fast",
    )

    def __init__(
        self,
        make_cell: Callable[[int, int, bool], KCell],
        *,
        supports_routing_fast: bool,
        supports_polygon_materialization: bool,
    ) -> None:
        self._make_cell = make_cell
        self._cache: dict[tuple[int, int], KCell] = {}
        self._cross_section_cache: dict[int, Any | None] = {}
        self.supports_routing_fast = supports_routing_fast
        self.supports_polygon_materialization = supports_polygon_materialization

    def __call__(self, width: int, length: int, routing_fast: bool = False) -> KCell:
        key = (width, length)
        straight_cell = self._cache.get(key)
        if straight_cell is None:
            straight_cell = self._make_cell(width, length, routing_fast)
            self._cache[key] = straight_cell
        return straight_cell

    def polygon_materialization_cross_section(self, width: int) -> Any | None:
        if width not in self._cross_section_cache:
            straight_cell = self(width=width, length=1, routing_fast=True)
            self._cross_section_cache[width] = (
                straight_cell._base.ports[0].cross_section
                if straight_cell._base.ports
                else None
            )
        return self._cross_section_cache[width]


def normalize_routing_straight_factory_dbu(
    straight_factory: StraightFactoryDBU,
) -> StraightFactoryDBU:
    routing_fast_factory = _get_routing_fast_straight_factory(straight_factory)
    if routing_fast_factory is not None:

        def make_straight_cell(width: int, length: int, routing_fast: bool) -> KCell:
            return _expect_kcell(
                routing_fast_factory(
                    width=width,
                    length=length,
                ),
                "routing_fast_factory",
            )

        return _CachedRoutingStraightFactoryDBU(
            make_straight_cell,
            supports_routing_fast=True,
            supports_polygon_materialization=True,
        )

    if _accepts_routing_fast_parameter_dbu(straight_factory):
        routing_fast_parameter_factory = straight_factory

        def make_straight_cell(width: int, length: int, routing_fast: bool) -> KCell:
            return _expect_kcell(
                routing_fast_parameter_factory(
                    width=width,
                    length=length,
                    routing_fast=True,
                ),
                "straight_factory(routing_fast=True)",
            )

        return _CachedRoutingStraightFactoryDBU(
            make_straight_cell,
            supports_routing_fast=True,
            supports_polygon_materialization=False,
        )

    return straight_factory


def normalize_routing_straight_factory_um(
    straight_factory: StraightFactoryUM,
    kcl: KCLayout,
) -> StraightFactoryDBU:
    routing_fast_parameter_factory = (
        straight_factory
        if _accepts_routing_fast_parameter_um(straight_factory)
        else None
    )

    def make_straight_cell(width: int, length: int, routing_fast: bool) -> KCell:
        width_um = kcl.to_um(width)
        length_um = kcl.to_um(length)
        if routing_fast_parameter_factory is not None:
            dkc = routing_fast_parameter_factory(
                width=width_um,
                length=length_um,
                routing_fast=True,
            )
        else:
            dkc = straight_factory(width=width_um, length=length_um)
        return kcl[dkc.cell_index()]

    return _CachedRoutingStraightFactoryDBU(
        make_straight_cell,
        supports_routing_fast=routing_fast_parameter_factory is not None,
        supports_polygon_materialization=False,
    )


def supports_routing_fast_factory(
    factory: StraightFactoryDBU,
) -> TypeGuard[_RoutingStraightFactoryDBU]:
    return getattr(factory, "supports_routing_fast", False) is True


def supports_polygon_materialization_factory(
    factory: StraightFactoryDBU,
) -> TypeGuard[_RoutingStraightFactoryDBU]:
    return getattr(factory, "supports_polygon_materialization", False) is True


def get_polygon_materialization_cross_section(
    factory: StraightFactoryDBU,
    width: int,
) -> Any | None:
    if _has_polygon_materialization_cross_section(factory):
        return factory.polygon_materialization_cross_section(width)
    return None


def _get_routing_fast_straight_factory(
    straight_factory: object,
) -> StraightFactoryDBU | None:
    if _has_routing_fast_factory(straight_factory):
        return straight_factory.routing_fast_factory
    if isinstance(straight_factory, partial) and _has_routing_fast_factory(
        straight_factory.func
    ):
        return partial(
            straight_factory.func.routing_fast_factory,
            *straight_factory.args,
            **(straight_factory.keywords or {}),
        )
    return None


def _has_routing_fast_factory(factory: object) -> TypeGuard[_HasRoutingFastFactory]:
    return hasattr(factory, "routing_fast_factory")


def _has_polygon_materialization_cross_section(
    factory: object,
) -> TypeGuard[_HasPolygonMaterializationCrossSection]:
    return hasattr(factory, "polygon_materialization_cross_section")


def _accepts_routing_fast_parameter_dbu(
    factory: Callable[..., object],
) -> TypeGuard[_RoutingFastParameterFactoryDBU]:
    try:
        return "routing_fast" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _accepts_routing_fast_parameter_um(
    factory: Callable[..., object],
) -> TypeGuard[_RoutingFastParameterFactoryUM]:
    try:
        return "routing_fast" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _expect_kcell(cell: ProtoTKCell[Any], context: str) -> KCell:
    if not isinstance(cell, KCell):
        raise TypeError(f"{context} must return a KCell, got {type(cell).__name__}")
    return cell
