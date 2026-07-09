import kfactory as kf
from kfactory.routing.factories import (
    get_polygon_materialization_cross_section,
    normalize_routing_straight_factory_dbu,
    normalize_routing_straight_factory_um,
    supports_polygon_materialization_factory,
    supports_routing_fast_factory,
)


class _RoutingFastAttributeFactory:
    def __init__(self, kcl: kf.KCLayout) -> None:
        self.kcl = kcl
        self.calls: list[tuple[int, int]] = []
        self.fast_calls: list[tuple[int, int]] = []

    def __call__(self, width: int, length: int) -> kf.KCell:
        self.calls.append((width, length))
        return self.kcl.kcell()

    def routing_fast_factory(self, width: int, length: int) -> kf.KCell:
        self.fast_calls.append((width, length))
        return self.kcl.kcell()


class _RoutingFastParameterFactory:
    def __init__(self, kcl: kf.KCLayout) -> None:
        self.kcl = kcl
        self.calls: list[tuple[int, int, bool]] = []

    def __call__(self, width: int, length: int, routing_fast: bool = False) -> kf.KCell:
        self.calls.append((width, length, routing_fast))
        return self.kcl.kcell()


class _UMRoutingFastParameterFactory:
    def __init__(self, kcl: kf.KCLayout) -> None:
        self.kcl = kcl
        self.calls: list[tuple[float, float, bool]] = []

    def __call__(
        self, width: float, length: float, routing_fast: bool = False
    ) -> kf.KCell:
        self.calls.append((width, length, routing_fast))
        return self.kcl.kcell()


def test_dbu_normalizer_prefers_explicit_routing_fast_factory(
    kcl: kf.KCLayout,
) -> None:
    factory = _RoutingFastAttributeFactory(kcl)
    normalized = normalize_routing_straight_factory_dbu(factory)

    assert supports_routing_fast_factory(normalized)
    assert supports_polygon_materialization_factory(normalized)

    cell1 = normalized(width=500, length=1000)
    cell2 = normalized(width=500, length=1000)

    assert cell1 is cell2
    assert factory.calls == []
    assert factory.fast_calls == [(500, 1000)]


def test_dbu_normalizer_caches_polygon_cross_section_by_width(
    kcl: kf.KCLayout,
) -> None:
    factory = _RoutingFastAttributeFactory(kcl)
    normalized = normalize_routing_straight_factory_dbu(factory)

    assert get_polygon_materialization_cross_section(normalized, 500) is None
    assert get_polygon_materialization_cross_section(normalized, 500) is None
    assert get_polygon_materialization_cross_section(normalized, 700) is None

    assert factory.calls == []
    assert factory.fast_calls == [(500, 1), (700, 1)]


def test_dbu_normalizer_uses_routing_fast_parameter(kcl: kf.KCLayout) -> None:
    factory = _RoutingFastParameterFactory(kcl)
    normalized = normalize_routing_straight_factory_dbu(factory)

    assert supports_routing_fast_factory(normalized)
    assert not supports_polygon_materialization_factory(normalized)

    cell1 = normalized(width=700, length=900, routing_fast=True)
    cell2 = normalized(width=700, length=900)

    assert cell1 is cell2
    assert factory.calls == [(700, 900, True)]


def test_um_normalizer_converts_dimensions_to_dbu_factory(kcl: kf.KCLayout) -> None:
    factory = _UMRoutingFastParameterFactory(kcl)
    normalized = normalize_routing_straight_factory_um(factory, kcl)

    assert supports_routing_fast_factory(normalized)
    assert not supports_polygon_materialization_factory(normalized)

    cell1 = normalized(width=500, length=2000, routing_fast=True)
    cell2 = normalized(width=500, length=2000)

    assert cell1 is cell2
    assert factory.calls == [(kcl.to_um(500), kcl.to_um(2000), True)]
