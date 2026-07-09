"""Extra tests for kfactory.routing.{electrical,optical} modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import kfactory as kf
from kfactory.routing.electrical import (
    place_dual_rails,
    place_single_wire,
    route_bundle,
    route_bundle_dual_rails,
    route_dual_rails,
)
from kfactory.routing.generic import (
    ManhattanBundlePlanner,
    ManhattanRoute,
)
from kfactory.routing.manhattan import (
    ManhattanChannelPlanner,
    ManhattanRouter,
    route_smart,
)
from kfactory.routing.optical import (
    LoopPosition,
    LoopSide,
    path_length_match,
    route_loopback,
    vec_angle,
)
from kfactory.routing.optical import (
    route_bundle as optical_route_bundle,
)
from kfactory.routing.steps import Straight

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tests.conftest import Layers

# vec_angle


def unreachable_placer(
    c: kf.KCell,
    p1: kf.Port,
    p2: kf.Port,
    pts: Sequence[kf.kdb.Point],
    route_width: int | None = None,
    **kwargs: object,
) -> ManhattanRoute:
    raise AssertionError("unreachable")


def test_vec_angle_positive_x() -> None:
    assert vec_angle(kf.kdb.Vector(100, 0)) == 0


def test_vec_angle_negative_x() -> None:
    assert vec_angle(kf.kdb.Vector(-100, 0)) == 2


def test_vec_angle_positive_y() -> None:
    assert vec_angle(kf.kdb.Vector(0, 100)) == 1


def test_vec_angle_negative_y() -> None:
    assert vec_angle(kf.kdb.Vector(0, -100)) == 3


def test_vec_angle_zero_vector() -> None:
    # zero vector returns -1 with a log warning
    assert vec_angle(kf.kdb.Vector(0, 0)) == -1


def test_vec_angle_non_manhattan_raises() -> None:
    with pytest.raises(ValueError, match="Non-manhattan"):
        vec_angle(kf.kdb.Vector(100, 100))


def test_route_bundle_plan_materializer_boundary(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    c = kcl.kcell("route_bundle_plan_boundary")
    enc = kf.LayerEnclosure(
        sections=[(layers.WGEX, 5000)],
        name="WG",
        main_layer=layers.WG,
    )
    xs = c.kcl.get_icross_section(
        cross_section=kf.SymmetricalCrossSection(width=1000, enclosure=enc)
    )
    start = c.create_port(
        trans=kf.kdb.Trans(rot=0, mirrx=False, x=0, y=0),
        cross_section=xs,
        name="start",
    )
    end = c.create_port(
        trans=kf.kdb.Trans(rot=2, mirrx=False, x=100_000, y=0),
        cross_section=xs,
        name="end",
    )

    planner = ManhattanBundlePlanner(
        c=c,
        start_ports=[start.base],
        end_ports=[end.base],
        route_width=None,
        on_collision="error",
        on_placer_error="error",
        collision_check_layers=None,
        routing_function=route_smart,
        routing_kwargs={
            "bend90_radius": 10_000,
            "separation": 5_000,
            "sort_ports": False,
            "bbox_routing": "minimal",
            "bboxes": [],
            "waypoints": None,
            "allow_sbend": False,
        },
        placer_function=unreachable_placer,
        placer_kwargs={},
        constraints=None,
        starts=[],
        ends=[],
        start_angles=None,
        end_angles=None,
        route_name=None,
    )

    placed: list[tuple[str | None, str | None, list[kf.kdb.Point]]] = []

    def placer(
        c: kf.KCell,
        p1: kf.Port,
        p2: kf.Port,
        pts: Sequence[kf.kdb.Point],
        route_width: int | None = None,
        **kwargs: object,
    ) -> ManhattanRoute:
        placed.append((p1.name, p2.name, list(pts)))
        return ManhattanRoute(backbone=list(pts), start_port=p1, end_port=p2)

    inputs = planner.normalize_inputs()
    plan = planner.plan(inputs)
    planner.placer_function = placer
    routes = planner.materialize(plan)

    assert len(plan.routers) == 1
    assert plan.start_ports == [start.base]
    assert plan.end_ports == [end.base]
    assert len(routes) == 1
    assert placed == [("start", "end", plan.routers[0].start.pts)]


def test_route_bundle_input_normalization(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("route_bundle_input_normalization")
    enc = kf.LayerEnclosure(
        sections=[(layers.WGEX, 5000)],
        name="WG",
        main_layer=layers.WG,
    )
    xs = c.kcl.get_icross_section(
        cross_section=kf.SymmetricalCrossSection(width=1000, enclosure=enc)
    )
    start_ports = [
        c.create_port(
            trans=kf.kdb.Trans(rot=0, mirrx=False, x=0, y=i * 10_000),
            cross_section=xs,
            name=f"start_{i}",
        ).base
        for i in range(2)
    ]
    end_ports = [
        c.create_port(
            trans=kf.kdb.Trans(rot=2, mirrx=False, x=100_000, y=i * 10_000),
            cross_section=xs,
            name=f"end_{i}",
        ).base
        for i in range(2)
    ]

    planner = ManhattanBundlePlanner(
        c=c,
        start_ports=start_ports,
        end_ports=end_ports,
        route_width=1200,
        on_collision="error",
        on_placer_error="error",
        collision_check_layers=None,
        routing_function=route_smart,
        routing_kwargs={},
        placer_function=unreachable_placer,
        placer_kwargs={},
        constraints=None,
        starts=50_000,
        ends=[[Straight(dist=10_000)], [Straight(dist=20_000)]],
        start_angles=1,
        end_angles=[3, 3],
        route_name=None,
    )
    inputs = planner.normalize_inputs()

    assert inputs.widths == [1200, 1200]
    assert [p.get_trans().angle for p in inputs.start_ports] == [1, 1]
    assert [p.get_trans().angle for p in inputs.end_ports] == [3, 3]
    assert all(
        isinstance(step, Straight)
        for route_steps in inputs.starts
        for step in route_steps
    )
    assert all(
        isinstance(step, Straight)
        for route_steps in inputs.ends
        for step in route_steps
    )
    assert [
        [step.dist for step in route_steps if isinstance(step, Straight)]
        for route_steps in inputs.starts
    ] == [
        [50_000],
        [50_000],
    ]
    assert [
        [step.dist for step in route_steps if isinstance(step, Straight)]
        for route_steps in inputs.ends
    ] == [
        [10_000],
        [20_000],
    ]


def test_route_smart_direct_parallel_bank(kcl: kf.KCLayout, layers: Layers) -> None:
    start_ports = [
        kf.Port(
            name=f"s{i}",
            trans=kf.kdb.Trans(1, False, i * 10_000, 0),
            width=1000,
            layer_info=layers.WG,
            kcl=kcl,
            port_type="optical",
        )
        for i in range(5)
    ]
    end_ports = [
        kf.Port(
            name=f"e{i}",
            trans=kf.kdb.Trans(3, False, i * 10_000, 100_000),
            width=1000,
            layer_info=layers.WG,
            kcl=kcl,
            port_type="optical",
        )
        for i in range(5)
    ]

    routers = route_smart(
        start_ports=[p.base for p in start_ports],
        end_ports=[p.base for p in end_ports],
        starts=[[] for _ in start_ports],
        ends=[[] for _ in start_ports],
        widths=[p.width for p in start_ports],
        bend90_radius=5_000,
        separation=2_000,
        bboxes=[],
        sort_ports=False,
    )

    assert len(routers) == len(start_ports)
    assert all(router.finished for router in routers)
    assert [router.start.pts for router in routers] == [
        [start.trans.disp.to_p(), end.trans.disp.to_p()]
        for start, end in zip(start_ports, end_ports, strict=False)
    ]


def test_channel_planner_routes_independent_no_obstacle_channels() -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=5_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(3, False, 20_000, 20_000),
            width=1_000,
        ),
        ManhattanRouter(
            bend90_radius=5_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 50_000),
            end_transformation=kf.kdb.Trans(3, False, 20_000, 70_000),
            width=1_000,
        ),
    ]

    planner = ManhattanChannelPlanner(
        routers=routers,
        starts=[[], []],
        ends=[[], []],
        bboxes=[],
        sort_ports=False,
        waypoints=None,
        separation=2_000,
        route_debug=None,
    )

    assert planner.try_route()
    assert [router.start.pts for router in routers] == [
        [kf.kdb.Point(0, 0), kf.kdb.Point(20_000, 0), kf.kdb.Point(20_000, 20_000)],
        [
            kf.kdb.Point(0, 50_000),
            kf.kdb.Point(20_000, 50_000),
            kf.kdb.Point(20_000, 70_000),
        ],
    ]


def test_channel_planner_routes_independent_channels_past_unrelated_bboxes() -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=5_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(3, False, 20_000, 20_000),
            width=1_000,
        )
    ]

    planner = ManhattanChannelPlanner(
        routers=routers,
        starts=[[]],
        ends=[[]],
        bboxes=[kf.kdb.Box(100_000, 100_000, 120_000, 120_000)],
        sort_ports=False,
        waypoints=None,
        separation=2_000,
        route_debug=None,
    )

    assert planner.try_route()
    assert routers[0].start.pts == [
        kf.kdb.Point(0, 0),
        kf.kdb.Point(20_000, 0),
        kf.kdb.Point(20_000, 20_000),
    ]


def test_channel_planner_rejects_independent_channels_hitting_bbox() -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=5_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(3, False, 20_000, 20_000),
            width=1_000,
        )
    ]
    initial_start_pts = list(routers[0].start.pts)
    initial_end_pts = list(routers[0].end.pts)

    planner = ManhattanChannelPlanner(
        routers=routers,
        starts=[[]],
        ends=[[]],
        bboxes=[kf.kdb.Box(10_000, -1_000, 11_000, 1_000)],
        sort_ports=False,
        waypoints=None,
        separation=2_000,
        route_debug=None,
    )

    assert not planner.try_route()
    assert routers[0].start.pts == initial_start_pts
    assert routers[0].end.pts == initial_end_pts
    assert not routers[0].finished


def test_channel_planner_rejects_independent_channels_starting_in_bbox() -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=5_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(3, False, 20_000, 20_000),
            width=1_000,
        )
    ]

    planner = ManhattanChannelPlanner(
        routers=routers,
        starts=[[]],
        ends=[[]],
        bboxes=[kf.kdb.Box(-1_000, -1_000, 1_000, 1_000)],
        sort_ports=False,
        waypoints=None,
        separation=2_000,
        route_debug=None,
    )

    assert not planner.try_route()
    assert routers[0].start.pts == [kf.kdb.Point(0, 0)]
    assert not routers[0].finished


# place_single_wire / route_dual_rails


def _e_port(
    kcl: kf.KCLayout,
    layers: Layers,
    name: str,
    angle: int,
    x: int,
    y: int,
    width: int = 1000,
) -> kf.Port:
    return kf.Port(
        name=name,
        trans=kf.kdb.Trans(angle, False, x, y),
        width=width,
        layer_info=layers.METAL1,
        kcl=kcl,
        port_type="electrical",
    )


def test_place_single_wire_basic(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("psw_basic")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    route = place_single_wire(c, p1, p2, pts)
    assert route.start_port is p1
    assert route.end_port is p2
    assert route.length_straights == 50_000


def test_place_single_wire_extra_kwargs_raises(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    c = kcl.kcell("psw_kwargs_err")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    with pytest.raises(ValueError, match="supported"):
        place_single_wire(c, p1, p2, pts, junk_kwarg=42)


def test_place_single_wire_explicit_layer_and_width(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    c = kcl.kcell("psw_explicit")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    route = place_single_wire(
        c, p1, p2, pts, route_width=2000, layer_info=layers.METAL2
    )
    assert route.polygons[layers.METAL2]


def test_place_dual_rails_basic(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("pdr_basic")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0, width=4000)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    route = place_dual_rails(c, p1, p2, pts, separation_rails=1000)
    assert route.start_port is p1
    assert route.end_port is p2
    # The shape polygons live on the port's layer
    assert route.polygons[layers.METAL1]


def test_place_dual_rails_missing_separation(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("pdr_no_sep")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0, width=4000)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    with pytest.raises(ValueError, match="Must specify"):
        place_dual_rails(c, p1, p2, pts)


def test_place_dual_rails_separation_too_large(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    c = kcl.kcell("pdr_sep_too_large")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=2000)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0, width=2000)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    with pytest.raises(ValueError, match="must be smaller"):
        place_dual_rails(c, p1, p2, pts, separation_rails=5000)


def test_place_dual_rails_extra_kwargs_raises(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("pdr_kwargs_err")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0, width=4000)
    pts = [kf.kdb.Point(0, 0), kf.kdb.Point(50_000, 0)]
    with pytest.raises(ValueError, match="supported"):
        place_dual_rails(c, p1, p2, pts, separation_rails=1000, junk=1)


def test_route_dual_rails(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("rdr_basic")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0, width=4000)
    route_dual_rails(
        c, p1, p2, width=4000, hole_width=1000, layer=kcl.find_layer(layers.METAL1)
    )
    # Some shapes should have been inserted
    shapes_count = 0
    for shape in c.shapes(kcl.find_layer(layers.METAL1)).each():
        shapes_count += 1
        _ = shape
    assert shapes_count > 0


def test_route_dual_rails_defaults(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("rdr_defaults")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 50_000, 0, width=4000)
    # Use the port-inferred defaults for width/hole_width/layer
    route_dual_rails(c, p1, p2)


# Integration: electrical route_bundle with um (DKCell) path


def test_electrical_route_bundle_dkcell(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.dkcell("e_route_bundle_dk")
    p1 = kf.DPort(
        name="in",
        dcplx_trans=kf.kdb.DCplxTrans(1, 0, False, 0, 0),
        width=10.0,
        layer_info=layers.METAL1,
        kcl=kcl,
        port_type="electrical",
    )
    p2 = kf.DPort(
        name="out",
        dcplx_trans=kf.kdb.DCplxTrans(1, 180, False, 50.0, 0),
        width=10.0,
        layer_info=layers.METAL1,
        kcl=kcl,
        port_type="electrical",
    )
    routes = route_bundle(
        c,
        start_ports=[p1],
        end_ports=[p2],
        separation=2.0,
        on_collision=None,
        starts=[0.0],
        ends=[0.0],
    )
    assert len(routes) == 1


def test_electrical_route_bundle_non_manhattan_waypoints_dbu(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    """Non-manhattan waypoints should produce a descriptive ValueError."""
    c = kcl.kcell("e_route_bundle_nmwp_dbu")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 100_000, 0, width=4000)
    with pytest.raises(ValueError, match="non-manhattan waypoints"):
        route_bundle(
            c,
            start_ports=[p1],
            end_ports=[p2],
            separation=4_000,
            on_collision=None,
            on_placer_error=None,
            waypoints=[
                kf.kdb.Point(0, 0),
                kf.kdb.Point(50_000, 10_000),
            ],
        )


def test_electrical_route_bundle_non_manhattan_waypoints_dk(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    """Non-manhattan waypoints on the DKCell path."""
    c = kcl.dkcell("e_route_bundle_nmwp_dk")
    p1 = kf.DPort(
        name="in",
        dcplx_trans=kf.kdb.DCplxTrans(1, 0, False, 0, 0),
        width=10.0,
        layer_info=layers.METAL1,
        kcl=kcl,
        port_type="electrical",
    )
    p2 = kf.DPort(
        name="out",
        dcplx_trans=kf.kdb.DCplxTrans(1, 180, False, 100.0, 0),
        width=10.0,
        layer_info=layers.METAL1,
        kcl=kcl,
        port_type="electrical",
    )
    with pytest.raises(ValueError, match="non-manhattan waypoints"):
        route_bundle(
            c,
            start_ports=[p1],
            end_ports=[p2],
            separation=2.0,
            on_collision=None,
            on_placer_error=None,
            starts=[0.0],
            ends=[0.0],
            waypoints=[
                kf.kdb.DPoint(0, 0),
                kf.kdb.DPoint(50.0, 10.0),
            ],
        )


def test_electrical_route_bundle_dual_rails_non_manhattan_waypoints(
    kcl: kf.KCLayout, layers: Layers
) -> None:
    c = kcl.kcell("e_route_bundle_dr_nmwp")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 100_000, 0, width=4000)
    with pytest.raises(ValueError, match="non-manhattan waypoints"):
        route_bundle_dual_rails(
            c,
            start_ports=[p1],
            end_ports=[p2],
            separation=4_000,
            separation_rails=500,
            on_collision=None,
            on_placer_error=None,
            waypoints=[
                kf.kdb.Point(0, 0),
                kf.kdb.Point(50_000, 10_000),
            ],
        )


def test_optical_route_bundle_non_manhattan_waypoints(
    bend90: kf.KCell,
    straight_factory_dbu: Callable[..., kf.KCell],
    kcl: kf.KCLayout,
    layers: Layers,
) -> None:
    c = kcl.kcell("opt_route_bundle_nmwp")
    p1 = kf.Port(
        name="in",
        trans=kf.kdb.Trans(0, False, 0, 0),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    p2 = kf.Port(
        name="out",
        trans=kf.kdb.Trans(2, False, 500_000, 0),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    with pytest.raises(ValueError, match="non-manhattan waypoints"):
        optical_route_bundle(
            c,
            start_ports=[p1],
            end_ports=[p2],
            separation=4_000,
            straight_factory=straight_factory_dbu,
            bend90_cell=bend90,
            on_collision=None,
            on_placer_error=None,
            waypoints=[
                kf.kdb.Point(0, 0),
                kf.kdb.Point(50_000, 10_000),
            ],
        )


def test_electrical_route_bundle_dual_rails(kcl: kf.KCLayout, layers: Layers) -> None:
    c = kcl.kcell("e_route_bundle_dual_rails")
    p1 = _e_port(kcl, layers, "in", 0, 0, 0, width=4000)
    p2 = _e_port(kcl, layers, "out", 2, 100_000, 0, width=4000)
    routes = route_bundle_dual_rails(
        c,
        start_ports=[p1],
        end_ports=[p2],
        separation=4_000,
        separation_rails=500,
        on_collision=None,
    )
    assert len(routes) == 1


def test_route_loopback_basic(layers: Layers, kcl: kf.KCLayout) -> None:
    # route_loopback returns a list of points for the loopback path
    p1 = kf.Port(
        name="p1",
        trans=kf.kdb.Trans(0, False, 0, 0),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    p2 = kf.Port(
        name="p2",
        trans=kf.kdb.Trans(0, False, 0, 50_000),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    pts = route_loopback(p1, p2, bend90_radius=10_000)
    assert isinstance(pts, list)
    assert len(pts) >= 2


def test_route_loopback_with_bend180(layers: Layers, kcl: kf.KCLayout) -> None:
    p1 = kf.Port(
        name="p1",
        trans=kf.kdb.Trans(0, False, 0, 0),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    p2 = kf.Port(
        name="p2",
        trans=kf.kdb.Trans(0, False, 0, 50_000),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    pts = route_loopback(p1, p2, bend90_radius=10_000, bend180_radius=20_000)
    assert isinstance(pts, list)


def test_route_loopback_inside(layers: Layers, kcl: kf.KCLayout) -> None:
    p1 = kf.Port(
        name="p1",
        trans=kf.kdb.Trans(0, False, 0, 0),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    p2 = kf.Port(
        name="p2",
        trans=kf.kdb.Trans(0, False, 0, 50_000),
        width=500,
        layer_info=layers.WG,
        kcl=kcl,
    )
    pts = route_loopback(p1, p2, bend90_radius=10_000, inside=True)
    assert isinstance(pts, list)


def test_path_length_match_left(layers: Layers, kcl: kf.KCLayout) -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 100_000, 0),
        ),
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 50_000),
            end_transformation=kf.kdb.Trans(2, False, 200_000, 50_000),
        ),
    ]
    for r in routers:
        r.start.straight(100_000)
        r.start.pts.append(r.start.t.disp.to_p())
        r.finished = True

    # Should run without error; modifies router.start.pts in place
    path_length_match(
        routers=routers,
        loops=1,
        loop_side=LoopSide.left,
        loop_position=LoopPosition.start,
    )
    # Path lengths should now match
    lengths = [r.start.path_length for r in routers]
    assert max(lengths) - min(lengths) <= 2  # 2 dbu rounding


def test_path_length_match_right(layers: Layers, kcl: kf.KCLayout) -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 100_000, 0),
        ),
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 200_000, 0),
        ),
    ]
    for r in routers:
        r.start.straight(100_000)
        r.start.pts.append(r.start.t.disp.to_p())
        r.finished = True

    path_length_match(routers=routers, loops=1, loop_side=LoopSide.right)


def test_path_length_match_center(layers: Layers, kcl: kf.KCLayout) -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 100_000, 0),
        ),
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 200_000, 0),
        ),
    ]
    for r in routers:
        r.start.straight(100_000)
        r.start.pts.append(r.start.t.disp.to_p())
        r.finished = True

    path_length_match(routers=routers, loops=1, loop_side=LoopSide.center)


def test_path_length_match_explicit_short_path_length(
    layers: Layers, kcl: kf.KCLayout
) -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 100_000, 0),
        ),
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 200_000, 0),
        ),
    ]
    for r in routers:
        r.start.straight(100_000)
        r.start.pts.append(r.start.t.disp.to_p())
        r.finished = True
    # path_length below max should log warning and use minimum
    path_length_match(
        routers=routers,
        loops=1,
        loop_side=LoopSide.left,
        path_length=10,
    )


def test_path_length_match_invalid_side_raises(
    layers: Layers, kcl: kf.KCLayout
) -> None:
    router = ManhattanRouter(
        bend90_radius=10_000,
        separation=2_000,
        start_transformation=kf.kdb.Trans(0, False, 0, 0),
        end_transformation=kf.kdb.Trans(2, False, 100_000, 0),
    )
    router.start.straight(100_000)
    router.start.pts.append(router.start.t.disp.to_p())
    router.finished = True
    with pytest.raises(ValueError, match="must be of any value"):
        path_length_match(routers=[router], loops=1, loop_side=42)  # ty:ignore[invalid-argument-type]


def test_path_length_match_invalid_position_raises(
    layers: Layers, kcl: kf.KCLayout
) -> None:
    routers = [
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 100_000, 0),
        ),
        ManhattanRouter(
            bend90_radius=10_000,
            separation=2_000,
            start_transformation=kf.kdb.Trans(0, False, 0, 0),
            end_transformation=kf.kdb.Trans(2, False, 200_000, 0),
        ),
    ]
    for r in routers:
        r.start.straight(100_000)
        r.start.pts.append(r.start.t.disp.to_p())
        r.finished = True
    with pytest.raises(ValueError, match="loop_position must be"):
        path_length_match(
            routers=routers,
            loops=1,
            loop_side=LoopSide.left,
            loop_position=42,  # ty:ignore[invalid-argument-type]
        )


# Optical route_bundle DKCell


def test_optical_route_bundle_dkcell(
    layers: Layers,
    kcl: kf.KCLayout,
    bend90: kf.KCell,
    straight_factory: Callable[..., kf.KCell],
) -> None:
    c = kcl.dkcell("opt_route_bundle_dk")
    p1 = kf.DPort(
        name="in",
        dcplx_trans=kf.kdb.DCplxTrans(1, 0, False, 0, 0),
        width=0.5,
        layer_info=layers.WG,
        kcl=kcl,
    )
    p2 = kf.DPort(
        name="out",
        dcplx_trans=kf.kdb.DCplxTrans(1, 180, False, 100.0, 50.0),
        width=0.5,
        layer_info=layers.WG,
        kcl=kcl,
    )

    def sf(*, width: float, length: float, **kwargs: object) -> kf.DKCell:
        return straight_factory(width=width, length=length).to_dtype()

    routes = optical_route_bundle(  # ty:ignore[no-matching-overload]
        c,
        start_ports=[p1],
        end_ports=[p2],
        separation=4.0,
        straight_factory=sf,
        bend90_cell=bend90.to_dtype(),
        on_collision=None,
        starts=[0.0],
        ends=[0.0],
    )
    assert len(routes) == 1
