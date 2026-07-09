"""Generic routing functions which are independent of the potential use."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeGuard, cast

import klayout.db as kdb
from klayout import rdb
from pydantic import BaseModel, Field

from ..conf import logger
from ..instance import Instance  # noqa: TC001
from ..port import BasePort, Port, ProtoPort
from ..typings import dbu  # noqa: TC001
from .length_functions import LengthFunction, get_length_from_area
from .manhattan import (
    ManhattanBundleRoutingFunction,
    ManhattanRouter,
    route_smart,
)
from .steps import Step, Straight

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..kcell import KCell
    from ..schematic import Constraint
    from .utils import RouteDebug

__all__ = [
    "ManhattanBundlePlan",
    "ManhattanBundlePlanner",
    "ManhattanRoute",
    "PlacerFunction",
    "check_collisions",
    "get_radius",
    "route_bundle",
]


class PlacerError(ValueError):
    pass


class PlacerFunction(Protocol):
    """A placer function. Used to place Instances given a path."""

    def __call__(
        self,
        c: KCell,
        p1: Port,
        p2: Port,
        pts: Sequence[kdb.Point],
        route_width: int | None = None,
        **kwargs: Any,
    ) -> ManhattanRoute:
        """Implementation of the function."""
        ...


class ManhattanRoute(BaseModel, arbitrary_types_allowed=True):
    """Optical route containing a connection between two ports.

    Attrs:
        backbone: backbone points
        start_port: port at the first instance denoting the start of the route
        end_port: port at the last instance denoting the end of the route
        instances: list of the instances in order from start to end of the route
        n_bend90: number of bends used
        length: length of the route without the bends
        length_straights: length of the straight_factory elements
    """

    backbone: list[kdb.Point]
    start_port: Port
    end_port: Port
    instances: list[Instance] = Field(default_factory=list)
    n_bend90: int = 0
    n_taper: int = 0
    bend90_radius: dbu = 0
    taper_length: dbu = 0
    """Length of backbone without the bends."""
    length_straights: dbu = 0
    polygons: dict[kdb.LayerInfo, list[kdb.Polygon]] = Field(default_factory=dict)
    length_function: LengthFunction = Field(default_factory=get_length_from_area)

    @property
    def length_backbone(self) -> dbu:
        """Length of the backbone in dbu."""
        length = 0
        p_old = self.backbone[0]
        for p in self.backbone[1:]:
            length += int((p - p_old).length())
            p_old = p
        return length

    @property
    def length(self) -> int | float:
        return self.length_function(self)


@dataclass(slots=True)
class ManhattanBundlePlan:
    routers: list[ManhattanRouter]
    start_ports: list[BasePort]
    end_ports: list[BasePort]


@dataclass(slots=True)
class _NormalizedManhattanBundleInputs:
    start_ports: list[BasePort]
    end_ports: list[BasePort]
    widths: Sequence[int]
    starts: Sequence[Sequence[Step]]
    ends: Sequence[Sequence[Step]]


@dataclass(slots=True)
class ManhattanBundlePlanner:
    c: KCell
    start_ports: list[BasePort]
    end_ports: list[BasePort]
    route_width: dbu | list[dbu] | None
    on_collision: Literal["error", "show_error"] | None
    on_placer_error: Literal["error", "show_error"] | None
    collision_check_layers: Sequence[kdb.LayerInfo] | None
    routing_function: ManhattanBundleRoutingFunction
    routing_kwargs: dict[str, Any]
    placer_function: PlacerFunction
    placer_kwargs: dict[str, Any]
    constraints: Sequence[Constraint] | None
    starts: dbu | list[dbu] | list[Step] | list[list[Step]] | None
    ends: dbu | list[dbu] | list[Step] | list[list[Step]] | None
    start_angles: int | list[int] | None
    end_angles: int | list[int] | None
    route_name: str | None

    def normalize_inputs(self) -> _NormalizedManhattanBundleInputs:
        if not self.start_ports:
            return _NormalizedManhattanBundleInputs([], [], [], [], [])
        if not (len(self.start_ports) == len(self.end_ports)):
            raise ValueError(
                "For bundle routing the input port list must have"
                " the same size as the end ports and be the same length."
            )

        length = len(self.start_ports)
        normalized_starts = self._normalize_route_steps(self.starts, length)
        normalized_ends = self._normalize_route_steps(self.ends, length)
        normalized_start_ports = self._normalize_port_angles(
            self.start_ports, self.start_angles
        )
        normalized_end_ports = self._normalize_port_angles(
            self.end_ports, self.end_angles
        )
        widths = self._normalize_route_widths(self.route_width, normalized_start_ports)

        return _NormalizedManhattanBundleInputs(
            start_ports=normalized_start_ports,
            end_ports=normalized_end_ports,
            widths=widths,
            starts=normalized_starts,
            ends=normalized_ends,
        )

    def plan(self, inputs: _NormalizedManhattanBundleInputs) -> ManhattanBundlePlan:
        routers = self.routing_function(
            start_ports=inputs.start_ports,
            end_ports=inputs.end_ports,
            widths=inputs.widths,
            starts=inputs.starts,
            ends=inputs.ends,
            **self.routing_kwargs,
        )

        if not routers:
            return ManhattanBundlePlan(routers=[], start_ports=[], end_ports=[])

        start_mapping = {sp.get_trans(): sp for sp in inputs.start_ports}
        end_mapping = {ep.get_trans(): ep for ep in inputs.end_ports}
        ordered_start_ports = []
        ordered_end_ports = []
        for router in routers:
            sp = start_mapping[router.start_transformation]
            ep = end_mapping[router.end_transformation]
            ordered_start_ports.append(sp)
            ordered_end_ports.append(ep)

        return ManhattanBundlePlan(
            routers=routers,
            start_ports=ordered_start_ports,
            end_ports=ordered_end_ports,
        )

    def enforce_constraints(self, plan: ManhattanBundlePlan) -> None:
        if self.constraints:
            for constraint in self.constraints:
                constraint.enforce(
                    c=self.c,
                    routers=plan.routers,
                    route_name=self.route_name,
                )

    def materialize(self, plan: ManhattanBundlePlan) -> list[ManhattanRoute]:
        routes: list[ManhattanRoute] = []
        placer_errors: list[Exception] = []
        error_routes: list[tuple[BasePort, BasePort, list[kdb.Point], int]] = []
        for router, ps, pe in zip(
            plan.routers, plan.start_ports, plan.end_ports, strict=False
        ):
            try:
                route = self.placer_function(
                    self.c,
                    Port(base=ps),
                    Port(base=pe),
                    router.start.pts,
                    **self.placer_kwargs,
                )
                routes.append(route)
            except Exception as e:
                placer_errors.append(e)
                error_routes.append((ps, pe, router.start.pts, router.width))
        if placer_errors and self.on_placer_error == "show_error":
            self.show_placer_errors(placer_errors, error_routes)
        if placer_errors and self.on_placer_error is not None:
            for error in placer_errors:
                logger.error(error)
            if self.c.name.startswith("Unnamed_"):
                self.c.name = self.c.kcl._future_cell_name or self.c.name
            raise PlacerError(
                "Failed to place routes for bundle routing from "
                f"{[p.name for p in plan.start_ports]} to "
                f"{[p.name for p in plan.end_ports]}"
            )

        return routes

    def show_placer_errors(
        self,
        placer_errors: Sequence[Exception],
        error_routes: Sequence[tuple[BasePort, BasePort, list[kdb.Point], int]],
    ) -> None:
        db = rdb.ReportDatabase("Route Placing Errors")
        self.c.name = self.c.kcl._future_cell_name or self.c.name
        cell = db.create_cell(self.c.name)
        for error, (ps, pe, pts, width) in zip(
            placer_errors, error_routes, strict=False
        ):
            cat = db.create_category(f"{ps.name} - {pe.name}")
            it = db.create_item(cell=cell, category=cat)
            it.add_value(
                f"Error while trying to place route from {ps.name} to {pe.name} at"
                f" points (dbu): {pts}"
            )
            it.add_value(f"Exception: {error}")
            path = kdb.Path(pts, width or ps.any_cross_section.width)
            it.add_value(self.c.kcl.to_um(path.polygon()))
        self.c.show(lyrdb=db)

    def record_constraint_routes(self, routes: list[ManhattanRoute]) -> None:
        if self.constraints:
            for constraint in self.constraints:
                constraint._routes[self.route_name] = routes

    def check_collisions(
        self, plan: ManhattanBundlePlan, routes: list[ManhattanRoute]
    ) -> None:
        check_collisions(
            c=self.c,
            start_ports=plan.start_ports,
            end_ports=plan.end_ports,
            on_collision=self.on_collision,
            collision_check_layers=self.collision_check_layers,
            routers=plan.routers,
            routes=routes,
        )

    @staticmethod
    def _normalize_route_steps(
        steps: dbu | list[dbu] | list[Step] | list[list[Step]] | None,
        length: int,
    ) -> Sequence[Sequence[Step]]:
        if steps is None or steps == []:
            return [[]] * length
        if isinstance(steps, int):
            return [[Straight(dist=steps)] for _ in range(length)]
        if _is_steps_list(steps):
            return [steps for _ in range(length)]
        if _is_steps_matrix(steps):
            return steps
        steps = cast("list[int]", steps)
        return [[Straight(dist=s) for s in steps]] * length

    @staticmethod
    def _normalize_port_angles(
        ports: list[BasePort],
        angles: int | list[int] | None,
    ) -> list[BasePort]:
        if angles is None:
            return ports
        if isinstance(angles, int):
            return [
                p.transformed(post_trans=kdb.Trans(angles - p.get_trans().angle))
                for p in ports
            ]
        if not len(angles) == len(ports):
            raise ValueError(
                "If more than one end port should be rotated,"
                " a rotation for all ports must be provided."
            )
        return [
            p.transformed(post_trans=kdb.Trans(a - p.get_trans().angle))
            for a, p in zip(angles, ports, strict=False)
        ]

    @staticmethod
    def _normalize_route_widths(
        route_width: dbu | list[dbu] | None,
        start_ports: Sequence[BasePort],
    ) -> Sequence[int]:
        if route_width:
            if isinstance(route_width, int):
                return [route_width] * len(start_ports)
            return route_width
        return [p.any_cross_section.width for p in start_ports]


def check_collisions(
    c: KCell,
    start_ports: Sequence[BasePort],
    end_ports: Sequence[BasePort],
    routers: Sequence[ManhattanRouter],
    routes: Sequence[ManhattanRoute],
    on_collision: Literal["error", "show_error"] | None = "show_error",
    collision_check_layers: Sequence[kdb.LayerInfo] | None = None,
) -> None:
    """Checks for collisions given manhattan routes.

    Args:
        c: The KCell to check.
        start_ports: Ports from which the routes are supposed to start.
        end_ports: Ports where the routes are supposed to end.
        routers: The ManhattanRouters that constructed the routes.
        routes: The ManhatnnaRoutes which were used by the placer.
        on_collision: What to do on error. Can either do nothing (None),
            throw an error ("error"), or throw an error and open the
            cell with report in Klayout ("show_error").
        collision_check_layers: Sequence of layers which should be checked for
            overlaps to determine error. If not defined, all layers occurring in
            ports will be used.
    """
    if on_collision is None:
        return
    collision_edges: dict[str, kdb.Edges] = {}
    inter_route_collisions = kdb.Edges()
    all_router_edges = kdb.Edges()
    for i, (ps, pe, router) in enumerate(
        zip(start_ports, end_ports, routers, strict=False)
    ):
        edges_, router_edges = router.collisions(log_errors=None)
        if not edges_.is_empty():
            collision_edges[f"{ps.name} - {pe.name} (index: {i})"] = edges_
        inter_route_collision = all_router_edges.interacting(router_edges)
        if not inter_route_collision.is_empty():
            inter_route_collisions.join_with(inter_route_collision)
        all_router_edges.join_with(router_edges)

    if collision_edges or not inter_route_collisions.is_empty():
        if collision_check_layers is None:
            collision_check_layers = list(
                {p.any_cross_section.main_layer for p in start_ports}
            )
        dbu = c.kcl.dbu
        db = rdb.ReportDatabase("Routing Errors")
        cat = db.create_category("Manhattan Routing Collisions")
        c.name = c.kcl._future_cell_name or c.name
        cell = db.create_cell(c.name)
        for name, edges in collision_edges.items():
            item = db.create_item(cell, cat)
            item.add_value(name)
            for edge in edges.each():
                item.add_value(edge.to_dtype(dbu))
        insts = [inst for route in routes for inst in route.instances]
        shapes: dict[kdb.LayerInfo, list[kdb.Region]] = defaultdict(list)
        for route in routes:
            for layer, _shapes in route.polygons.items():
                shapes[layer].append(kdb.Region(_shapes))
        layer_cats: dict[kdb.LayerInfo, rdb.RdbCategory] = {}

        def layer_cat(layer_info: kdb.LayerInfo) -> rdb.RdbCategory:
            if layer_info not in layer_cats:
                layer_cats[layer_info] = db.category_by_path(
                    layer_info.to_s()
                ) or db.create_category(layer_info.to_s())
            return layer_cats[layer_info]

        any_layer_collision = False

        for layer_info in collision_check_layers:
            shapes_regions = shapes[layer_info]
            layer_ = c.kcl.layout.layer(layer_info)
            error_region_instances = kdb.Region()
            error_region_shapes = kdb.Region()
            inst_regions: dict[int, kdb.Region] = {}
            inst_region = kdb.Region()
            shape_region = kdb.Region()
            for r in shapes_regions:
                if not (shape_region & r).is_empty():
                    error_region_shapes.insert(shape_region & r)
                shape_region.insert(r)
            for i, inst in enumerate(insts):
                inst_region_ = kdb.Region(inst.bbox(layer_))
                if not (inst_region & inst_region_).is_empty():
                    # if inst_shapes is None:
                    inst_shapes = kdb.Region()
                    shape_it = c.begin_shapes_rec_overlapping(layer_, inst.bbox(layer_))
                    shape_it.select_cells([inst.cell.cell_index()])
                    shape_it.min_depth = 1
                    for _it in shape_it.each():
                        if _it.path()[0].inst() == inst.instance:
                            inst_shapes.insert(
                                _it.shape().polygon.transformed(_it.trans())
                            )
                    for j, _reg in inst_regions.items():
                        if _reg & inst_region_:
                            reg = kdb.Region()
                            shape_it = c.begin_shapes_rec_touching(
                                layer_, (_reg & inst_region_).bbox()
                            )
                            shape_it.select_cells([insts[j].cell.cell_index()])
                            shape_it.min_depth = 1
                            for _it in shape_it.each():
                                if _it.path()[0].inst() == insts[j].instance:
                                    reg.insert(
                                        _it.shape().polygon.transformed(_it.trans())
                                    )

                            error_region_instances.insert(reg & inst_shapes)
                inst_region += inst_region_
                inst_regions[i] = inst_region_

            if not error_region_shapes.is_empty():
                any_layer_collision = True
                if on_collision == "error":
                    continue
                cat = layer_cat(layer_info)
                sc = db.category_by_path(
                    f"{cat.path()}.RoutingErrors"
                ) or db.create_category(layer_cat(layer_info), "RoutingErrors")
                for poly in error_region_shapes.merge().each():
                    it = db.create_item(cell, sc)
                    it.add_value("Route shapes overlapping with other shapes")
                    it.add_value(c.kcl.to_um(poly.downcast()))
            if not error_region_instances.is_empty():
                any_layer_collision = True
                if on_collision == "error":
                    continue
                cat = layer_cat(layer_info)
                sc = db.category_by_path(
                    f"{cat.path()}.RoutingErrors"
                ) or db.create_category(layer_cat(layer_info), "RoutingErrors")
                for poly in error_region_instances.merge().each():
                    it = db.create_item(cell, sc)
                    it.add_value("Route instances overlapping with other instances")
                    it.add_value(c.kcl.to_um(poly.downcast()))

        if any_layer_collision:
            match on_collision:
                case "show_error":
                    c.show(lyrdb=db)
                    raise RuntimeError(
                        f"Routing collision in {c.kcl._future_cell_name or c.name}"
                    )
                case "error":
                    raise RuntimeError(
                        f"Routing collision in {c.kcl._future_cell_name or c.name}"
                    )


PORTS_FOR_RADIUS = 2


def get_radius(ports: Sequence[ProtoPort[Any]]) -> dbu:
    """Calculates a radius between two ports.

    This can be used to determine the radius of two bend ports.

    Args:
        ports: A sequence of exactly two ports.

    Returns:
        Radius in dbu.

    Raises:
        ValueError: Radius cannot be determined
    """
    ports_ = tuple(p.to_itype() for p in ports)
    if len(ports_) != PORTS_FOR_RADIUS:
        raise ValueError(
            "Cannot determine the maximal radius of a bend with more than two ports."
        )
    p1, p2 = ports_
    if p1.angle == p2.angle:
        return int((p1.trans.disp - p2.trans.disp).length())
    p = kdb.Point(1, 0)
    e1 = kdb.Edge(p1.trans.disp.to_p(), p1.trans * p)
    e2 = kdb.Edge(p2.trans.disp.to_p(), p2.trans * p)

    center = e1.cut_point(e2)
    if center is None:
        raise ValueError("Could not determine the radius. Something went very wrong.")
    return int(
        max((p1.trans.disp - center).length(), (p2.trans.disp - center).length())
    )


def route_bundle(
    *,
    c: KCell,
    start_ports: list[BasePort],
    end_ports: list[BasePort],
    route_width: dbu | list[dbu] | None = None,
    on_collision: Literal["error", "show_error"] | None = "show_error",
    on_placer_error: Literal["error", "show_error"] | None = "show_error",
    collision_check_layers: Sequence[kdb.LayerInfo] | None = None,
    routing_function: ManhattanBundleRoutingFunction = route_smart,
    routing_kwargs: dict[str, Any] | None = None,
    placer_function: PlacerFunction,
    placer_kwargs: dict[str, Any] | None = None,
    constraints: Sequence[Constraint] | None = None,
    starts: dbu | list[dbu] | list[Step] | list[list[Step]] | None = None,
    ends: dbu | list[dbu] | list[Step] | list[list[Step]] | None = None,
    start_angles: int | list[int] | None = None,
    end_angles: int | list[int] | None = None,
    route_debug: RouteDebug | None = None,
    route_name: str | None = None,
) -> list[ManhattanRoute]:
    r"""Route a bundle from starting ports to end_ports.

    Waypoints will create a front which will create ports in a 1D array. If waypoints
    are a transformation it will be like a point with a direction. If multiple points
    are passed, the direction will be invfered.
    For orientation of 0 degrees it will create the following front for 4 ports:

    ```
          │
          │
          │
          p1 ->
          │
          │
          │


          │
          │
          │
          p2 ->
          │
          │
          │
      ___\waypoint
         /
          │
          │
          │
          p3 ->
          │
          │
          │


          │
          │
          │
          p4 ->
          │
          │
          │
    ```

    Args:
        c: Cell to place the route in.
        start_ports: List of start ports.
        end_ports: List of end ports.
        route_width: Width of the route. If None, the width of the ports is used.
        sort_ports: Automatically sort ports.
        on_collision: Define what to do on routing collision. Default behaviour is to
            open send the layout of c to klive and open an error lyrdb with the
            collisions. "error" will simply raise an error. None will ignore any error.
        on_placer_error: If a placing of the components fails, use the strategy above to
            handle the error. show_error will visualize it in klayout with the intended
            route along the already placed parts of c. Error will just throw an error.
            None will ignore the error.
        collision_check_layers: Layers to check for actual errors if manhattan routes
            detect potential collisions.
        routing_function: Function to place the routes. Must return a corresponding list
            of OpticalManhattan routes.
            Must accept the following protocol:
            ```
            routing_function(
                c: KCell, p1: Port, p2: Port, pts: list[Point], **placer_kwargs
            )
            ```
        routing_kwargs: Additional kwargs passed to the placer_function.
        placer_function: Function to place the routes. Must return a corresponding list
            of OpticalManhattan routes.
            Must accept the following protocol:
            ```
            placer_function(
                c: KCell, p1: Port, p2: Port, pts: list[Point], **placer_kwargs
            )
            ```
        placer_kwargs: Additional kwargs passed to the placer_function.
        constraints: Routing constraints to enforce after routing but before placement.
            Each constraint's `enforce` method is called with the routers and routing
            kwargs (e.g. separation, bend90_radius).
        starts: List of steps to use on each starting port or all of them.
        ends: List of steps to use on each end port or all of them.
        start_angles: Overwrite the port orientation of all start_ports together
            (single value) or each one (list of values which is as long as start_ports).
        end_angles: Overwrite the port orientation of all start_ports together
            (single value) or each one (list of values which is as long as end_ports).

    Returns:
        List of ManattanRoutes containing the instances of the route.

    Raises:
        PlacerError: Something went wrong and the resulting route of the placer function
            is not manhattan or the elements cannot be fitted.
        ValueError: Ports or places or args are misconfigured.
    """
    if ends is None:
        ends = []
    if starts is None:
        starts = []
    if placer_kwargs is None:
        placer_kwargs = {}
    if routing_kwargs is None:
        routing_kwargs = {"bbox_routing": "minimal"}
    if route_debug is not None:
        routing_kwargs["route_debug"] = route_debug
    planner = ManhattanBundlePlanner(
        c=c,
        start_ports=start_ports,
        end_ports=end_ports,
        route_width=route_width,
        on_collision=on_collision,
        on_placer_error=on_placer_error,
        collision_check_layers=collision_check_layers,
        routing_function=routing_function,
        routing_kwargs=routing_kwargs,
        placer_function=placer_function,
        placer_kwargs=placer_kwargs,
        constraints=constraints,
        starts=starts,
        ends=ends,
        start_angles=start_angles,
        end_angles=end_angles,
        route_name=route_name,
    )
    inputs = planner.normalize_inputs()
    if not inputs.start_ports:
        return []

    plan = planner.plan(inputs)
    if not plan.routers:
        return []

    planner.enforce_constraints(plan)
    routes = planner.materialize(plan)
    planner.check_collisions(plan, routes)
    planner.record_constraint_routes(routes)
    return routes


def _is_steps_list(
    step_list: list[Step] | list[int] | list[list[Step]],
) -> TypeGuard[list[Step]]:
    return isinstance(step_list[0], Step)


def _is_steps_matrix(
    step_list: list[Step] | list[int] | list[list[Step]],
) -> TypeGuard[list[list[Step]]]:
    return isinstance(step_list[0], list)
