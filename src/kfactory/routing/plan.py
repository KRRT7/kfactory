from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .. import kdb
    from ..port import Port
    from ..typings import dbu
    from .route_ports import RoutePort


@dataclass(slots=True)
class StraightRouteOp:
    p1: Port | RoutePort
    p2: Port | RoutePort
    width: int
    route_width: int | None
    update_start: bool = False
    update_end: bool = False


@dataclass(slots=True)
class TaperedStraightRouteOp:
    p1: Port | RoutePort
    p2: Port | RoutePort
    width: int
    route_width: int | None
    taper_ports: tuple[Port, Port]
    update_start: bool = False
    update_end: bool = False


@dataclass(slots=True)
class SBendRouteOp:
    p1: Port | RoutePort
    p2: Port | RoutePort
    width: int
    update_start: bool = False


@dataclass(slots=True)
class CreateBend90RouteOp:
    trans: kdb.Trans


@dataclass(slots=True)
class AppendBend90RouteOp:
    pass


@dataclass(slots=True)
class SetRouteEndpointOp:
    endpoint: Literal["start", "end"]
    port: Port | RoutePort
    copy_port: bool


type RouteOp = (
    StraightRouteOp
    | TaperedStraightRouteOp
    | SBendRouteOp
    | CreateBend90RouteOp
    | AppendBend90RouteOp
    | SetRouteEndpointOp
)


@dataclass(slots=True)
class OpticalRoutePlan:
    backbone: list[kdb.Point]
    start_port: Port
    end_port: Port
    width: int
    bend90_radius: dbu = 0
    taper_length: dbu = 0
    ops: list[RouteOp] = field(default_factory=list)
    name_endpoints: bool = False
