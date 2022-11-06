"""Core module"""

import enum
from dataclasses import dataclass
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import dimless


@dataclass
class FanProperties:
    """Fan properties class"""
    D2: float


class _FanIntegralQuantity:

    def __getitem__(self, item):
        return FanTotalPressureDifference(self.vfr[item],
                                          self.values[item])

    def plot(self, *args, **kwargs):
        """Plot the total pressure difference. x value is the volume flow rate"""
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        ax.plot(self.vfr, self.values, *args, **kwargs)
        ax.set_xlabel('Volume flow rate / $m^3/s$')
        return ax

    def mean(self) -> Tuple[float, float]:
        """Calls np.nanmean"""
        return np.nanmean(self.vfr), np.nanmean(self.values)

    def std(self) -> Tuple[float, float]:
        """Calls np.nanstd"""
        return np.nanstd(self.vfr), np.nanstd(self.values)


class FanTotalEfficiency(_FanIntegralQuantity):
    """Fan Total Efficiency Class"""

    def __init__(self,
                 vfr: Union[float, np.ndarray],
                 etatot: Union[float, np.ndarray]):
        self.vfr = vfr
        self.values = etatot

    def plot(self, *args, **kwargs):
        """Plot the total pressure difference. x value is the volume flow rate"""
        ax = super().plot(*args, **kwargs)
        ax.set_ylabel('Fan total efficiency / -')
        return ax


class FanTotalPressureDifference(_FanIntegralQuantity):
    """Fan Total Pressure Difference Class"""

    def __init__(self,
                 vfr: Union[float, np.ndarray],
                 dptot: Union[float, np.ndarray]):
        self.vfr = vfr
        self.values = dptot

    def plot(self, *args, **kwargs):
        """Plot the total pressure difference. x value is the volume flow rate"""
        ax = super().plot(*args, **kwargs)
        ax.set_ylabel('Fan total pressure difference / $Pa$')
        return ax


class FanPsi(_FanIntegralQuantity):
    """Pressure Number Class"""

    def __init__(self,
                 vfr: Union[float, np.ndarray],
                 psi: Union[float, np.ndarray]):
        self.vfr = vfr
        self.values = psi

    def plot(self, *args, **kwargs):
        """Plot the total pressure difference. x value is the volume flow rate"""
        ax = super().plot(*args, **kwargs)
        ax.set_ylabel('Pressure Coefficient / -')
        return ax


class FanPhi(_FanIntegralQuantity):
    """Flow Coefficient Class"""

    def __init__(self,
                 vfr: Union[float, np.ndarray],
                 phi: Union[float, np.ndarray]):
        self.vfr = vfr
        self.values = phi

    def plot(self, *args, **kwargs):
        """Plot the total pressure difference. x value is the volume flow rate"""
        ax = super().plot(*args, **kwargs)
        ax.set_ylabel('Flow Number / -')
        return ax


class FanOperationPoint:
    """Fan Operation Point class"""

    def __init__(self,
                 vfr: np.ndarray = None,
                 dptot: np.ndarray = None,
                 etatot: np.ndarray = None,
                 revspeed: Union[float, np.ndarray] = None,
                 fan_properties: FanProperties = None,
                 density: Union[float, np.ndarray] = 1.2
                 ):
        """Fan Operation Point class.
        Takes differential pressure data (difference between outlet and inlet).

        At this stage of development input data is expected not to come with a unit.
        In future this may change, e.g. because the user passes xarray.

        Attributes
        ----------
        vfr: Union[float, np.ndarray]
            Volume flow rate in [m^3/s]
        dptot: Union[float, np.ndarray]
            Total pressure difference between outlet and inlet of the fan
        etatot: Union[float, np.ndarray]
            Total efficiency of the fan.
        revspeed: Union[float, np.ndarray]
            Revolution speed of the fan.
        fan_properties: FanProperties
            The properties of the fan, like outer diameter D2 etc.
        """
        # correct negative vfr
        if np.median(vfr) < 0.:
            self._vfr = -1 * vfr
        else:
            self._vfr = vfr
        self._dptot = dptot
        self._etatot = etatot
        self._rho = density
        self._revspeed = revspeed
        if not isinstance(fan_properties, FanProperties):
            raise TypeError(f'Expect class FanProperties for parameter fan_properties but git {type(fan_properties)}')
        self._fan_properties = fan_properties

    def __lt__(self, other) -> bool:
        return np.mean(self._vfr) < np.mean(other.vfr)

    def __getitem__(self, item) -> "FanOperationPoint":
        return FanOperationPoint(self.vfr[item],
                                 self.dptot[item],
                                 self.etatot[item],
                                 self.revspeed[item],
                                 self.fan_properties)

    @property
    def vfr(self) -> Union[float, np.ndarray]:
        """Volume flow rate"""
        return self._vfr

    @property
    def dptot(self) -> FanTotalPressureDifference:
        """Fan total pressure difference"""
        return FanTotalPressureDifference(self._vfr, self._dptot)

    @property
    def etatot(self) -> FanTotalEfficiency:
        """Fan total efficiency"""
        return FanTotalEfficiency(self._vfr, self._etatot)

    def plot_fan_total_pressure(self, ax=None, mean=False, marker='+', color=None):
        """Plot fan total pressure"""
        if ax is None:
            ax = plt.gca()
        if mean:
            ax.plot(np.mean(self.vfr), np.mean(self.dptot), marker=marker, color=color)
        else:
            ax.plot(self.vfr, self.dptot, marker=None, color=color)
        return ax

    @property
    def psi(self):
        """Head coefficient (dimensionless number)"""
        return FanPsi(self._vfr,
                      dimless.psi(self._dptot,
                                  self._revspeed,
                                  self._rho,
                                  self._fan_properties.D2)
                      )

    @property
    def phi(self) -> Union[float, np.ndarray]:
        """Flow coefficient (dimensionles number)"""
        return FanPhi(self._vfr,
                      dimless.phi(self._vfr,
                                  self._revspeed,
                                  self._fan_properties.D2)
                      )

    def affine(self, n_new: float) -> "FanOperationPoint":
        """Return an affine operation point with new revolution speed n_new"""
        return FanOperationPoint(self._vfr * n_new / self._revspeed,
                                 self._dptot * n_new ** 2 / self._revspeed ** 2,
                                 self._etatot,
                                 self._revspeed,
                                 fan_properties=self._fan_properties,
                                 density=self._rho)


class DataSource(enum.Enum):
    """Data source"""
    experimental = 1
    numerical = 2


class ExperimentalOperationPoint(FanOperationPoint):
    source: DataSource.experimental


@dataclass
class FanCurve:
    """Fan (characteristic) curve class."""
    operation_points: List[FanOperationPoint]
    name: str = None

    def __repr__(self) -> str:
        return f'<FanCurve nop={self.__len__()}>'

    def __len__(self) -> int:
        return len(self.operation_points)

    def __getitem__(self, item) -> Union[FanOperationPoint,
                                         "FanCurve"]:
        """Return a single operation point or multiple ones
        as a 'new' FanCurve"""
        if isinstance(item, int):
            return self.operation_points[item]
        else:
            return FanCurve(self.operation_points[item],
                            self.name)

    def __post_init__(self):
        self.operation_points = sorted(self.operation_points)

    def plot_pressure_curve(self, *args, **kwargs):
        """Plot the pressure fan curve"""
        ax = kwargs.pop('ax', None)
        dimless = kwargs.pop('dimless', False)
        if ax is None:
            ax = plt.gca()
        if dimless:
            op_mean = [op.psi.mean() for op in self.operation_points]
            op_std = [op.psi.std() for op in self.operation_points]
        else:
            op_mean = [op.dptot.mean() for op in self.operation_points]
            op_std = [op.dptot.std() for op in self.operation_points]
        ax.errorbar([opm[0] for opm in op_mean],
                    [opm[1] for opm in op_mean],
                    [ops[1] for ops in op_std],
                    [ops[0] for ops in op_std],
                    *args, **kwargs)
        ax.set_xlabel('Volume flow rate / $m^3/s$')
        if dimless:
            ax.set_ylabel('Pressure coefficient / $-$')
        else:
            ax.set_ylabel('Fan total pressure difference / $Pa$')
        return ax

    def plot_efficiency_curve(self, *args, **kwargs):
        """Plot the fan efficiency curve"""
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        op_mean = [op.etatot.mean() for op in self.operation_points]
        op_std = [op.etatot.std() for op in self.operation_points]
        ax.errorbar([opm[0] for opm in op_mean],
                    [opm[1] for opm in op_mean],
                    [ops[1] for ops in op_std],
                    [ops[0] for ops in op_std],
                    *args, **kwargs)
        ax.set_xlabel('Volume flow rate / $m^3/s$')
        ax.set_ylabel('Fan total efficiency / $-$')

    def affine(self, n_new: float) -> "FanCurve":
        """Return an affine fan curve with new revolution speed `n_new`"""
        return FanCurve(operation_points=[op.affine(n_new) for op in self.operation_points],
                        name=self.name)
