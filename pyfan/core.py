"""Core module"""
import enum
import pathlib
from abc import ABC
from dataclasses import dataclass
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import dimless


@dataclass
class FanProperties:
    """Fan properties class"""
    D2: float
    area_inlet: float
    area_outlet: float


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


class FanStaticPressureDifference(_FanIntegralQuantity):
    """Fan Total Pressure Difference Class"""

    def __init__(self,
                 vfr: Union[float, np.ndarray],
                 dpstat: Union[float, np.ndarray]):
        self.vfr = vfr
        self.values = dpstat

    def plot(self, *args, **kwargs):
        """Plot the total pressure difference. x value is the volume flow rate"""
        ax = super().plot(*args, **kwargs)
        ax.set_ylabel('Fan static pressure difference / $Pa$')
        return ax


class FanRevolutionSpeed(_FanIntegralQuantity):
    """Fan Total Pressure Difference Class"""

    def __init__(self,
                 vfr: Union[float, np.ndarray],
                 revspeed: Union[float, np.ndarray]):
        self.vfr = vfr
        self.values = revspeed

    def plot(self, *args, **kwargs):
        """Plot the revolution speed"""
        ax = super().plot(*args, **kwargs)
        ax.set_ylabel('Fan revolution speed / $1/s$')
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
        ax.set_xlabel('Flow Number / -')
        return ax


class Pressure(ABC):
    """Pressure (abstract) class"""

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)


class PressureDifference(Pressure):
    """Pressure difference between two points"""


class TotalPressure(Pressure):
    """Total pressure (static+dynamic)"""


class TotalPressureDifference(Pressure):
    """Total pressure difference between two points"""


class StaticPressure(Pressure):
    """Static pressure"""


class StaticPressureDifference(Pressure):
    """Static pressure difference between two points"""


class DynamicPressure(Pressure):
    """Dynamic Pressure"""


class FanOperationPoint:
    """Fan Operation Point class
    Units are expected to be SI units!
    """

    def __init__(self,
                 vfr: np.ndarray,
                 pressure_difference: Union[PressureDifference, List[PressureDifference]],
                 revspeed: Union[float, np.ndarray],
                 density: Union[float, np.ndarray],
                 torque: np.ndarray,
                 fan_properties: FanProperties,
                 time_vector: Union[np.ndarray] = None
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

        if isinstance(pressure_difference, list):
            # expecting two entries max!
            if len(pressure_difference) != 2:
                raise ValueError('Expecting exactly 2 entries')

            if isinstance(pressure_difference[0], TotalPressureDifference):
                _dptot = pressure_difference[0].values
            elif isinstance(pressure_difference[0], StaticPressureDifference):
                _dpstat = pressure_difference[0].values
            else:
                raise TypeError('Expected TotalPressureDifference or StaticPressureDifference')

            if isinstance(pressure_difference[1], TotalPressureDifference):
                _dptot = pressure_difference[1].values
            elif isinstance(pressure_difference[1], StaticPressureDifference):
                _dpstat = pressure_difference[1].values
            else:
                raise TypeError('Expected TotalPressureDifference or StaticPressureDifference')

            if time_vector is not None:
                xr_time_vector = xr.DataArray(dims='time', data=time_vector,
                                              attrs={'long_name': 'time',
                                                     'units': 's'})
                self._dptot = xr.DataArray(dims='time', data=_dptot,
                                           coords={'time': xr_time_vector},
                                           attrs={'long_name': 'Total pressure difference',
                                                  'units': 'Pa'})

                self._dpstat = xr.DataArray(dims='time', data=_dpstat,
                                            coords={'time': xr_time_vector},
                                            attrs={'long_name': 'Static pressure difference',
                                                   'units': 'Pa'})
            else:
                self._dpstat = _dpstat
                self._dptot = _dptot
        else:
            if not isinstance(pressure_difference, (TotalPressureDifference,
                                                    StaticPressureDifference)):
                raise TypeError(
                    'Expecting data of type TotalPressureDifference or StaticPressureDifference.'
                    ' Please provide data as TotalPressureDifference or '
                    f'StaticPressureDifference but not {type(pressure_difference)}'
                )

            self._fan_properties = fan_properties

            if isinstance(pressure_difference, StaticPressureDifference):
                print('Computing total pressure difference based on static')
                p_dyn_out = density / 2 * (vfr / self._fan_properties.area_inlet) ** 2
                p_dyn_in = density / 2 * (vfr / self._fan_properties.area_outlet) ** 2
                dptot = pressure_difference.values + p_dyn_out - p_dyn_in
                if time_vector is not None:
                    xr_time_vector = xr.DataArray(dims='time', data=time_vector,
                                                  attrs={'long_name': 'time',
                                                         'units': 's'})
                    self._dptot = xr.DataArray(dims='time', data=dptot,
                                               coords={'time': xr_time_vector},
                                               attrs={'long_name': 'Total pressure difference',
                                                      'units': 'Pa'})

                    self._dpstat = xr.DataArray(dims='time', data=pressure_difference.values,
                                                coords={'time': xr_time_vector},
                                                attrs={'long_name': 'Static pressure difference',
                                                       'units': 'Pa'})
                else:
                    self._dptot = dptot
                    self._dpstat = pressure_difference.values
            elif isinstance(pressure_difference, TotalPressureDifference):
                self._dpstat = None
                if time_vector is not None:
                    xr_time_vector = xr.DataArray(dims='time', data=time_vector,
                                                  attrs={'long_name': 'time',
                                                         'units': 's'})
                    self._dptot = xr.DataArray(dims='time', data=pressure_difference.values,
                                               coords={'time': xr_time_vector},
                                               attrs={'long_name': 'Total pressure difference',
                                                      'units': 'Pa'})
                else:
                    self._dptot = pressure_difference.values

        self._torque = torque
        self._etatot = None
        self._fluid_power = None
        self._mech_power = None
        self._rho = density
        self._revspeed = revspeed
        if not isinstance(fan_properties, FanProperties):
            raise TypeError(f'Expect class FanProperties for parameter fan_properties but git {type(fan_properties)}')

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
    def dpstat(self) -> FanTotalPressureDifference:
        """Fan total pressure difference"""
        if self._dpstat is None:
            return None
        return FanStaticPressureDifference(self._vfr, self._dpstat)

    @property
    def revspeed(self):
        """Fan revolution speed"""
        return FanRevolutionSpeed(self._vfr, self._revspeed)

    @property
    def etatot(self) -> FanTotalEfficiency:
        """Fan total efficiency"""
        if self._etatot is None:
            self._fluid_power = np.nanmean(self._vfr * self._dptot)
            self._mech_power = np.nanmean(self._torque) * 2 * np.pi * np.nanmean(self._revspeed)
            self._etatot = self._fluid_power / self._mech_power
        return FanTotalEfficiency(self._vfr, np.ones_like(self._vfr) * self._etatot)

    def plot_fan_total_pressure(self, mean=False, **kwargs):
        """Plot fan total pressure"""
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        if mean:
            return ax.plot(np.mean(self.vfr), np.mean(self.dptot), **kwargs)
        return ax.plot(self.vfr, self.dptot, **kwargs)

    @property
    def psi(self):
        """Head coefficient (dimensionless number)"""
        return dimless.psi(dp=self._dptot,
                           n=self._revspeed,
                           rho=self._rho,
                           D=self._fan_properties.D2)

    @property
    def phi(self) -> Union[float, np.ndarray]:
        """Flow coefficient (dimensionless number)"""
        return dimless.phi(self._vfr,
                           self._revspeed,
                           self._fan_properties.D2)

    def affine(self, n_new: float) -> "FanOperationPoint":
        """Return an affine operation point with new revolution speed n_new"""
        new_data = dict(vfr=self._vfr * n_new / self._revspeed,
                        density=self._rho,
                        revspeed=n_new,
                        torque=self._torque,
                        fan_properties=self._fan_properties)
        if self._dpstat is not None and self._dptot is not None:
            pressure_difference = [TotalPressureDifference(self._dptot * n_new ** 2 / self._revspeed ** 2),
                                   StaticPressureDifference(self._dpstat * n_new ** 2 / self._revspeed ** 2),
                                   ]
        elif self._dpstat is not None:
            pressure_difference = TotalPressureDifference(self._dptot * n_new ** 2 / self._revspeed ** 2)
        elif self._dptot is not None:
            pressure_difference = StaticPressureDifference(self._dptot * n_new ** 2 / self._revspeed ** 2)
        new_data['pressure_difference'] = pressure_difference
        return FanOperationPoint(**new_data)


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

    @property
    def vfr(self) -> np.ndarray:
        """average volume flow rate of every op"""
        return [np.nanmean(op._vfr) for op in self.operation_points]

    @property
    def dptot(self) -> np.ndarray:
        """average total pressure difference of every op"""
        return [np.nanmean(op._dptot) for op in self.operation_points]

    @property
    def dpstat(self) -> np.ndarray:
        """average static pressure difference of every op"""
        return [np.nanmean(op._dpstat) for op in self.operation_points]

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

    def plot_total_pressure_curve(self, *args, **kwargs):
        """Plot the pressure fan curve"""
        ax = kwargs.pop('ax', None)
        plot_dimless = kwargs.pop('dimless', False)
        if ax is None:
            ax = plt.gca()
        if plot_dimless:
            psi_mean = [np.nanmean(op.psi) for op in self.operation_points]
            phi_mean = [np.nanmean(op.phi) for op in self.operation_points]
            psi_std = [np.nanstd(op.psi) for op in self.operation_points]
            phi_std = [np.nanstd(op.phi) for op in self.operation_points]
            ax.errorbar(phi_mean,
                        psi_mean,
                        phi_std,
                        psi_std,
                        *args, **kwargs)
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

    def plot_static_pressure_curve(self, *args, **kwargs):
        """Plot the static pressure fan curve if exists"""
        ax = kwargs.pop('ax', None)
        plot_dimless = kwargs.pop('dimless', False)
        if ax is None:
            ax = plt.gca()
        if plot_dimless:
            psi_mean = [np.nanmean(op.psi) for op in self.operation_points]
            phi_mean = [np.nanmean(op.phi) for op in self.operation_points]
            psi_std = [np.nanstd(op.psi) for op in self.operation_points]
            phi_std = [np.nanstd(op.phi) for op in self.operation_points]
            ax.errorbar(phi_mean,
                        psi_mean,
                        phi_std,
                        psi_std,
                        *args, **kwargs)
        else:
            op_mean = [op.dpstat.mean() for op in self.operation_points]
            op_std = [op.dpstat.std() for op in self.operation_points]
            ax.errorbar([opm[0] for opm in op_mean],
                        [opm[1] for opm in op_mean],
                        [ops[1] for ops in op_std],
                        [ops[0] for ops in op_std],
                        *args, **kwargs)
        ax.set_xlabel('Volume flow rate / $m^3/s$')
        if dimless:
            ax.set_ylabel('Pressure coefficient / $-$')
        else:
            ax.set_ylabel('Fan static pressure difference / $Pa$')
        return ax

    def plot_efficiency_curve(self, *args, **kwargs):
        """Plot the fan efficiency curve"""
        ax = kwargs.pop('ax', None)
        plot_dimless = kwargs.pop('dimless', False)
        if ax is None:
            ax = plt.gca()
        if plot_dimless:
            eta_mean = [np.nanmean(op.etatot.values) for op in self.operation_points]
            phi_mean = [np.nanmean(op.phi) for op in self.operation_points]
            eta_std = [np.nanstd(op.etatot.values) for op in self.operation_points]
            phi_std = [np.nanstd(op.phi) for op in self.operation_points]
            ax.errorbar(phi_mean,
                        eta_mean,
                        phi_std,
                        eta_std,
                        *args, **kwargs)
            ax.set_xlabel('phi / $-$')
        else:
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

    def write_csv(self,
                  filename: pathlib.Path,
                  list_of_variables: List[str],
                  overwrite: bool = False) -> pathlib.Path:
        """Write csv file"""
        filename = pathlib.Path(filename)
        if filename.exists() and not overwrite:
            raise FileNotFoundError('File exists and overwrite is set to False!')
        variables = [self.__getattribute__(name) for name in list_of_variables]
        if not all([isinstance(_var, list) for _var in variables]):
            raise TypeError('Expecting all variables to be list!')
        with open(filename, 'w') as f:
            f.write(f','.join(list_of_variables))
            for idx in range(len(variables[0])):
                line = f','.join([str(variables[ivar][idx]) for ivar in range(len(variables))])
                f.write(f'\n{line}')
        return filename
