import matplotlib.pyplot as plt
import numpy as np
import unittest
import xarray as xr

import pyfan


def create_op(vfr, dptot, etatot, revspeed, fan_properties, n=100):
    """create artificial op"""
    return dict(vfr=np.random.normal(vfr, 0.1 * vfr, n),
                density=1.2,
                pressure_difference=pyfan.TotalPressureDifference(np.random.normal(dptot, 0.1 * dptot, n)),
                torque=np.random.normal(etatot, 0.1, n),
                revspeed=np.random.normal(revspeed, 5., n),
                fan_properties=fan_properties)


class TestOP(unittest.TestCase):

    def setUp(self) -> None:
        self.n = 100
        self.fan_properties = pyfan.FanProperties(D2=0.2, area_inlet=1, area_outlet=1)

    def test_op(self):
        with self.assertRaises(TypeError):
            op = pyfan.FanOperationPoint(vfr=0, pressure_difference=60,
                                         density=1.2,
                                         revspeed=10, fan_properties=self.fan_properties)
        op = pyfan.FanOperationPoint(vfr=0,
                                     pressure_difference=pyfan.TotalPressureDifference(60.),
                                     density=1.2,
                                     torque=0.5,
                                     revspeed=10, fan_properties=self.fan_properties)
        self.assertIsInstance(op.dptot, pyfan.FanTotalPressureDifference)
        self.assertIsInstance(op.dptot.values, float)
        self.assertEqual(op.dpstat, None)

    def test_xr(self):
        data = create_op(0.05, 100, 0.7, 1000, self.fan_properties, self.n)
        op = pyfan.FanOperationPoint(**data)
        self.assertIsInstance(op.dptot.values, np.ndarray)
        data['time_vector'] = np.arange(0, len(data['pressure_difference']))
        op = pyfan.FanOperationPoint(**data)
        self.assertIsInstance(op.dptot.values, xr.DataArray)
        op.dptot.values.plot()
        plt.show()

    def test_fancurve(self):
        data = create_op(0.05, 100, 0.7, 1000, self.fan_properties, self.n)
        ops = [pyfan.FanOperationPoint(**data)
               for (vfr, dptot, etatot) in
               zip(
                   (0.01, 0.03, 0.05, 0.08),
                   (67.8, 70.4, 61.1, 40.),
                   (0.3, 0.6, 0.7, 0.54)
               )
               ]
        fc = pyfan.FanCurve(ops)
        self.assertIsInstance(fc[0].dptot, pyfan.FanTotalPressureDifference)

        plt.figure()
        ax = fc.plot_total_pressure_curve('rs--', dimless=True)
        _ = fc.plot_efficiency_curve('bs--', ax=ax.twinx())
        plt.tight_layout()
        plt.show()

        plt.figure()
        fc.plot_total_pressure_curve('rs--', label='n=1000 1/min')
        fc.affine(800 / 60).plot_total_pressure_curve('bs--', label='n=800 1/min')
        fc.affine(10).plot_total_pressure_curve('ys--', label='n=600 1/min')
        plt.legend()
        plt.tight_layout()
        plt.show()
