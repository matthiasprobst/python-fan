import unittest

import matplotlib.pyplot as plt
import numpy as np

import pyfan


def create_op(vfr, dptot, etatot, revspeed, fan_properties, n=100):
    """create artificial op"""
    return dict(vfr=np.random.normal(vfr, 0.1 * vfr, n),
                dptot=np.random.normal(dptot, 0.1 * dptot, n),
                etatot=np.random.normal(etatot, 0.1, n),
                revspeed=np.random.normal(revspeed, 5., n),
                fan_properties=fan_properties)


class TestOP(unittest.TestCase):

    def setUp(self) -> None:
        self.n = 100
        self.fan_properties = pyfan.FanProperties(D2=0.2)

    def test_op(self):
        op = pyfan.FanOperationPoint(
            **create_op(0.05, 100, 0.7, 1000, self.fan_properties, self.n)
        )
        self.assertIsInstance(op.dptot, pyfan.FanTotalPressureDifference)

    def test_fancurve(self):
        ops = [pyfan.FanOperationPoint(
            **create_op(vfr, dptot, etatot, revspeed=1000, fan_properties=self.fan_properties, n=self.n))
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
        ax = fc.plot_pressure_curve('rs--', dimless=True)
        _ = fc.plot_efficiency_curve('bs--', ax=ax.twinx())
        plt.tight_layout()
        plt.show()

        plt.figure()
        fc.plot_pressure_curve('rs--', label='n=1000 1/min')
        fc.affine(800).plot_pressure_curve('bs--', label='n=800 1/min')
        fc.affine(600).plot_pressure_curve('ys--', label='n=600 1/min')
        plt.legend()
        plt.tight_layout()
        plt.show()
