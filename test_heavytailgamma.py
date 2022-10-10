import numpy as np

from qsimpy_aqm.random import HeavyTailGamma

service = HeavyTailGamma(
    seed=102,
    gamma_concentration=5,
    gamma_rate=0.5,
    gpd_concentration=0.3,
    threshold_qnt=0.8,
    dtype="float64",
    batch_size=1000,
    be_quiet=False,
)
service.prepare_for_run()

n = 10000
ldp = 0.9

samples = service.sample_n(n)
print(len(samples))
quant = np.quantile(samples, 1.00 - ldp)

print(quant)

# sample conditioned on a longer_delay_prob
remaining_service_delay = service.sample_ldp(ldp=ldp)

print(remaining_service_delay)
