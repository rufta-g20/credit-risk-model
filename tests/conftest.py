import pandas as pd
import pandas.core.algorithms as algos

# Manually add the missing attribute if it's gone
if not hasattr(algos, 'quantile'):
    import numpy as np
    algos.quantile = np.quantile