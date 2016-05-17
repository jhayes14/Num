import pandas as pd
import numpy as np

if __name__ == '__main__':
    tube = pd.read_csv('./data/tube.csv',
                       index_col=0,
                       true_values=['Y'],
                       false_values=['N'])

    materials = pd.read_csv('./data/bill_of_materials.csv', index_col=0)
    specs = pd.read_csv('./data/specs.csv', index_col=0)

    for idx in tube.index:
        if tube.ix[idx, 'material_id'] is not np.nan:
            tube.ix[idx, tube.ix[idx, 'material_id']] = 1
        for i in range(1, 11):
            if i < 9 and materials.ix[idx, 'component_id_%d' % i] is not np.nan:
                tube.ix[idx, materials.ix[idx, 'component_id_%d' % i]] = \
                    materials.ix[idx, 'quantity_%d' % i]
            if specs.ix[idx, 'spec%d' % i] is not np.nan:
                tube.ix[idx, specs.ix[idx, 'spec%d' % i]] = 1
    #tube.drop('material_id', inplace=True, axis=1)
    tube.fillna(0, inplace=True)
    tube.to_csv('tube_extended.csv')