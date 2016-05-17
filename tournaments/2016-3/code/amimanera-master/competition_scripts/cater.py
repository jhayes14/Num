#!/usr/bin/python
# coding: utf-8

from qsprLib import *
from lasagne_tools import *
from keras_tools import *

from interact_analysis import *

import matplotlib.pyplot as plt
import math
import os
# pd.options.display.mpl_style = 'default'

from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean, seuclidean



def f_hyperbel(x, a, b):
    return -1.0 * a / x + b


def f_lin(x, a, b):
    return a * x + b


def f_quad(x, a, b, c):
    return a * x + b * x ** 2 + c


def loadData(nsamples=-1, verbose=False, useRdata=False, useFrequencies=False, concat=True, balance=None,
             bagofwords=None, bagofwords_v2_0=None, endform=True, comptypes=0, logtransform=None, NA_filler=0,
             skipLabelEncoding=None, useTubExtended=False, encodeKeepNumber=False, loadBN=False):
    # load training and test datasets
    comp_cols = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5',
                 'component_id_6', 'component_id_7', 'component_id_8']

    if isinstance(loadBN, str):
        print "Loading BN data..."

        if loadBN in "logtransformed" or loadBN in "nn":
            Xtrain = pd.read_csv("./data/training_Aug23_log.csv")
            Xtest = pd.read_csv("./data/testing_Aug23_log.csv")
        else:
            Xtrain = pd.read_csv("./data/training_Aug15_v2.csv")
            Xtest = pd.read_csv("./data/testing_Aug15_v2.csv")
        Xtrain.drop(['cost'], axis=1, inplace=True)
        Xtest.drop(['cost'], axis=1, inplace=True)
        print Xtrain.shape
        print Xtest.shape

        print list(Xtrain.columns)
        # return Xtest,Xtrain
        Xtrain['tube_assembly_id'] = pd.read_csv("./data/train_set.csv", usecols=['tube_assembly_id'])
        Xtest['tube_assembly_id'] = "NA"
        y = pd.read_csv("./data/train_set.csv", usecols=['cost']).values

        idx = pd.read_csv('./data/test_set.csv', usecols=['id']).values.astype(int)

        Xall = pd.concat([Xtest, Xtrain])
        # raw_input()

    if not useRdata and not loadBN and not isinstance(loadBN, str):
        print "Generating data from scratch..."
        Xtrain = pd.read_csv('./data/train_set.csv', parse_dates=[2, ])
        Xtest = pd.read_csv('./data/test_set.csv', parse_dates=[3, ])

        if useTubExtended:
            print "Extended data!"
            tube_data = pd.read_csv('./data/tube_extended.csv')
            # tube_data = pd.read_csv('./data/tube.csv')
            # verbose=True
        else:
            tube_data = pd.read_csv('./data/tube.csv')
        bill_of_materials_data = pd.read_csv('./data/bill_of_materials.csv')
        specs_data = pd.read_csv('./data/specs.csv')

        components = pd.read_csv('./data/components.csv')

        y = Xtrain.cost.values
        Xtrain.drop(['cost'], axis=1, inplace=True)

        idx = Xtest.id.values.astype(int)
        Xtest.drop(['id'], axis=1, inplace=True)

        if verbose:
            print("train columns")
            print(Xtrain.columns)
            print Xtrain.shape
            print Xtrain.describe(include='all')
            # print Xtrain
            raw_input()

            print("test columns")
            print(Xtest.columns)
            print Xtest.shape
            print Xtest.describe(include='all')
            # print Xtest
            raw_input()

            print("tube.csv df columns")
            print(tube_data.columns)
            print tube_data.shape
            print tube_data.describe(include='all')
            # print tube_data
            raw_input()

            print("bill_of_materials.csv df columns")
            print(bill_of_materials_data.columns)
            print bill_of_materials_data.shape
            print bill_of_materials_data.describe(include='all')
            # print bill_of_materials_data
            raw_input()

            print("specs.csv df columns")
            print(specs_data.columns)
            print specs_data.shape
            print specs_data.describe(include='all')
            # print specs_data
            raw_input()

        Xall = pd.concat([Xtest, Xtrain])

        Xall = pd.merge(Xall, tube_data, on='tube_assembly_id')
        Xall = pd.merge(Xall, bill_of_materials_data, on='tube_assembly_id')
        Xall = pd.merge(Xall, specs_data, on='tube_assembly_id')

        if endform:
            end_form = pd.read_csv('./data/tube_end_form.csv')
            end_form.rename(columns={'end_form_id': 'end_a', 'forming': 'forming_enda'}, inplace=True)
            Xall = pd.merge(Xall, end_form, on='end_a', how='left')
            end_form.rename(columns={'end_a': 'end_x', 'forming_enda': 'forming_endx'}, inplace=True)
            Xall = pd.merge(Xall, end_form, on='end_x', how='left')
            print Xall[['forming_enda', 'forming_endx']].describe(include='all')
            print Xall.forming_enda.value_counts()
            print Xall.forming_endx.value_counts()

        if comptypes > 0:
            n = Xall.shape[1]
            print "Loading all component files...columns before", n
            keyMerge = 0
            for cid in xrange(1, 9):
                if cid > comptypes: break
                for filenname in os.listdir("./data"):
                    if filenname.startswith("comp_"):
                        print(filenname)
                        compl = 'component_id_' + str(cid)
                        print "Compl:", compl,
                        compr = 'component_id_' + str(cid)
                        print "Compr:", compr
                        keyMerge = filenname.replace("comp_", "").replace(".csv", "") + "_" + compr
                        print "keyMerge:", keyMerge
                        # print "File:",filenname,
                        _comp_data = pd.read_csv('./data/' + filenname)

                        _comp_data.columns = [col + '_' + str(keyMerge) for col in _comp_data.columns]
                        _comp_data.rename(columns={'component_id_' + str(keyMerge): compl}, inplace=True)

                        # do label encoding directly!
                        for col in list(_comp_data.select_dtypes(include=['object']).columns):
                            lbl = preprocessing.LabelEncoder()
                            if col <> compl:
                                _comp_data[col] = lbl.fit_transform(_comp_data[col].values)

                        # print _comp_data.shape
                        # print _comp_data.head(20)
                        # print _comp_data.describe(include='all')
                        # raw_input()

                        Xall = pd.merge(Xall, _comp_data, on=compl, how='left')
                        # complexity


                        # print Xall.columns
                        # print Xall.shape
                        # print Xall.head()
                        # keyMerge+=1

                        # raw_input()
            # print Xall.iloc[:,n:].columns
            Xall['comp_complexity'] = (Xall.iloc[:, n:] != 0).astype(int).sum(axis=1)
            # Xall['complexity_std'] = (Xall.iloc[:,n:] != 0).astype(int).std(axis=1)
            # print Xall._get_numeric_data().columns
            print "Loaded all component files...columns after", Xall.shape[1]
            # raw_input()

        # create some new features
        Xall['year'] = Xall.quote_date.dt.year
        Xall['month'] = Xall.quote_date.dt.month
        # Xall["dow"]   = Xall.apply(lambda row: row["quote_date"].dayofweek, axis=1)
        # Xall['dayofyear'] = Xall.quote_date.dt.dayofyear
        # Xall['dayofweek'] = Xall.quote_date.dt.dayofweek
        # Xall['day'] = Xall.quote_date.dt.day
        # Xall['week'] = Xall.quote_date.dt.dayofyear % 52




        # Boruta: 2 attributes confirmed unimportant: spec10, spec9.
        Xall.drop(['quote_date', 'spec9', 'spec10'], axis=1, inplace=True)

        Xall['material_id'].replace(np.nan, ' ', regex=True, inplace=True)
        for i in range(1, 9):
            column_label = 'component_id_' + str(i)
            Xall[column_label].replace(np.nan, ' ', regex=True, inplace=True)

        for i in range(1, 9):
            column_label = 'spec' + str(i)
            Xall[column_label].replace(np.nan, ' ', regex=True, inplace=True)


        # corrections
        Xall.fillna(NA_filler, inplace=True)
        correctLengths = True
        if correctLengths:  # https://www.kaggle.com/c/caterpillar-tube-pricing/forums/t/15001/ta-04114
            corr = {}
            corr['TA-00152'] = 19
            corr['TA-00154'] = 75
            corr['TA-00156'] = 24
            corr['TA-01098'] = 10
            corr['TA-01631'] = 48
            corr['TA-03520'] = 46
            corr['TA-04114'] = 135
            corr['TA-17390'] = 40
            corr['TA-18227'] = 74
            corr['TA-18229'] = 51
            for i in xrange(Xall.shape[0]):
                tube_id = Xall['tube_assembly_id'].iloc[i]
                if tube_id in corr.keys():
                    Xall.ix[i, ['length']] = corr[tube_id]

        # b=['supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']

        # quantity / specs?
        if balance is not None:
            Xtrain = Xall[len(Xtest.index):]
            Xtest = Xall[:len(Xtest.index)]
            for col in Xall.columns:
                if col in balance:
                    print "FEATURE:", col
                    uniq_train = Xtrain[col].unique()
                    uniq_test = Xtest[col].unique()
                    uniq_features = compareList(uniq_train, uniq_test, verbose=False)
                    if uniq_features.shape[0] > 0:
                        # print Xtrain[col].value_counts()
                        print "Removing unique categorical values in Train/Test", uniq_features
                        replace_dic = {}
                        for feat in uniq_features:
                            replace_dic[feat] = 'AMBIGIOUS'
                        Xall[col].replace(replace_dic, inplace=True)
                        # print Xall[col].value_counts()
                        # print Xtest[col].value_counts()

        if bagofwords_v2_0 is not None:
            print "Bag of wordsv2.0..."
            # balance components
            Xtrain = Xall[len(Xtest.index):]
            Xtest = Xall[:len(Xtest.index)]
            uniq_train = np.unique(Xtrain[comp_cols].values)
            uniq_test = np.unique(Xtest[comp_cols].values)
            uniq_features = compareList(uniq_train, uniq_test, verbose=False)
            if uniq_features.shape[0] > 0:
                # print Xtrain[col].value_counts()
                print "Removing unique categorical values in Train/Test", uniq_features
                replace_dic = {}
                for feat in uniq_features:
                    replace_dic[feat] = 'AMBIGIOUS'
                Xall.replace(replace_dic, inplace=True)

            my_array = np.empty(Xall.shape[0], dtype="S100")
            for row in range(Xall.shape[0]):
                value = ""
                for i in range(1, 8):
                    comp = Xall['component_id_' + str(i)].iloc[row]
                    quant = int(Xall['quantity_' + str(i)].iloc[row])
                    # print "I:%d C:%s Q:%d"%(i,comp,quant)
                    for j in range(0, quant):
                        value = value + " " + comp
                        # print value
                # for i in range(1,8):
                #    spec = Xall['spec'+str(i)].iloc[row]
                #    value = value +" "+spec
                if value == "":
                    value = "None"
                my_array[row] = value
            Xall['doc_component_id'] = pd.Series(my_array)
            print Xall
            Xall['doc_component_id'] = Xall.doc_component_id.str.replace("-", "_")
            vectorizer = CountVectorizer(min_df=3, max_features=100, lowercase=True, analyzer="word",
                                         ngram_range=(1, 1), stop_words=None, strip_accents='unicode',
                                         token_pattern=r'\w{1,}')
            # vectorizer = TfidfVectorizer(min_df=1,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 1), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = None,token_pattern=r'\w{1,}',norm='l2')#default
            Xtmp = vectorizer.fit_transform(Xall['doc_component_id']).todense()
            column_names = vectorizer.get_feature_names()
            Xtmp = pd.DataFrame(Xtmp, columns=column_names)
            keys = vectorizer.vocabulary_.keys()
            print "Length of dic:", len(keys)
            print "New features:", Xtmp.shape
            print "Columns:", column_names
            Xall = pd.concat([Xall, Xtmp], axis=1)
            print Xall[column_names].describe()

            # discarding columns
            for col in Xall.columns:
                if col.startswith('component_id') or col.startswith('quantity_') or col.startswith(
                        'doc_component_id') or col.startswith('spec'):
                    print "Dropping:", col
                    Xall.drop([col], axis=1, inplace=True)

        if concat or bagofwords is not None:

            Xall['doc_component_id'] = Xall.apply(lambda x: '%s %s %s %s %s %s %s %s' % (
                x['component_id_1'], x['component_id_2'], x['component_id_3'], x['component_id_4'], x['component_id_5'],
                x['component_id_6'], x['component_id_7'], x['component_id_8']), axis=1)
            Xall['doc_spec'] = Xall.apply(lambda x: '%s %s %s %s %s %s %s %s' % (
                x['spec1'], x['spec2'], x['spec3'], x['spec4'], x['spec5'], x['spec6'], x['spec7'], x['spec8']), axis=1)

            Xall['doc_component_id'] = Xall.doc_component_id.str.replace("-", "_")
            Xall['doc_component_id'] = Xall.doc_component_id.str.replace("AMBIGIOUS", "AMBIGIOUS_COMP")
            Xall['doc_spec'] = Xall.doc_spec.str.replace("-", "_")
            Xall['doc_spec'] = Xall.doc_spec.str.replace("AMBIGIOUS", "AMBIGIOUS_SPEC")

            print Xall['doc_component_id'].value_counts()
            print Xall['doc_spec'].value_counts()

            Xall['doc_component_id'].fillna('NA', inplace=True)
            Xall['doc_spec'].fillna('NA', inplace=True)

            if bagofwords is not None:
                print "Creating bag of components and of specs..."
                bagcolumns = bagofwords  #
                for col in bagcolumns:
                    vectorizer = CountVectorizer(min_df=2, max_features=100, lowercase=True, analyzer="word",
                                                 ngram_range=(1, 1), stop_words=None, strip_accents='unicode',
                                                 token_pattern=r'\w{1,}')
                    Xtmp = vectorizer.fit_transform(Xall[col]).todense()
                    Xtmp = pd.DataFrame(Xtmp, columns=vectorizer.get_feature_names())
                    keys = vectorizer.vocabulary_.keys()
                    print "Length of dic:", len(keys)
                    print "New features:", Xtmp.shape
                    # print Xtmp.describe()
                    for key in keys:
                        print "%-20s %10d" % (key, vectorizer.vocabulary_[key])

                    Xall = pd.concat([Xall, Xtmp], axis=1)
                    Xall.drop([col], axis=1, inplace=True)
                    print "New shape:", Xall.shape
                    print Xall.describe()

                for col in Xall.columns:
                    if col.startswith('component_id'):
                        print "Dropping:", col
                        Xall.drop([col], axis=1, inplace=True)

        b = ['supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a',
             'end_x', 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5',
             'component_id_6', 'component_id_7', 'component_id_8']
        b = b + ['doc_component_id', 'doc_spec']
        b = b + ['forming_endx', 'forming_enda']

        if useFrequencies:
            print "Substituting categories by frequencies"
            # preprocess
            for col in Xall.columns:
                if col in b or col.startswith('spec'):
                    df = Xall[col].value_counts()
                    df[df < 5] = 5
                    Xall = Xall.replace({col: df.to_dict()})



        # label encode the categorical variables
        for col in Xall.columns:
            lbl = preprocessing.LabelEncoder()
            if skipLabelEncoding is not None and col in skipLabelEncoding:
                print "SKIP Label_encoding:", col
                continue
            if encodeKeepNumber and (col in b or col.startswith('spec')):

                if not Xall[col].str.contains("-").any():
                    print "encoder now", tmp.isnull().sum(), " col:", col
                    Xall[col] = lbl.fit_transform(Xall[col].values)
                else:
                    print "extract number"
                    Xall[col] = Xall[col].astype(str)
                    print Xall[col].head(10)
                    tmp = Xall[col].str.split('-').str.get(1)
                    print "NaN:", tmp.isnull().sum()
                    tmp.fillna(0, inplace=True)

                    Xall[col] = tmp.astype(int)
                    print Xall[col].head(10)

                    # raw_input()
            elif col in b or col.startswith('spec'):
                print "Label_encoding:", col
                # print(Xall[col].iloc[::10000])
                Xall[col] = lbl.fit_transform(Xall[col].values)
                # print(Xall[col].iloc[::10000])


                # print("All df columns")
                # print(Xall.columns.to_series().groupby(Xall.dtypes).groups)
                # print(Xall.columns)
                # print Xall.shape
                # print Xall.describe(include='all')

                # drop some cols
                # Xall.drop([ 'component_id_7','quantity_7','component_id_8','quantity_8'], axis = 1,inplace=True)

    elif useRdata:
        print "Loading R generated dataframe"
        Xtrain = pd.read_csv('./data/train_R.csv', sep=';')
        Xtest = pd.read_csv('./data/test_R.csv', sep=';')

        y = Xtrain.cost.values
        idx = np.arange(Xtest.shape[0]) + 1  # .reshape((Xtest.shape[0],1))
        Xall = pd.concat([Xtest, Xtrain])

        # create some new features
        Xall['quote_date'] = pd.to_datetime(Xall['quote_date'])
        Xall['year'] = Xall.quote_date.dt.year
        Xall['month'] = Xall.quote_date.dt.month
        Xall.drop(
            ['cost', 'component_id_7', 'quantity_7', 'component_id_8', 'quantity_8', 'quote_date', 'spec9', 'spec10'],
            axis=1, inplace=True)

        # corrections
        Xall.fillna(0, inplace=True)

        b = ['supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a',
             'end_x', 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5',
             'component_id_6', 'component_id_7', 'component_id_8']
        # b=b+['spec']
        # label encode the categorical variables
        for col in Xall.columns:
            if col in skipLabelEncoding:
                continue

            if col in b or col.startswith('spec'):
                lbl = preprocessing.LabelEncoder()
                Xall[col] = lbl.fit_transform(Xall[col].values)
        print Xall.dtypes
        print Xall.describe(include='all')

    ta = Xall.tube_assembly_id.values
    ta_train = ta[len(Xtest.index):]
    ta_test = ta[:len(Xtest.index)]
    Xall.drop(['tube_assembly_id'], axis=1, inplace=True)

    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]

    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print "Shuffle train data..."
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print "unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0]))
        Xtrain = Xtrain.iloc[rows, :]
        y = y[rows]
        ta_train = ta_train[rows]


    return Xtest, Xtrain, y, idx, ta_train, ta_test


def prepareDataset(seed=123, nsamples=-1, addNoiseColumns=None, log1p=True, useRdata=False, createFeatures=False,
                   verbose=False, standardize=None, oneHotenc=None, concat=False, bagofwords=None, bagofwords_v2_0=None,
                   balance=None, removeComp=False, removeSpec=False, createVolumeFeats=False, useFrequencies=False,
                   dropFeatures=None, keepFeatures=None, createSparse=False, removeRare=None, logtransform=None,
                   createVerticalFeatures=False, createSupplierFeatures=None, owenEncoding=None, computeDiscount=False,
                   comptypes=0, NA_filler=0, removeLowVariance=False, skipLabelEncoding=None, createInflationData=False,
                   yBinning=0, outputSmearing=False, rootTransformation=False, useSampleWeights=False,
                   materialCost=False, shapeFeatures=False, timeFeatures=False, biningData=None,
                   createMaterialFeatures=None, useTubExtended=False, encodeKeepNumber=False, invertQuantity=False,
                   computeFixVarCost=False, removeRare_freq=None, createVerticalFeaturesV2=None, loadBN=False,
                   holdout=False):
    np.random.seed(seed)

    Xtest, Xtrain, y, idx, ta_train, ta_test = loadData(nsamples=nsamples, useRdata=useRdata, concat=concat,
                                                        bagofwords=bagofwords, bagofwords_v2_0=bagofwords_v2_0,
                                                        balance=balance, useFrequencies=useFrequencies,
                                                        logtransform=logtransform, comptypes=comptypes,
                                                        NA_filler=NA_filler, skipLabelEncoding=skipLabelEncoding,
                                                        useTubExtended=useTubExtended,
                                                        encodeKeepNumber=encodeKeepNumber, loadBN=loadBN)
    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    if addNoiseColumns is not None:
        print "Adding %d random noise columns" % (addNoiseColumns)
        Xrnd = pd.DataFrame(np.random.randn(Xall.shape[0], addNoiseColumns))
        for col in Xrnd.columns:
            Xrnd = Xrnd.rename(columns={col: 'rnd' + str(col + 1)})
        Xall = pd.concat([Xall, Xrnd], axis=1)

    sample_weight = None
    if useSampleWeights:
        # a=1.0/2.0
        print "Creating sample_weights..."

        # print Xtrain['annual_usage'].describe()
        sample_weight = np.sqrt(1.0 / Xtrain['quantity'].values)
        # sample_weight =Xtrain['quantity'].values
        print pd.Series(sample_weight).describe()
    # plt.hist(sample_weight,bins=50)
    # plt.show()

    if invertQuantity:
        Xall['quantity_inv'] = 1.0 / Xall['quantity']

    if createFeatures:
        print Xall.columns
        spec_cols = [col for col in Xall.columns if col.startswith('spec')]
        if len(spec_cols) > 0:
            Xspec = Xall[spec_cols]
            Xall['nspecs'] = (Xspec != 0).astype(int).sum(axis=1)
        # comp_cols = [col for col in Xall.columns if col.startswith('component_id')]
        # if len(comp_cols)>0:
        #    Xcomp = Xall[comp_cols]
        #    Xall['ncomponents'] = (Xcomp != 0).astype(int).sum(axis=1)
        qual_cols = [col for col in Xall.columns if col.startswith('quantity')]
        if len(qual_cols) > 0:
            Xqual = Xall[qual_cols]
            Xall['nparts'] = (Xqual != 0).astype(int).sum(axis=1)
            # raw_input()

            # print Xall[['nspecs','ncomponents','nparts']].describe()

    if biningData is not None:
        print "Binning..."
        columns = ['quantity', 'annual_usage']
        for col in columns:
            print Xall[col].describe()
            # print Xall[col].unique()
            bins = [-1, 0, 1, 2, 5, 10, 25, 50, 100, 250, 1000, 1E15]

            df = pd.cut(Xall[col], bins=bins)
            # create bins without last point
            bins = bins[:-1]
            # bins = np.diff(bins,n=1)
            # print bins
            # print df
            df.cat.categories = bins
            df = df.values.astype(np.float32)
            Xall[col] = df
            # print Xall[col].head(50)
            # print Xall[col].unique()
            print Xall[col].describe()

    if shapeFeatures:
        print "Shape features..."
        # is straight
        Xall['isStraight'] = (Xall['num_bends'] == 0).astype(int)
        print Xall['isStraight'].describe()
    # Xall['no_annual_usage'] = (Xall['annual_usage'] == 0).astype(int)
    # print Xall['no_annual_usage'].describe()
    # Xall['no_min_order_quantity'] = (Xall['min_order_quantity'] == 0).astype(int)
    # print Xall['no_min_order_quantity'].describe()

    # Xall['longer_than_wide'] = (Xall['length']>Xall['diameter']).astype(int)
    # print Xall['longer_than_wide'].describe()
    # Xall['length_diameter'] = Xall['length']/Xall['diameter']
    # print Xall['length_diameter'].describe()

    # timeFeatures
    # Xall['total_months'] = 2017 *12 - Xall['year']*12 + Xall['month']
    # print Xall['year'].describe()
    # print Xall['total_months'].describe()
    # Xall['isFuture'] = (Xall['year']>2015).astype(int)
    # print Xall['isFuture'].describe()
    # Xall.drop(['year'],axis=1,inplace=True)
    # raw_input()

    if timeFeatures:
        print "Time features"
        Xall['total_months'] = 2017 * 12 - Xall['year'] * 12 + Xall['month']
        Xtrain = Xall[len(Xtest.index):]

        target = 'year'
        # target = 'total_months'
        Xall[target] = Xall[target].astype(int)

        _tmp = pd.DataFrame(np.log1p(y), columns=['cost'], index=Xtrain.index)
        Xtrain_ta_y = pd.concat([Xtrain, pd.DataFrame(ta_train, columns=['ta'], index=Xtrain.index), _tmp], axis=1)

        grouped = Xtrain_ta_y.groupby(target)
        Xnew = grouped.aggregate(np.mean)

        cost_per_period = Xnew['cost']
        cost_per_period.set_value(1981, 0.0)
        cost_per_period.set_value(1983, cost_per_period.get_value(1982))
        cost_per_period.set_value(1984, cost_per_period.get_value(1983))
        cost_per_period.set_value(1985, cost_per_period.get_value(1984))
        cost_per_period.set_value(1986, cost_per_period.get_value(1985))
        cost_per_period.set_value(1990, cost_per_period.get_value(1989))
        cost_per_period.set_value(1991, cost_per_period.get_value(1990))
        cost_per_period = cost_per_period.sort_index()

        diff = pd.Series(np.diff(cost_per_period.values), index=cost_per_period.index[1:])

        diff = diff.to_dict()

        Xall['deltacost'] = Xall['year'].copy()
        Xall['deltacost'].replace(diff, inplace=True)

        Xall['cost_per_period'] = Xall['year'].copy()
        Xall['cost_per_period'].replace(cost_per_period, inplace=True)
    # Xall['annual_usage'] = Xall['annual_usage'].map(lambda x:max(1.0,x))
    # plt.scatter(cost_per_period.index,cost_per_period)
    # plt.show()

    if computeFixVarCost:
        print "Fix & Var cost model"
        # Xall['quantity'] = 1.0/Xall['quantity']
        Xtrain = Xall[len(Xtest.index):]
        _tmp = pd.DataFrame(np.log1p(y), columns=['cost'], index=Xtrain.index)
        Xtrain_ta_y = pd.concat([Xtrain, pd.DataFrame(ta_train, columns=['ta'], index=Xtrain.index), _tmp], axis=1)
        Xtrain_ta_y['fixcost'] = 0.0
        Xtrain_ta_y['varcost'] = 0.0
        Xtrain_ta_y['special'] = 0.0
        ta_uniques = np.unique(ta_train)
        for i, ta in enumerate(ta_uniques):
            idx = Xtrain_ta_y.ta == ta
            _df = Xtrain_ta_y.loc[idx]
            # _minq = _df['quantity'].min()
            # print _df.describe()
            if _df.shape[0] > 1:
                param, err = curve_fit(f_hyperbel, Xtrain_ta_y.loc[idx, 'quantity'], Xtrain_ta_y.loc[idx, 'cost'])

                print _df.head()
                print "ta:", ta
                print "param[0]", param[0]
                print "param[1]", param[1]
                print "err", err

                plt.scatter(_df['quantity'], _df['cost'], c='r', s=15)
                plt.scatter(_df['quantity'], f_hyperbel(_df['quantity'], param[0], param[1]), c='b')
                plt.xlabel('quantity')
                plt.ylabel('cost')
                plt.show()

                if param[0] < 105 and param[0] > 95:
                    print "not fittable...+100"
                    # print err
                    Xtrain_ta_y.loc[idx, 'varcost'] = -2.0
                    Xtrain_ta_y.loc[idx, 'fixcost'] = 1.7
                    Xtrain_ta_y.loc[idx, 'special'] = 1
                elif param[0] > -105 and param[0] < -95:
                    print "not fittable...-100"
                    # print err
                    Xtrain_ta_y.loc[idx, 'varcost'] = -2.0
                    Xtrain_ta_y.loc[idx, 'fixcost'] = 1.7
                    Xtrain_ta_y.loc[idx, 'special'] = 1
                elif np.all(np.isfinite(err)):

                    Xtrain_ta_y.loc[idx, 'varcost'] = param[0]
                    Xtrain_ta_y.loc[idx, 'fixcost'] = param[1]
                    Xtrain_ta_y.loc[idx, 'special'] = 0
                else:
                    print "inf..."
                    Xtrain_ta_y.loc[idx, 'varcost'] = -2.0
                    Xtrain_ta_y.loc[idx, 'fixcost'] = 1.7
                    Xtrain_ta_y.loc[idx, 'special'] = 1
            else:
                Xtrain_ta_y.loc[idx, 'varcost'] = -2.0
                Xtrain_ta_y.loc[idx, 'fixcost'] = 1.7
                Xtrain_ta_y.loc[idx, 'special'] = 1
            if i % 1000 == 0:
                print "Iteration i %d/%d:" % (i, len(ta_uniques))

        print "varcost", Xtrain_ta_y['varcost'].describe()
        print "fixcost", Xtrain_ta_y['fixcost'].describe()
        print "special", Xtrain_ta_y['special'].describe()

        y_tmpa = Xtrain_ta_y['varcost'].values
        y_tmpb = Xtrain_ta_y['fixcost'].values
        y_tmpc = Xtrain_ta_y['special'].values

        Xtrain_ta_y.drop(['varcost', 'fixcost', 'ta', 'cost', 'special'], axis=1, inplace=True)
        tmp_model = RandomForestRegressor(n_estimators=200, n_jobs=4)

        print dir(tmp_model)
        # a_train = tmp_model.predict(Xtrain_ta_y)# tmp_model.oob_prediction_
        a_train = getOOBCVPredictions(tmp_model, Xtrain_ta_y, y_tmpa,
                                      cv=KLabelFolds(pd.Series(ta_train), n_folds=8, repeats=1), returnSD=False,
                                      score_func='rmse')
        tmp_model.fit(Xtrain_ta_y, y_tmpa)

        plt.scatter(a_train, y_tmpa)
        plt.xlabel('predicted')
        plt.ylabel('orig')
        # plt.xlim(-10,10)
        # plt.ylim(-10,10)
        plt.show()
        a_test = tmp_model.predict(Xall[:len(Xtest.index)].values)

        b_train = getOOBCVPredictions(tmp_model, Xtrain_ta_y, y_tmpb,
                                      cv=KLabelFolds(pd.Series(ta_train), n_folds=8, repeats=1), returnSD=False,
                                      score_func='rmse')
        tmp_model.fit(Xtrain_ta_y, y_tmpb)
        # b_train = tmp_model.predict(Xall[len(Xtest.index):].values)# tmp_model.oob_prediction_

        plt.scatter(b_train, y_tmpb)
        plt.xlabel('predicted')
        plt.ylabel('orig')
        plt.show()

        b_test = tmp_model.predict(Xall[:len(Xtest.index)].values)

        print a_test.shape
        print a_train.shape
        Xall['varcost'] = np.hstack((a_test, a_train))
        Xall['fixcost'] = np.hstack((b_test, b_train))
    # Xall['quantity'] = 1.0/Xall['quantity']

    if computeDiscount:
        # plot versus supplier or material id
        Xtrain = Xall[len(Xtest.index):]
        fitting = True
        testing = False
        print "Compute log discount..."
        if not isinstance(computeDiscount, str):
            _tmp = pd.DataFrame(np.log1p(y), columns=['cost'], index=Xtrain.index)
            Xtrain_ta_y = pd.concat([Xtrain, pd.DataFrame(ta_train, columns=['ta'], index=Xtrain.index), _tmp], axis=1)

            Xtrain_ta_y['discount'] = 0.0
            Xtrain_ta_y['discount_fita'] = 0.0
            Xtrain_ta_y['discount_fitb'] = 0.0
            ta_uniques = np.unique(ta_train)
            # pretty inefficient...
            if not testing:
                for i, ta in enumerate(ta_uniques):
                    idx = Xtrain_ta_y.ta == ta
                    _df = Xtrain_ta_y.loc[idx]
                    # print _df
                    _minq = _df['quantity'].min()  # could be larger than 1
                    # print "minq:",_minq
                    ref_cost = _df.loc[_df['quantity'] == _minq, ['cost']].values
                    # print "refcost",ref_cost
                    ref_cost = float(ref_cost.max())  # could be more than one value
                    Xtrain_ta_y.loc[idx, 'discount'] = -1 * (Xtrain_ta_y.loc[idx, 'cost'] - ref_cost)
                    # fitting hyperbel to discount
                    if _df.shape[0] > 1 and fitting:
                        param, err = curve_fit(f_hyperbel, Xtrain_ta_y.loc[idx, 'quantity'],
                                               Xtrain_ta_y.loc[idx, 'discount'])
                        # plt.scatter(Xtrain_ta_y.loc[idx,'quantity'],Xtrain_ta_y.loc[idx,'discount'],c='r')
                        # plt.scatter(Xtrain_ta_y.loc[idx,'quantity'],f_hyperbel(Xtrain_ta_y.loc[idx,'quantity'],param[0],param[1]),c='b')
                        # plt.xlabel('quantity')
                        # plt.ylabel('discount')
                        # plt.show()
                        Xtrain_ta_y.loc[idx, 'discount_fita'] = param[0]
                        Xtrain_ta_y.loc[idx, 'discount_fitb'] = param[1]
                    else:
                        Xtrain_ta_y.loc[idx, 'discount_fita'] = 0
                        Xtrain_ta_y.loc[idx, 'discount_fitb'] = 0

                    if i % 500 == 0:
                        print "Iteration i %d/%d:" % (i, len(ta_uniques))
            else:
                pass

            # fitting the discount

            print "param a:", Xtrain_ta_y['discount_fita'].describe()
            print "param b:", Xtrain_ta_y['discount_fitb'].describe()
            y_tmpa = Xtrain_ta_y['discount_fita'].values
            y_tmpb = Xtrain_ta_y['discount_fitb'].values
            Xtrain_ta_y.drop(['discount', 'discount_fita', 'discount_fitb', 'ta', 'cost'], axis=1, inplace=True)

            # learn discount!!!
            # tmp_model = LinearRegression()
            tmp_model = RandomForestRegressor()
            a_train = getOOBCVPredictions(tmp_model, Xtrain_ta_y, y_tmpa,
                                          cv=KLabelFolds(pd.Series(ta_train), n_folds=8, repeats=1), returnSD=False,
                                          score_func='rmse')
            # a_train = tmp_model.predict(Xall[len(Xtest.index):].values)# tmp_model.oob_prediction_
            tmp_model.fit(Xtrain_ta_y, y_tmpa)
            a_test = tmp_model.predict(Xall[:len(Xtest.index)].values)

            b_train = getOOBCVPredictions(tmp_model, Xtrain_ta_y, y_tmpb,
                                          cv=KLabelFolds(pd.Series(ta_train), n_folds=8, repeats=1), returnSD=False,
                                          score_func='rmse')
            tmp_model.fit(Xtrain_ta_y, y_tmpb)
            # b_train = tmp_model.predict(Xall[len(Xtest.index):].values)# tmp_model.oob_prediction_
            b_test = tmp_model.predict(Xall[:len(Xtest.index)].values)

            print a_test.shape
            print a_train.shape
            Xall['discount_fita'] = np.hstack((a_test, a_train))
            Xall['discount_fitb'] = np.hstack((b_test, b_train))

            # plt.scatter(Xtrain_ta_y['quantity'],Xtrain_ta_y['discount'])
            # plt.xlabel('quantity')
            # plt.ylabel('discount')
            # plt.show()
            pd.DataFrame(y, columns=['discount']).to_csv('./data/discount.csv', index=False)
        # pd.DataFrame(ta_train,columns=['tube_assembly_id']).to_csv('./data/ta.csv',index=False)
        else:
            y = pd.read_csv('./data/discount.csv').values
            print y

            # plt.hist(y,bins=50)
            # plt.show()

    if createInflationData:
        # works but against the rules
        # http://inflationdata.com/Inflation/Inflation_Rate/HistoricalInflation.aspx
        print "Inflation data"
        print Xall['year']
        ir = {}
        ir[2017] = 0.016
        ir[2016] = 0.016
        ir[2015] = 0.016
        ir[2014] = 0.016
        ir[2013] = 0.015
        ir[2012] = 0.021
        ir[2011] = 0.032
        ir[2010] = 0.016
        ir[2009] = -0.003
        ir[2008] = 0.039
        ir[2007] = 0.029
        ir[2006] = 0.032
        ir[2005] = 0.034
        ir[2004] = 0.027
        ir[2003] = 0.023
        ir[2002] = 0.016
        ir[2001] = 0.028
        ir[2000] = 0.034
        ir[1999] = 0.022
        ir[1998] = 0.016
        ir[1997] = 0.023
        ir[1996] = 0.029
        ir[1995] = 0.028
        ir[1994] = 0.026
        ir[1993] = 0.030
        ir[1992] = 0.030
        ir[1991] = 0.043
        ir[1990] = 0.054
        ir[1989] = 0.048
        ir[1988] = 0.041
        ir[1987] = 0.037
        ir[1986] = 0.019
        ir[1985] = 0.036
        ir[1984] = 0.043
        ir[1983] = 0.032
        ir[1982] = 0.062
        ir[1981] = 0.104
        ir[1980] = 0.136
        Xall['ir'] = Xall['year'].copy()
        Xall['ir'].replace(ir, inplace=True)
        print Xall['ir']

        # http://www.boerse.de/historische-kurse/Euro-Dollar/EU0009652759
        eur = {}
        eur[2017] = 1.1158
        eur[2016] = 1.1158
        eur[2015] = 1.1158
        eur[2014] = 1.2098
        eur[2013] = 1.3743
        eur[2012] = 1.3187
        eur[2011] = 1.2945
        eur[2010] = 1.3391
        eur[2009] = 1.4325
        eur[2008] = 1.3978
        eur[2007] = 1.4599
        eur[2006] = 1.32
        eur[2005] = 1.1839
        eur[2004] = 1.3567
        eur[2003] = 1.2586
        eur[2002] = 1.0488
        eur[2001] = 0.8907
        eur[2000] = 0.9393
        eur[1999] = 1.0064
        eur[1998] = 1.1736
        eur[1997] = 1.0872
        eur[1996] = 1.2709
        eur[1995] = 1.3639
        eur[1994] = 1.2622
        eur[1993] = 1.1243
        eur[1992] = 1.2553
        eur[1991] = 1.4201
        eur[1990] = 1.4507
        eur[1989] = 1.2793
        eur[1988] = 1.0986
        eur[1987] = 1.3754
        eur[1986] = 1.0187
        eur[1985] = 0.7983
        eur[1984] = 0.6189
        eur[1983] = 0.7164
        eur[1982] = 0.8218
        eur[1981] = 0.8731
        eur[1980] = 0.9928
        Xall['eur'] = Xall['year'].copy()
        Xall['eur'].replace(eur, inplace=True)
        print Xall['eur']
    # GDP?
    # http://www.tradingeconomics.com/united-states/gdp

    if createVerticalFeatures:  # possibly flawed..?
        print "Creating vertical features..."
        ta = np.concatenate((ta_test, ta_train))
        ta_unique = np.unique(ta)
        Xall['max_quantity'] = 0
        Xall['mean_quantity'] = 0
        Xall['n_positions'] = 0
        for i, uv in enumerate(ta_unique):
            bool_idx = ta == uv
            # print Xall[bool_idx]
            Xall.loc[bool_idx, ['max_quantity']] = Xall.loc[bool_idx, ['quantity']].max(axis=0).values
            Xall.loc[bool_idx, ['mean_quantity']] = Xall.loc[bool_idx, ['quantity']].mean(axis=0).values
            Xall.loc[bool_idx, ['n_positions']] = Xall.loc[bool_idx, ['quantity']].shape[0]
            # print Xall[bool_idx]
            # raw_input()
            if i % 500 == 0:
                print "iteration %d/%d" % (i, len(ta_unique))

    if createVerticalFeaturesV2:
        print "Creating vertical features V2.0..."
        ta = np.concatenate((ta_test, ta_train))
        ta_unique = np.unique(ta)
        Xall_ta = pd.concat([Xall, pd.DataFrame(ta, columns=['ta'], index=Xall.index)], axis=1)
        Xall['max_annual_usage'] = 0
        Xall['median_annual_usage'] = 0
        Xall['min_annual_usage'] = 0
        Xall['min_quantity'] = 0
        Xall['min_min_order_quantity'] = 0
        Xall['max_min_order_quantity'] = 0

        for i, uv in enumerate(ta_unique):
            bool_idx = ta == uv
            # print Xall_ta.loc[bool_idx,:]
            Xall.loc[bool_idx, ['max_annual_usage']] = Xall.loc[bool_idx, ['annual_usage']].max(axis=0).values
            Xall.loc[bool_idx, ['min_annual_usage']] = Xall.loc[bool_idx, ['annual_usage']].min(axis=0).values
            Xall.loc[bool_idx, ['min_quantity']] = Xall.loc[bool_idx, ['quantity']].min(axis=0).values
            # Xall.loc[bool_idx,['median_annual_usage']] = Xall.loc[bool_idx,['annual_usage']].median(axis=0).values
            Xall.loc[bool_idx, ['min_min_order_quantity']] = Xall.loc[bool_idx, ['min_order_quantity']].max(
                axis=0).values
            Xall.loc[bool_idx, ['max_min_order_quantity']] = Xall.loc[bool_idx, ['min_order_quantity']].min(
                axis=0).values

            # distance
            # _df = Xall.loc[bool_idx]
            # _minq = _df['quantity'].min() # could be larger than 1
            # print "minq:",_minq
            # ref_idx = _df['quantity']==_minq
            # print "ref_idx",ref_idx
            # for i in range(_df.shape[1]-1):
            # print "i:",i
            ##print "df:",_df.iloc[i]
            # a = _df.iloc[i].values
            ##print a
            # b = _df.loc[ref_idx].values.flatten()
            ##print b
            # dist = euclidean(a,b)
            # print dist


            # print Xall.loc[bool_idx,['max_annual_usage','min_annual_usage']]
            if i % 500 == 0:
                print "iteration %d/%d" % (i, len(ta_unique))
        Xall['diff_annual_usage'] = Xall['annual_usage'] - Xall['min_annual_usage']
        Xall['diff_order_quantity'] = Xall['min_order_quantity'] - Xall['min_min_order_quantity']

    if createSupplierFeatures is not None:
        print "Creating supplier features..."
        print createSupplierFeatures
        grouped = Xall.loc[:, createSupplierFeatures].groupby('supplier', as_index=False)
        print grouped.describe()
        Xnew = grouped.aggregate(np.sum)
        Xnew.columns = [x + '_sum_supp' for x in Xnew.columns]
        Xnew.rename(columns={'supplier_sum_supp': 'supplier'}, inplace=True)
        print "Xnew", Xnew.describe()
        Xall = pd.merge(Xall, Xnew, on='supplier', how='left')

        Xnew = grouped['quantity'].aggregate(np.size)
        Xnew.columns = [x + '_size_supp' for x in Xnew.columns]
        Xnew.rename(columns={'supplier_size_supp': 'supplier'}, inplace=True)
        print "Xnew", Xnew.describe()
        Xall = pd.merge(Xall, Xnew, on='supplier', how='left')

        Xnew = grouped.aggregate(np.mean)
        Xnew.columns = [x + '_mean_supp' for x in Xnew.columns]
        Xnew.rename(columns={'supplier_mean_supp': 'supplier'}, inplace=True)
        print "Xnew", Xnew.describe()
        Xall = pd.merge(Xall, Xnew, on='supplier', how='left')

    # print grouped['annual_usage'].describe()

    if createMaterialFeatures is not None:
        print "Creating material_id features..."
        grouped = Xall.loc[:, createMaterialFeatures].groupby('material_id', as_index=False)
        print grouped.describe()
        Xnew = grouped.aggregate(np.sum)
        Xnew.columns = [x + '_sum_mat' for x in Xnew.columns]
        Xnew.rename(columns={'material_id_sum_mat': 'material_id'}, inplace=True)
        print "Xnew", Xnew.describe()
        Xall = pd.merge(Xall, Xnew, on='material_id', how='left')

        Xnew = grouped['quantity'].aggregate(np.size)
        Xnew.columns = [x + '_size_mat' for x in Xnew.columns]
        Xnew.rename(columns={'material_id_size_mat': 'material_id'}, inplace=True)
        print "Xnew", Xnew.describe()
        Xall = pd.merge(Xall, Xnew, on='material_id', how='left')

        Xnew = grouped.aggregate(np.mean)
        Xnew.columns = [x + '_mean_mat' for x in Xnew.columns]
        Xnew.rename(columns={'material_id_mean_mat': 'material_id'}, inplace=True)
        print "Xnew", Xnew.describe()
        Xall = pd.merge(Xall, Xnew, on='material_id', how='left')

        print "Xall", Xall.describe()

    if removeComp:
        for col in Xall.columns:
            if col.startswith('component'):
                print "Dropping:", col
                Xall.drop([col], axis=1, inplace=True)

    if createVolumeFeats:
        # Xall.drop(spec_cols,axis=1,inplace=True)
        # Xall['surface'] = math.pi*Xall['diameter'] * Xall['length']
        Xall['volume_tub'] = 0.5 * math.pi * Xall['diameter'] * Xall['diameter'] * Xall['length'] + 1.0
        Xall['volume_tub'] = Xall['volume_tub'].map(np.log)
        # Xall['volume_mat'] =  math.pi*Xall['diameter']*Xall['wall'] * Xall['length']+1.0
        # Xall['volume_mat'] = Xall['volume_mat'].map(np.log)

        Xall['crossection'] = 0.5 * math.pi * Xall['diameter'] * Xall['diameter'] + 1.0
        Xall['crossection'] = Xall['crossection'].map(np.log)

        # Xall['wallXlength'] = Xall['wall']* Xall['length']
        # Xall['diameterXlength'] = Xall['diameter']* Xall['length']
        Xall.drop(['diameter'], axis=1, inplace=True)
    # Xall.drop(['diameter'],axis=1,inplace=True)

    if removeSpec:
        for col in Xall.columns:
            if col.startswith('spec'):
                print "Dropping:", col
                Xall.drop([col], axis=1, inplace=True)

    if removeRare is not None:
        for col in oneHotenc:
            ser = Xall[col]
            print ser.value_counts()
            counts = ser.value_counts().keys()

            print "%s has %d different values before" % (col, len(counts))
            threshold = removeRare
            if len(counts) > threshold:
                ser[~ser.isin(counts[:threshold])] = 9999
            if len(counts) <= 1:
                print("Dropping Column %s with %d values" % (col, len(counts)))
                Xall = Xall.drop(col, axis=1)
            else:
                Xall[col] = ser.astype('category')
            print ser.value_counts()
            counts = ser.value_counts().keys()
            print "%s has %d different values after" % (col, len(counts))

    if removeRare_freq is not None:
        print "Remove rare features based on frequency..."
        for col in oneHotenc:
            ser = Xall[col]
            counts = ser.value_counts().keys()
            idx = ser.value_counts() > removeRare_freq
            threshold = idx.astype(int).sum()
            print "%s has %d different values before, min freq: %d - threshold %d" % (
                col, len(counts), removeRare_freq, threshold)
            if len(counts) > threshold:
                ser[~ser.isin(counts[:threshold])] = 9999
            if len(counts) <= 1:
                print("Dropping Column %s with %d values" % (col, len(counts)))
                Xall = Xall.drop(col, axis=1)
            else:
                Xall[col] = ser.astype('category')
            print ser.value_counts()
            counts = ser.value_counts().keys()
            print "%s has %d different values after" % (col, len(counts))

    if oneHotenc is not None:
        # Xtrain = Xall[len(Xtest.index):].values
        # Xtest = Xall[:len(Xtest.index)].values
        print "1-0 Encoding categoricals...", oneHotenc
        for col in oneHotenc:
            print "Unique values for col:", col, " -", np.unique(Xall[col].values)
            encoder = OneHotEncoder()
            X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values).todense())
            X_onehot.columns = [col + "_" + str(column) for column in X_onehot.columns]
            print "One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape)
            # print X_onehot.describe()
            Xall.drop([col], axis=1, inplace=True)
            Xall = pd.concat([Xall, X_onehot], axis=1)
            print "One-hot-encoding final shape:", Xall.shape
            # raw_input()

    if materialCost:  # Overfitt!!
        print "Material cost..."
        vn_y = 'cost'
        vn = 'material_id'
        # reduce df with quantity ==1 minimum
        Xtrain = Xall[len(Xtest.index):]
        df_yt = pd.DataFrame(np.zeros((Xtrain.shape[0], 2)), columns=[vn, vn_y])
        df_yt[vn_y] = y
        df_yt[vn] = Xtrain[vn].values
        mean0 = df_yt[vn_y].mean() * np.ones(Xall.shape[0])
        # group by material_id
        grp1 = df_yt.groupby([vn])
        # sum cost for category
        sum1 = grp1[vn_y].aggregate(np.sum)
        # count for material_id
        cnt1 = grp1[vn_y].aggregate(np.size)

        # merge data
        _key_codes = Xall[vn]
        _sum = sum1.loc[_key_codes].values
        _cnt = cnt1.loc[_key_codes].values
        # get owen feature for material

        vn_sum = 'sum_' + vn
        vn_cnt = 'cnt_' + vn
        # Xall[vn_sum] = _sum
        # Xall[vn_cnt] = _cnt

        df_y = np.zeros(Xall.shape[0])
        df_y[len(Xtest.index):] = df_yt[vn_y].values
        df_y[:len(Xtest.index)] = 0

        _sum -= df_y
        _cnt[len(Xtest.index):] -= 1
        # average material cost
        Xall['material_cost'] = _sum / _cnt
        Xall['material_cost'] = Xall['material_cost'].map(lambda x: x + np.random.normal(loc=0.0, scale=.15))
        # divide by length/volume (assume density=1)
        Xall['volume_mat'] = math.pi * Xall['diameter'] * Xall['wall'] * Xall['length'] + 1.0
        Xall['material_vol'] = Xall['material_cost'] / Xall['volume_mat']
    # Xall.drop(['material_id','volume_mat'],axis=1,inplace=True)
    # Xall.drop(['volume_mat'],axis=1,inplace=True)
    # raw_input()

    if owenEncoding is not None:
        # better for classification...!
        # https://github.com/owenzhang/kaggle-avazu/blob/master/utils.py
        print "Owen Encoding categoricals...", owenEncoding
        vn_y = 'cost'
        for vn in owenEncoding:
            # filter_train = np.zeros(Xall.shape[0], dtype=bool)
            # filter_train[len(Xtest.index):]=True
            # cred_k = 50

            # prepare dataframe
            Xtrain = Xall[len(Xtest.index):]

            df_yt = pd.DataFrame(np.zeros((Xtrain.shape[0], 2)), columns=[vn, vn_y])
            df_yt[vn_y] = y
            df_yt[vn] = Xtrain[vn].values


            # compute mean from training data
            mean0 = df_yt[vn_y].mean() * np.ones(Xall.shape[0])
            print "overall mean0:", mean0
            # group by category ie suppliers
            grp1 = df_yt.groupby([vn])
            # print grp1.describe()

            # sum cost for category
            sum1 = grp1[vn_y].aggregate(np.sum)
            # count for suppliers
            cnt1 = grp1[vn_y].aggregate(np.size)



            # merge data
            _key_codes = Xall[vn]
            _sum = sum1.loc[_key_codes].values
            _cnt = cnt1.loc[_key_codes].values



            _cnt[np.isnan(_sum)] = 0
            _sum[np.isnan(_sum)] = 0

            vn_sum = 'sum_' + vn
            vn_cnt = 'cnt_' + vn
            # Xall[vn_sum] = _sum
            # Xall[vn_cnt] = _cnt

            df_y = np.zeros(Xall.shape[0])
            df_y[len(Xtest.index):] = df_yt[vn_y].values
            df_y[:len(Xtest.index)] = 0


            _sum -= df_y
            _cnt[len(Xtest.index):] -= 1


            # Xall['oe_'+vn] = (_sum + cred_k * mean0)/(_cnt + cred_k)
            # Leustagos:
            # alpha = 1 / (_cnt +1 )
            # print alpha
            # Xall['oe_'+vn] = (1.0 - alpha) * _sum / (_cnt +1) + alpha * mean0
            Xall['oe_' + vn] = _sum / (_cnt + 1)
            # adding noise to avoid overfitting
            Xall['oe_' + vn] = Xall['oe_' + vn].map(lambda x: x + np.random.normal(loc=0.0, scale=.12))

            # simple
            # Xall['exp2_'+vn] = (_sum + cred_k * mean0)/(_cnt + cred_k)
            # Xall['oe_'+vn] = _sum / (_cnt +1)



        Xall.drop(owenEncoding, axis=1, inplace=True)


    if removeLowVariance:
        print Xall
        print "remove low var..."
        Xall = removeLowVar(Xall, threshhold=1E-5)

    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if logtransform is not None:
        print "log Transform"
        for col in logtransform:
            if col in Xall.columns: Xall[col] = Xall[col].map(np.log1p)
            # print Xall[col].describe(include=all)

    if standardize is not None and standardize is not False:
        ncol = Xall.shape[1]

        if verbose:
            for i in xrange(0, ncol - 5, 5):
                tmp = Xall.iloc[:, i:i + 5]
                print tmp.describe(include='all')
                tmp.hist(bins=50)
                plt.show()

        if not isinstance(Xall, pd.DataFrame):
            print "X is not a DataFrame, converting from,", type(Xall)
            Xall = pd.DataFrame(Xall.todense())

        if isinstance(standardize, str) and 'all' in standardize:
            Xall = scaleData(lXs=Xall, lXs_test=None)
        else:
            print "Standardizing cols:", standardize
            standardize = [x for x in standardize if x in Xall.columns]
            Xall.loc[:, standardize] = scaleData(lXs=Xall.loc[:, standardize])

        if verbose:
            for i in xrange(0, ncol - 5, 5):
                tmp = Xall.iloc[:, i:i + 5]
                print tmp.describe(include='all')
                tmp.hist(bins=50)
                plt.show()
                # raw_input()

    if createSparse:
        print "Creating sparse matrix..."
        if isinstance(Xtrain, pd.DataFrame): Xall = Xall.values
        Xall = csr_matrix(Xall)
        print type(Xall)
        density(Xall)

    if isinstance(Xall, pd.DataFrame):
        for col in Xall.columns:
            print "Col: %20s Is null: %d" % (col, Xall[col].isnull().sum())

    Xtrain = Xall[len(Xtest.index):]

    Xtest = Xall[:len(Xtest.index)]

    if verbose:
        # Xall['curvature'] = 1.0/Xall['diameter']
        plt.scatter(Xtrain['volume_mat'], np.log(y))
        # plt.scatter(Xtrain['surface'].map(np.log),np.log(y))
        # plt.scatter(Xtrain['volume'].map(np.log),np.log(y))
        plt.show()

    if log1p:
        print "Transform y: y=log(y+1)"
        y = np.log1p(y)

    if rootTransformation:
        y = np.power(y + 1.0, 1 / 64)

    if outputSmearing:
        y = y.map(lambda x: x + np.random.normal(loc=0.0, scale=.05))

    if yBinning > 0:
        print "Binning target value:", yBinning
        ys = pd.Series(y)
        y, bins = pd.qcut(ys, yBinning, retbins=True)
        # y,bins = pd.cut(ys, yBinning,retbins=True)
        # create bins without last point
        bins = bins[:-1]
        # bins = np.diff(bins,n=1)
        print bins
        y.cat.categories = bins
        y = y.values.astype(np.float32)
    # plt.hist(y,bins=2*bins.shape[0])
    # plt.show()

    if isinstance(Xtrain, pd.DataFrame): print "#columns", list(Xtrain.columns)

    if holdout:
        print "Split holdout..."

        unique_labels = np.unique(ta_train)

        print type(unique_labels)

        unique_labels = shuffle(np.asarray(unique_labels,dtype='S8'),random_state=seed)
        split_pc = 0.7
        index = int(split_pc * len(unique_labels))

        unique_val = unique_labels[index:]

        print unique_val[:10]
        print unique_val[-10:]

        val_mask = np.in1d(ta_train,unique_val)
        train_mask = np.logical_not(val_mask)

        Xval = Xtrain[val_mask].copy()
        Xtrain = Xtrain[train_mask].copy()
        yval = y[val_mask].copy()
        y = y[train_mask].copy()
        ta_train = ta_train[train_mask].copy()

        print "Xtrain:",Xtrain.shape
        print "yval:",yval.shape
        print "ta_train:",ta_train.shape


    print "#Xtrain:", Xtrain.shape
    print "#Xtest:", Xtest.shape

    # print type(ytrain)
    print "#ytrain:", y.shape

    if holdout:
        print "#Xval:", Xval.shape
        return Xtest, Xtrain, y, idx, ta_train, sample_weight, Xval, yval

    else:
        print "#data preparation finished!\n\n"
        return Xtest, Xtrain, y, idx, ta_train, sample_weight


def makePredictions(model=None, Xtest=None, idx=None, filename='submission.csv', log1p=True):
    if model is not None:
        preds = model.predict(Xtest)
    else:
        preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0]) + 1
    print preds.shape
    print idx.shape
    preds = np.clip(preds, a_min=0.0, a_max=1E15)
    if log1p:
        print "Backtransform y -> exp(y)-1"
        preds = np.expm1(preds)
    submission = pd.DataFrame({"id": idx, "cost": preds})  # order?
    submission = submission[['id', 'cost']]
    submission.to_csv(filename, index=False)
    print "Saving submission: ", filename


def loadBNdata():
    Xtrain = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/caterpillar/data/training_Aug23_log.csv")
    Xtest = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/caterpillar/data/testing_Aug23_log.csv")
    Xtrain.drop(['cost'], axis=1, inplace=True)
    print list(Xtrain.columns)
    return Xtest, Xtrain


if __name__ == "__main__":
    """
    MAIN PART
    """
    # TODO feature engineering: volume material, number of specs, EUR<->USD avg of year
    # TODO more day feature see abhishek btb

    # count features component, specs etc. OK
    # compute discount per TA: use cost vs. quantiyi slope & intercept per tube as feature
    # http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.stats.boxcox.html
    # can we get material cost? e.g. correlate material id with cost/volumet?
    # http://stackoverflow.com/questions/14248706/how-can-i-plot-a-histogram-in-pandas-using-nominal-values
    # one  hot encode: rare feature collect + supplier
    # delete things which are in train but not in test set... OK

    # make a bag of words from comp1 to comp9 OK
    # use also quantity OK
    # use components.csv to replace CP-10001
    # number of total components use also quantitiy OK
    # https://www.kaggle.com/jkapila/caterpillar-tube-pricing/0-24-with-xgboost-in-r
    # check script with many features: https://www.kaggle.com/ademyttenaere/caterpillar-tube-pricing/0-2748-with-rf-and-log-transformation/code
    # use quantity and better balance for comps!!!!
    # num connections
    # total quantity n supplier n materials
    # pandas impute data unique in train or test set
    # try also get dummies and one hot encoding from amazon
    # we could try to impute lables unique for train and test???
    # https://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/5283/winning-solution-code-and-methodology
    # make linear regressor with bounds!!!
    # postprocess predictions
    # try transformation other
    # directy predict discount
    # NN: normalize 0:1 instead of standardize
    # bracket prizing number of positions and max number
    # number of different suppliers for single TA (competition)?
    # number of TA a supplier offers (supplier size)
    # create tuples from ???
    # can we aggregate component features?
    # http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.stats.boxcox.html
    # Owen method: http://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions
    # Owen: https://github.com/owenzhang/kaggle-avazu/blob/master/utils.py
    # Leustagos: https://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/5283/winning-solution-code-and-methodology
    # https://github.com/diefimov/amazon_employee_access_2013/blob/master/%5B01%5Dtrain.gbm.lme.R
    # http://inflationdata.com/Inflation/Inflation_Rate/HistoricalInflation.aspx
    # http://www.boerse.de/historische-kurse/Euro-Dollar/EU0009652759
    # http://www.caterpillar.com/en/investors/stock-information/stock-quote/historical-pricelookup.html
    # learn discount model!!!!->like Owen
    # use pandas.qcut for classification !!!!
    # Monte Carlo wrapper around nnet ensemlber that remove randomly high pairwise correlated features
    # make 2nd level stacker!!!
    # impute unbalanced values by knn?
    # showMisclass
    # Use nonuniform bins for classification via cut
    # go back to the 1st level as currenlty no systematic improvement can be made
    # use a hold out set for the ensemle level
    # discount as first level feature
    # make nets deeper
    # reduce number of models without overfitting!
    # use pydev ide
    # use material cost + discount
    # REMOVE variables!!!!!!!!1
    # encode and keep last unit
    # removeRare needs min_freq clone
    # model with weights according to bracket_pricing
    # grouping of materialid & supplier

    t0 = time()

    print "numpy:", np.__version__
    print "pandas:", pd.__version__
    print "scipy:", sp.__version__
    categoricals = ['supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x',
                    'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5',
                    'component_id_6', 'component_id_7', 'component_id_8', 'spec1', 'spec2', 'spec3', 'spec4', 'spec5',
                    'spec6', 'spec7', 'spec8']
    categoricals_nospec = ['supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x',
                           'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5',
                           'component_id_6', 'component_id_7', 'component_id_8']
    categoricals_small = ['supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']
    base_cols = ['supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a',
                 'end_x']
    comp_cols = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5',
                 'component_id_6', 'component_id_7', 'component_id_8']
    spec_cols = ['spec1', 'spec2', 'spec3', 'spec4', 'spec5', 'spec6', 'spec7', 'spec8']
    numerical_cols = ['annual_usage', 'min_order_quantity', 'quantity', 'diameter', 'wall', 'length', 'num_bends',
                      'bend_radius', 'num_boss', 'num_bracket', 'other', 'quantity_1', 'quantity_2', 'quantity_3',
                      'quantity_4', 'quantity_5', 'quantity_6', 'quantity_7', 'quantity_8', 'year', 'month']
    log_cols = ['annual_usage', 'min_order_quantity', 'quantity', 'diameter', 'wall', 'length', 'num_bends',
                'bend_radius']
    new_feats = ['nspecs', 'nparts', 'max_quantity', 'mean_quantity', 'n_positions']
    new_feats2 = ['deltacost', 'cost_per_period', 'quantity_sum_supp', 'annual_usage_sum_supp', 'diameter_sum_supp',
                  'quantity_size_supp', 'quantity_mean_supp', 'annual_usage_mean_supp', 'diameter_mean_supp',
                  'quantity_sum_mat', 'annual_usage_sum_mat', 'diameter_sum_mat', 'quantity_size_mat',
                  'quantity_mean_mat', 'annual_usage_mean_mat', 'diameter_mean_mat']

    time_cols = ['year']

    seed = 421
    nsamples = 'shuffle'
    holdout = False

    addNoiseColumns = None
    useRdata = False
    useSampleWeights = False
    useTubExtended = False
    loadBN = 'other'

    # output transformations
    log1p = True
    rootTransformation = False
    outputSmearing = False
    yBinning = -1
    computeDiscount = False  # 'load'#True#'load'#True
    computeFixVarCost = False

    # other transformation
    NA_filler = 0
    comptypes = -1  # comptypes=1-5->0.241 comptypes=6#0.240   comptype=7->0.239 comptype=8->0.241
    logtransform = None  # log_cols
    balance = None  # base_cols + comp_cols + spec_cols
    invertQuantity = False

    # feature creation
    createFeatures = False
    createVolumeFeats = False
    createVerticalFeatures = False  # Some overfit??
    createVerticalFeaturesV2 = False
    createSupplierFeatures = None  # ['supplier','quantity','annual_usage','diameter']
    createMaterialFeatures = None  # ['material_id','quantity','annual_usage','diameter']
    materialCost = False  # True->Overfit!!
    shapeFeatures = False
    timeFeatures = False
    createInflationData = False  # True->>not allowed

    verbose = False

    bagofwords = None  # ['component_id_doc','spec_doc']
    bagofwords_v2_0 = None

    # other
    encodeKeepNumber = False
    skipLabelEncoding = None  # ['year']#['supplier']
    removeRare = None  # 10#None#15#None#categoricals#['material_id','supplier']
    removeRare_freq = None  # 200
    oneHotenc = None  # ['supplier','material_id']  # None#['supplier']#categoricals_nospec#['supplier']#categoricals_nospec#['supplier','material_id']
    owenEncoding = None  # ['quantity']#categoricals_nospec#['material_id']#None#['supplier','materials_id']
    biningData = None  # 30

    useFrequencies = False
    concat = False

    # final transformatins
    standardize = None  # numerical_cols#True
    removeLowVariance = True
    removeSpec = False
    removeComp = False
    createSparse = False

    dropFeatures = None  # ['oe_quantity']#None#['supplier_2','supplier_45','supplier_40','supplier_33','supplier_23','supplier_21','supplier_8','supplier_7','supplier_3','supplier_1','quantity_8','supplier_11','supplier_0','supplier_27','supplier_28','supplier_34','supplier_6','component_id_8','material_id_1','supplier_44','component_id_7','quantity_7']#R feature importance
    keepFeatures = None  # round2#["quantity","length","annual_usage","num_bends","supplier_18","supplier_24","min_order_quantity","month","component_id_1","diameter","year","supplier_32","supplier_31","nspecs","component_id_2","bend_radius","quantity_2","quantity_1","supplier_35","material_id_10","supplier_26","component_id_3","wall","end_a","end_x","bracket_pricing","end_a_2x","quantity_3","supplier_10","supplier_9","material_id_5","material_id_13","material_id_2","num_boss","component_id_4","supplier_14","other","material_id_4","end_x_2x","end_x_1x","material_id_12","material_id_0","material_id_14","end_a_1x","quantity_4","supplier_15","material_id_3","num_bracket","supplier_39","supplier_16","material_id_15","supplier_43","supplier_30","material_id_16","supplier_20","material_id_8","supplier_22","supplier_29","supplier_5","component_id_6","quantity_5","quantity_6","supplier_12","supplier_42","supplier_25","material_id_6","component_id_5","supplier_37","supplier_19","material_id_17"]#rf_select
    # load=''

    Xtest, Xtrain, ytrain, idx, ta_train, sample_weight = prepareDataset(seed=seed, nsamples=nsamples,
                                                                         addNoiseColumns=addNoiseColumns, log1p=log1p,
                                                                         useRdata=useRdata,
                                                                         createFeatures=createFeatures, verbose=verbose,
                                                                         standardize=standardize, oneHotenc=oneHotenc,
                                                                         concat=concat, bagofwords=bagofwords,
                                                                         balance=balance, removeComp=removeComp,
                                                                         removeSpec=removeSpec,
                                                                         createVolumeFeats=createVolumeFeats,
                                                                         useFrequencies=useFrequencies,
                                                                         dropFeatures=dropFeatures,
                                                                         keepFeatures=keepFeatures,
                                                                         bagofwords_v2_0=bagofwords_v2_0,
                                                                         createSparse=createSparse,
                                                                         removeRare=removeRare,
                                                                         logtransform=logtransform,
                                                                         computeDiscount=computeDiscount,
                                                                         createVerticalFeatures=createVerticalFeatures,
                                                                         createSupplierFeatures=createSupplierFeatures,
                                                                         owenEncoding=owenEncoding, comptypes=comptypes,
                                                                         NA_filler=NA_filler,
                                                                         removeLowVariance=removeLowVariance,
                                                                         skipLabelEncoding=skipLabelEncoding,
                                                                         createInflationData=createInflationData,
                                                                         yBinning=yBinning,
                                                                         outputSmearing=outputSmearing,
                                                                         rootTransformation=rootTransformation,
                                                                         useSampleWeights=useSampleWeights,
                                                                         materialCost=materialCost,
                                                                         shapeFeatures=shapeFeatures,
                                                                         timeFeatures=timeFeatures,
                                                                         biningData=biningData,
                                                                         createMaterialFeatures=createMaterialFeatures,
                                                                         useTubExtended=useTubExtended,
                                                                         encodeKeepNumber=encodeKeepNumber,
                                                                         invertQuantity=invertQuantity,
                                                                         computeFixVarCost=computeFixVarCost,
                                                                         removeRare_freq=removeRare_freq,
                                                                         createVerticalFeaturesV2=createVerticalFeaturesV2,
                                                                         loadBN=loadBN, holdout=holdout)
    if isinstance(Xtrain, pd.DataFrame):
        Xtrain.to_csv('./data/Xtrain.csv', index=False)
        Xtest.to_csv('./data/Xtest.csv', index=False)
        pd.DataFrame(ytrain, columns=['cost']).to_csv('./data/ytrain.csv', index=False)
        pd.DataFrame(ta_train, columns=['tube_assembly_id']).to_csv('./data/ta.csv', index=False)

    # interact_analysis(Xtrain)
    # showCorrelations(Xtrain,steps=3)
    # Xtrain.iloc[:,:30].hist(bins=50)
    # plt.show()
    #Xtrain, Xtest = removeCorrelations(Xtrain, Xtest, 0.995)
    ##MODELS##
    # model = KNeighborsRegressor(n_neighbors=5)
    # model = SVR()
    # model = LassoLarsCV()
    # model = SGDRegressor()
    # model = LinearRegression()True
    # model = SVR(C=100.0, gamma=0.0, verbose = 0)
    # model = PLSRegression(n_components=50)
    # model = XgboostRegressor(booster='gblinear',n_estimators=100,alpha_L1=1.0,lambda_L2=1.0,n_jobs=2,objective='reg:linear',eval_metric='rmse',silent=1)#0.63
    # model = XgboostRegressor(n_estimators=2000,learning_rate=0.05,max_depth=7,subsample=.8,colsample_bytree=0.8,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # model = XgboostRegressor(max_depth=7,subsample=.85,colsample_bytree=0.75,min_child_weight=6,gamma=2,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gblinear',silent=1,eval_size=0.0)#some R model
    # model = XgboostRegressor(n_estimators=1200,learning_rate=0.025,max_depth=15,subsample=.5,n_jobs=2,colsample_bytree=1.0,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # xgbFeatureImportance(model,Xtrain,ytrain)
    # model = GradientBoostingRegressor(n_estimators=500,learning_rate=0.06,max_depth=7,subsample=.5)
    # DEFAULT
    # model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/3)
    # model = XgboostClassifier(NA=0,n_estimators=400,learning_rate=0.1,max_depth=15,subsample=.5,colsample_bytree=0.8,min_child_weight=5,n_jobs=4,objective='multi:softmax',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # model = XgboostRegressor(NA=0,n_estimators=400,learning_rate=0.05,max_depth=15,subsample=.5,colsample_bytree=0.8,min_child_weight=1,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # model = XgboostRegressor(n_estimators=400,learning_rate=0.0632,max_depth=13,subsample=.58,colsample_bytree=0.82,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # model = XgboostRegressor(n_estimators=4000,learning_rate=0.02,max_depth=7,subsample=.95,colsample_bytree=.6,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # model = BaggingRegressor(base_estimator=model,n_estimators=10,n_jobs=1,verbose=2,random_state=None,max_samples=0.96,max_features=.96,bootstrap=False)
    model = XgboostRegressor(n_estimators=1000,learning_rate=0.05,max_depth=8,subsample=.8,colsample_bytree=0.8,min_child_weight=1,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    # model = RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=True)
    #model = RandomForestRegressor(n_estimators=250, max_depth=None, min_samples_leaf=1, n_jobs=4,  max_features=Xtrain.shape[1] / 2)
    # model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/3)
    # model = GradientBoostingRegressor(loss='ls',n_estimators=100, learning_rate=0.05, max_depth=10,subsample=.5,verbose=0)
    # model = KerasNNReg(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=50,learning_rate=0.015,validation_split=0.0,batch_size=128,verbose=1)
    # xgbFeatureImportance(model,Xtrain,Xtest)
    Xtrain = scaleData(Xtrain,normalize=False).values
    # model = Pipeline([('scaler', StandardScaler()), ('model',nn10_BN)])
    #model = nnet_BN_deep  # nn10_BN
    print model
    # featureImportance(model,Xtrain,ytrain)

    if log1p:
        scoring_func = make_scorer(root_mean_squared_error, greater_is_better=False)
    else:
        scoring_func = make_scorer(root_mean_squared_log_error, greater_is_better=False)
    # XVALIDATION#
    # cv = KFold(ytrain.shape[0],5,shuffle=True)
    # cv = LeavePLabelOutWrapper(ta_train,n_folds=3,p=1)
    cv = KLabelFolds(pd.Series(ta_train), n_folds=8, repeats=1)
    # XVAL END

    if isinstance(model, NeuralNet) or isinstance(model, KerasNNReg):
        print "Converting df to np..."
        if isinstance(Xtrain, pd.DataFrame):
            Xtrain = Xtrain.values
        ytrain = ytrain.reshape((ytrain.shape[0], 1))

        # xgbFeatureImportance(model,Xtrain,ytrain)
    # model.fit(Xtrain,ytrain)
    # parameters = {'n_estimators':[400],'max_depth':[7],'learning_rate':[0.1,0.05,0.01],'subsample':[0.5],'colsample_bytree':[.8],'min_child_weight':[1]}
    # parameters = {'n_estimators':[2000],'max_depth':[8],'learning_rate':[0.01,0.02,0.03],'subsample':[0.75],'colsample_bytree':[0.75],'min_child_weight':[5]}
    # parameters = {'n_estimators':[8000],'max_depth':[8],'learning_rate':[0.008,0.01,0.02],'subsample':[0.7],'colsample_bytree':[0.7],'min_child_weight':[5]}
    # parameters = {'dropout0_p':[0.0],'hidden1_num_units': [256],'dropout1_p':[0.0,0.1,0.15],'hidden2_num_units': [256],'dropout2_p':[0.0,0.1,0.15],'max_epochs':[100,125,150]}
    parameters = {'dropout0_p': [0.0], 'hidden1_num_units': [600], 'dropout1_p': [0.0, 0.1,0.2], 'hidden2_num_units': [600,800],'dropout2_p': [0.0, 0.1,0.2], 'hidden3_num_units': [600,800], 'dropout3_p': [0.0, 0.1,0.2],'max_epochs': [100]}
    # parameters = {'n_estimators':[250,500],'max_features':[85,90,95],'min_samples_leaf':[1,2],'bootstrap':[False]}
    # parameters = {'n_neighbors':[4,5,6,7]}
    # parameters = {'C': [5000,1000,500]}
    # parameters = {'n_estimators':[15,20,25],'max_features':[0.95],'max_samples':[1.0]}
    #model = makeGridSearch(model, Xtrain, ytrain, n_jobs=1, refit=True, cv=cv, scoring=scoring_func,parameters=parameters, random_iter=30)

    #model = buildXvalModel(model,Xtrain,ytrain,sample_weight=sample_weight,refit=True,cv=cv,class_names=ta_train)
    model = buildModel(model,Xtrain,ytrain,cv=cv,scoring=scoring_func,n_jobs=8,trainFull=True,verbose=True)
    #iterativeFeatureSelection(model, Xtrain, Xtest, ytrain.ravel(), iterations=50, nrfeats=2, scoring=scoring_func, cv=cv, n_jobs=2)
    # print Xtest.describe()
    # print Xtest.shape
    # makePredictions(model=model, Xtest=Xtest, idx=idx, filename='submissions/sub16082015xxx.csv', log1p=log1p)
    print("Model building done in %fs" % (time() - t0))
