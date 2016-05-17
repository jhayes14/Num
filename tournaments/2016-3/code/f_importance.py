from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
#import f_select
import warnings
import dill
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore")


def get_best_features(n=21,f_files=18):
    train_top = np.zeros((57771, f_files*n))
    test_top  = np.zeros((13349, f_files*n))
    new_Xall_df = pd.DataFrame()
    c=0
    for i in range(0, 19):
        if i == 8:
            pass
        else:
            print
            print "F number: ", i
            print

            feature_file = 'features__' + str(i) + '.pickle'
            if os.path.exists(feature_file):
                with open(feature_file, "rb") as input_file:
                    D = dill.load(input_file)
            else:
                "Feature file missing"

            X               = D[0]
            training_target = D[1]
            Y               = D[2]
            test_id         = D[3]
            Xall_np         = D[4]
            Xall_df         = D[5]
           
            check_array = np.random.rand(2,2)
            if type(check_array) == type(Xall_df):
                Xall_df = pd.DataFrame(data=Xall_df, columns=["golden_f_"+str(j) for j in range(len(Xall_df[0]))])

            print("Training:")
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
            clf.fit(X, training_target)
            #std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
            #indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            print("Feature ranking:")
            print "Features sorted by their score:"
            sorted_features = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), Xall_df.columns), 
                                 reverse=True)
            print sorted_features
            with open("sorted_features__"+str(i)+".pickle", "wb") as output_file:
                dill.dump(sorted_features, output_file)


            # choose top n features
            f_ranking = clf.feature_importances_
            top_n = f_ranking.argsort()[-n:][::-1]
            train_top[:, range(c,n+c)] = X[:, top_n] 
            test_top[:, range(c, n+c)] = Y[:, top_n]
            new_Xall_df = pd.concat([new_Xall_df,Xall_df.iloc[:,top_n]])
        with open("features__best_"+str(n*f_files)+".pickle", "wb") as output_file:
            dill.dump((train_top, training_target, test_top, test_id, new_Xall_df), output_file)

get_best_features()

#for f in range(X.shape[1]):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
#        plt.figure()
#        plt.title("Feature importances")
#        plt.bar(range(X.shape[1]), importances[indices],
#                       color="r", yerr=std[indices], align="center")
#        plt.xticks(range(X.shape[1]), indices)
#        plt.xlim([-1, X.shape[1]])
#        plt.show()
