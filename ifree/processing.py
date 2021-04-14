import os, csv, six
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from minepy import MINE
from scipy import stats
from itertools import permutations, combinations
from sklearn import feature_selection
from sklearn import ensemble
from sklearn import linear_model

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from tqdm import tqdm
from radiomics import featureextractor
import SimpleITK as sitk

class FeatureExtractor():
    def __init__(self, idList, imagePaths, maskPaths, paramPath, outputPath):
        self.idList = idList
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.paramPath = paramPath
        self.outputPath = outputPath
        assert len(self.imagePaths) == len(self.maskPaths), "#imagePaths #maskPaths is not consistent!"

    def save2csv(self, features, idx):
        assert len(features) == len(idx), "#features #idx is not consistent!"

        if len(features) == 0:
            print("No features!")
        else:
            obs_header = ["id"]
            obs_header.extend(list(features[0].keys()))
            if os.path.exists(self.outputPath) or os.path.isfile(self.outputPath):
                os.remove(self.outputPath)
            f = open(self.outputPath, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
            w = csv.DictWriter(open(self.outputPath, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
            for i in range(len(features)):
                fea = features[i]
                fea["id"] = idx[i]
                w.writerows([fea])
            print("features saved to ", self.outputPath)

    def singleExtract(self, imageName, maskName, params):
        if imageName.split(".")[-1] in ["bmp", "jpg", "png"]:
            return self.singleExtract_BMP(imageName, maskName, params)
        else:
            return self.singleExtract_NII(imageName, maskName, params)

    def singleExtract_BMP(self, imageName, maskName, params):

        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params)


            color_channel = 0
            im = sitk.ReadImage(imageName)
            selector = sitk.VectorIndexSelectionCastImageFilter()
            selector.SetIndex(color_channel)
            im = selector.Execute(im)


            color_channel = 0
            mas = sitk.ReadImage(maskName)
            selector = sitk.VectorIndexSelectionCastImageFilter()
            selector.SetIndex(color_channel)
            mas = selector.Execute(mas)

            result = extractor.execute(im, mas)
            feature = {key: val for key, val in six.iteritems(result)}

            
        except Exception as e:
            print(e)
            print("error when extacting ", imageName)
            feature = None
        return feature
    def singleExtract_NII(self, imageName, maskName, params):

        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            result = extractor.execute(imageName, maskName)
            feature = {key: val for key, val in six.iteritems(result)}

        except Exception as e:
            print(e)
            print("error when extacting ", imageName)
            feature = None
        return feature

    def extract(self, force=False):
        if os.path.exists(self.outputPath) and not force:
            print("file already exist : ", self.outputPath)
        else:
            features = []
            idx = []
            lens = len(self.imagePaths)
            for i in tqdm(range(lens), total=lens):
                imageName = self.imagePaths[i]
                maskName = self.maskPaths[i]
                feature = self.singleExtract(imageName, maskName, self.paramPath)
                if feature is not None:
                    features.append(feature)
                    idx.append(self.idList[i])
            self.save2csv(features, idx)




class FeatureProcess():
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def simpleImpute(self, strategy='mean'):
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imp.fit(self.X_train)
        self.X_train = imp.transform(self.X_train)
        self.X_test = imp.transform(self.X_test)
        
    def standardScale(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def normalizer(self):
        nor = Normalizer()
        nor.fit(self.X_train)
        self.X_train = nor.transform(self.X_train)
        self.X_test = nor.transform(self.X_test)

    def minMaxScaler(self):
        mms = MinMaxScaler()
        mms.fit(self.X_train)
        self.X_train = mms.transform(self.X_train)
        self.X_test = mms.transform(self.X_test)
    
    def pca(self, n_components=10):
        pca = PCA(n_components=n_components)
        pca.fit(self.X_train)
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)




class bicluster:
    def __init__(self, vec, left=None,right=None,distance=0.0,id=None) :
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance

class FeatureSelector():
    def __init__(self, features, labels, featureNames, eps=1e-3):
        self.x = features
        self.y = labels
        self.featureNames = np.array(featureNames)
        self.eps = eps
        self.n = self.x.shape[1]
        self.indexs = None

    ############################
    #   univariate selection   #
    ############################
    def univarSelector(self, top_k=1, method_name="f_classif", inplace=True):
        """
        :method_name {"chi2", "f_classif", "mutual_info_classif"}
        """
        print("Feature selecting method: ", method_name)
        selector = {"chi2": feature_selection.chi2,
                    "f_classif": feature_selection.f_classif,
                    "mutual_info_classif": feature_selection.mutual_info_classif}
        func = selector[method_name]
        sler = feature_selection.SelectKBest(func, k=top_k)
        sler.fit(self.x, self.y)
        self.indexs = sler.get_support()

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    # regression selecting  #
    #########################
    def lrSelector(self, method_name="lr", inplace=True):
        """
        :method_name {"lr", "ridge"}
        """
        print("Feature selecting method: ", method_name)
        selector = {"lr": linear_model.LinearRegression(), "ridge": linear_model.Ridge()}
        lr = selector[method_name]
        lr.fit(self.x, self.y)
        coefs = lr.coef_.tolist()
        self.indexs = [i for i in range(len(coefs)) if np.abs(coefs[i]) > self.eps]
        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    ############################
    #      model selecting     #
    ############################
    def modelSelector(self, model_name="rf", inplace=True):
        """
        :method_name {"rf", "lasso"}
        """
        print("Feature selecting method: ", model_name)
        selector = {"rf": ensemble.RandomForestClassifier(n_estimators=10), "lasso": linear_model.LassoCV(cv=5, max_iter=5000)}
        model = selector[model_name]
        sler = feature_selection.SelectFromModel(model)
        sler.fit(self.x, self.y)
        self.indexs = sler.get_support()

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y


    #########################
    # correlation selecting #
    #########################
    def calMic(self, x1, x2):
        # Maximal Information Coefficient
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x1, x2)
        return mine.mic()

    def hcluster(self, X, calDistance) :
        biclusters = [ bicluster(vec = X[:, i], id = i ) for i in range(X.shape[1]) ]
        distances = {}
        flag = None
        currentclusted = -1
        print("features dim: ", len(biclusters))
        while(len(biclusters) > 1) :
            max_val = -1
            biclusters_len = len(biclusters)
            for i in range(biclusters_len-1) :
                for j in range(i + 1, biclusters_len) :
                    if distances.get((biclusters[i].id,biclusters[j].id)) == None:
                        distances[(biclusters[i].id,biclusters[j].id)], _ = calDistance(biclusters[i].vec,biclusters[j].vec)
                    d = distances[(biclusters[i].id,biclusters[j].id)] 
                    if d > max_val :
                        max_val = d
                        flag = (i,j)
            bic1,bic2 = flag
            newvec = (biclusters[bic1].vec + biclusters[bic2].vec) / 2
            newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=max_val, id = currentclusted)
            currentclusted -= 1
            del biclusters[bic2]
            del biclusters[bic1]
            biclusters.append(newbic)
        return biclusters[0]

    def corrSelector(self, method_name="pearson", num=1, inplace=True):
        """
        :method_name {"pearson", "kendall", "spearman", "mic"}
        """
        print("Feature selecting method: ", method_name)
        selector = {"pearson": stats.pearsonr, "kendall": stats.kendalltau,
                    "spearman": stats.spearmanr, "mic": self.calMic}
        func = selector[method_name]
        root_node = self.hcluster(self.x, func)
        node_ids = []
        nodes = [root_node]
        while(len(nodes)>0):
            tmp = nodes.pop(0)
            if tmp.id<0:
                nodes.append(tmp.left)
                nodes.append(tmp.right)
            else:
                node_ids.append(tmp.id)
        self.indexs = np.asarray(node_ids)[:num]

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    #    T test selecting   #
    #########################
    def calTtest(self, x1, x2, threshold=0.05):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        stat, p = stats.levene(x1, x2)
        if p > threshold:
            _, t = stats.ttest_ind(x1, x2)
        else:
            _, t = stats.ttest_ind(x1, x2, equal_var=False)
        return t

    def ttestSelector(self, threshold=0.05, inplace=True):
        print("Feature selecting method: T test")
        labels = np.unique(self.y)
        num_labels = len(labels)
        cbs = list(combinations(range(num_labels), 2))
        num_cbs = len(cbs)
        eps = 0.1
        self.indexs = []
        for i in range(self.n):
            count = 0
            for c in range(num_cbs):
                index1, index2 = cbs[c]
                row1, row2 = labels[index1], labels[index2]
                r1, r2 = self.y==row1, self.y==row2
                t = self.calTtest(self.x[r1, i], self.x[r2, i])
                if t > threshold:
                    count += 1
            if count/num_cbs < eps:
                self.indexs.append(i)

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    #   mann whitney u test #
    #########################
    def mannSelector(self, threshold=0.05, inplace=True):
        print("Feature selecting method: mann whitney u test")
        labels = np.unique(self.y)
        num_labels = len(labels)
        cbs = list(combinations(range(num_labels), 2))
        num_cbs = len(cbs)
        eps = 0.1
        self.indexs = []
        for i in range(self.n):
            count = 0
            for c in range(num_cbs):
                index1, index2 = cbs[c]
                row1, row2 = labels[index1], labels[index2]
                r1, r2 = self.y==row1, self.y==row2
                _, t = stats.mannwhitneyu(self.x[r1, i], self.x[r2, i], alternative='two-sided')
                if t > threshold:
                    count += 1
            if count/num_cbs < eps:
                self.indexs.append(i)

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    #       combination     #
    #########################
    def combination(self, train_test_run, x_test, y_test, inplace=True):
        print("Feature selecting method: combination")
        len_f = self.x.shape[1]
        res_indexs = []
        min_value = 2
        for c_f in range(1,len_f):
            cbs = list(combinations(list(range(len_f)), c_f))
            for c in cbs:
                indexs = [i for i in c]
                error_mse, error_mae = train_test_run(self.x[:, indexs], self.y, x_test[:, indexs], y_test)
                if error_mae<min_value:
                    min_value=error_mae
                    self.indexs = indexs
                    res_indexs.append({"error_mse":error_mse,"error_mae":error_mae,"indexs":indexs})
        print(res_indexs[-1])
        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            self.featureNames = self.featureNames[self.indexs]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y
