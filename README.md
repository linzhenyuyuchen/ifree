# ifree 

[![Python package](https://img.shields.io/badge/Python-ifree-orange)](https://github.com/linzhenyuyuchen/ifree/actions)
[![PyPI](https://img.shields.io/pypi/v/ifree)](https://pypi.org/project/ifree/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ifree.svg)
[![License](https://img.shields.io/pypi/l/ifree)](https://github.com/linzhenyuyuchen/ifree/blob/master/LICENSE)

**ifree** is a small python package for making you free. 

- Radiomics Feature Extraction
- Feature Pre-Processing and Selecting, 
- Dicom Processing
- Crop CT/MR according to RT Structure
- Calculate dose volume histogram (DVH)
  
One of the main benefits of **ifree** is that it handles data easily (see [example below](#example)).


## Installation

**ifree** can be installed with pip:

```bash
pip install ifree
```
For the bleeding edge version, install directly from github (consider adding `--force-reinstall`):
```bash
pip install git+git://github.com/linzhenyuyuchen/ifree.git
```
or by cloning the repo:
```bash
git clone https://github.com/linzhenyuyuchen/ifree.git
cd ifree
python setup.py install
```

## Example

### Feature Processing

#### Radiomics Feature Extractor

radiomics.yaml

```yaml
# https://github.com/Radiomics/pyradiomics/blob/master/examples/exampleSettings/exampleMR_NoResampling.yaml
# https://github.com/Radiomics/pyradiomics/blob/master/examples/exampleSettings/MR_2D_extraction.yaml
imageType:
  Original: {}
  LoG:
   # If the in-plane spacing is large (> 2mm), consider removing sigma value 1.
   sigma: [ 1.0, 3.0, 5.0 ]
  Wavelet: {}

setting:
  label: 1 # 255
#  interpolator: 'sitkBSpline'
#  resampledPixelSpacing: [2, 2, 2]

#  normalize: true
#  normalizeScale: 100  # This allows you to use more or less the same bin width.

featureClass:
  firstorder:
  shape:
  glcm:
  glrlm:
  glszm:
  gldm:
  ngtdm:
```

```python
from ifree import processing

idx = list()
imagePaths = list()
maskPaths = list()
paramPath = "./radiomics.yaml"
outputPath = "./features.csv"

helper = processing.FeatureExtractor(idx, imagePaths, maskPaths, paramPath, outputPath)
helper.extract(force=True)

```

#### Feature Pre-Processing

```python
df_train = pd.DataFrame()
df_test = pd.DataFrame()
helper = processing.FeatureProcess(df_train, df_test)
# preprocessing methods
helper.simpleImpute(strategy='mean') # mean, median, most_frequent, constant
helper.standardScale()
helper.normalizer()
helper.minMaxScaler()
helper.pca(n_components=10)
array_train = helper.X_train
array_test = helper.X_test
```


#### Feature Selector

```python
x_array = np.array()
y_list = list()
featureNames = list()
selector = processing.FeatureSelector(x_array, y_list, featureNames)
# selection methods
x_array_new, y_array_new = selector.univarSelector(top_k=600, method_name="f_classif", inplace=True)
print(x_array_new.shape)
x_array_new, y_array_new = selector.ttestSelector(inplace=True)
print(x_array_new.shape)
x_array_new, y_array_new = selector.mannSelector(inplace=True)
print(x_array_new.shape)
x_array_new, y_array_new = selector.modelSelector(inplace=True)
print(x_array_new.shape)
# name of selected features 
print(selector.featureNames)
```

---

### Dicom Processing

#### Crop CT/MR according to RT

```python
from ifree import dicom

# get paths for CT, MR, DOSE and RT
fileDir = "./p/"
ctfiles, rtfile, mrfiles, dosefile, patientID, patientName = dicom.GetFilePath(fileDir)

# get MRs or CTs related to RT and copy them to new dir
patientNames, patientIDs, id2mrs, id2rt, id2dose = dicom.archiveFiles(old_Dir, new_Dir, copyed=True)

# crop ROI-MASK and its CT or MR
roiName = "ctv"
newDir = "./p/"
newSize = [100, 100, 100] # leave None to get origin size
idx_done_list = dicom.cropROI(id2mrs, id2rt, roiName, newDir, newSize)

idx_done_list = dicom.cropDose(id2dose, id2rt, roiName, newDir, newSize)
```

### Get Dose-Volume Histogram (DVH)

```python
# Get Dose-Volume Histogram (DVH)
rtssfile = "./RT.dicom"
rtdosefile = "./RS.dicom"
# percents is the space between 0-100
roiname = "Body"
data, percents = dicom.get_dvh_of_key(rtssfile, rtdosefile)
dose_list, max_value, min_value, mean_value = data[roiname]

# saved to dataframe
dose_features = {}
roiname = "Body"
for i, (pid, rtssfile) in enumerate(id2rt.item()):
    rtdosefile = id2dose[pid]
    # percents is the space between 0-100
    data, percents = dicom.get_dvh_of_key(rtssfile, rtdosefile)
    dose_list, max_value, min_value, mean_value = data[roiname]
    tmp = {"pid": pid, "dose_max": max_value, "dose_min": min_value, "dose_mean": mean_value,}
    for i, pc in enumerate(percents):
        tmp[f"dose_{pc}"] = dose_list[i]
    dose_features[i] = tmp
df_dose_features = pd.DataFrame.from_dict(dose_features, orient='index')
df_dose_features.head()
```


For more examples, see the [examples folder](https://github.com/linzhenyuyuchen/ifree/tree/master/examples).
