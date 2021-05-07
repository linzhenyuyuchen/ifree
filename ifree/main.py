from ifree import processing
from ifree import dicom
import os
import pickle
import pandas as pd
import numpy as np
import itk
def readNII(dcm_path):
    ImageType = itk.Image[itk.F, 3]
    reader = itk.ImageSeriesReader[ImageType].New()
    reader.SetFileName(dcm_path)
    reader.Update()
    image3d = reader.GetOutput()
    return itk.GetArrayFromImage(image3d)

if __name__ == "__main__":

    # get MRs or CTs related to RT and copy them to new dir
    old_Dir = "../originalDicom/"
    newDir = "../new/"
    patientNames, patientIDs, id2mrs, id2rt, id2dose = dicom.archiveFiles(old_Dir)
    pickle.dump((patientNames, patientIDs, id2mrs, id2rt, id2dose), open(newDir+"archive.pkl", "wb"))
    #data = pickle.load(open(newDir+"archive.pkl", "rb"))
    #patientNames, patientIDs, id2mrs, id2rt, id2dose = data


    # crop ROI-MASK and its MR
    newDir = "../new/MRandMask/"
    newSize = None # leave None to get origin size
    idx_done_list = dicom.cropROI(id2mrs, id2rt, "GTV", newDir, newSize)
    pickle.dump(idx_done_list, open(newDir+"MRandMask.pkl", "wb"))
    #idx_done_list = pickle.load(open(newDir+"MRandMask.pkl", "rb"))



    # get MR-ROI radiomics feature
    newDir = "../new/MRandMask/"
    idx = os.listdir(newDir)
    imagePaths = [newDir + i +"/mr.nii" for i in idx]
    maskPaths = [newDir + i +"/mask.nii" for i in idx]
    paramPath = "./radiomics.yaml"
    outputPath = "../new/MRandMask_features.csv"
    helper = processing.FeatureExtractor(idx, imagePaths, maskPaths, paramPath, outputPath)
    helper.extract(force=True)


    # crop ROI-MASK and its Dose
    newDir = "../new/DOSEandMask/"
    newSize = None # leave None to get origin size
    idx_done_list = dicom.cropDose(id2dose, id2rt, "GTV", newDir, newSize)
    pickle.dump(idx_done_list, open(newDir+"DOSEandMask.pkl", "wb"))
    #idx_done_list = pickle.load(open(newDir+"DOSEandMask.pkl", "rb"))

    # Get Dose-Volume Histogram (DVH)
    newDir = "../new/"
    s = "BODY"
    dose_features = {}
    selected_position_list = [s, s.lower(), s.upper(), s.capitalize(), s.title(), s.replace("-", " ")]
    for i, (pid, rtssfile) in enumerate(id2rt.items()):
        print(f"{i}/{len(id2rt)}")
        rtdosefile = id2dose[pid]
        # percents is the space between 0-100
        data, percents = dicom.get_dvh_of_key(rtssfile, rtdosefile, s)
        for roiname in selected_position_list:
            if roiname in data.keys():
                selected_position = roiname
        dose_list, max_value, min_value, mean_value = data[selected_position]
        tmp = {"pid": pid, "dose_max": max_value, "dose_min": min_value, "dose_mean": mean_value,}
        for j, pc in enumerate(percents):
            tmp[f"dose_{pc}"] = dose_list[j]
        dose_features[i] = tmp
    df_dose_features = pd.DataFrame.from_dict(dose_features, orient='index')
    df_dose_features.to_csv(newDir+"dvh_BODY.csv", index=False)
    # df_dose_features.head()