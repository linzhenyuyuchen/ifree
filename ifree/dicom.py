import os
import cv2
import itk
import vtk
import shutil
import pickle
import pydicom
import argparse
import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from dicompylercore import dvhcalc
from dicompylercore import dicomparser
from vtk.util.numpy_support import vtk_to_numpy

ImageType   =  itk.Image[itk.F, 3]
ImageTypeUC =  itk.Image[itk.UC,3]

def reorientation(input_image):
    region3d = input_image.GetBufferedRegion()
    start3d = region3d.GetIndex()

    # region3d = mask3d.GetBufferedRegion()
    # start3d = region3d.GetIndex()
    # size3d = region3d.GetSize()
    spacing3d = input_image.GetSpacing()
    #new_direction_list = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    new_direction_list = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    new_direction = itk.matrix_from_array(np.array(new_direction_list))
    #print(new_direction)
    

    # def is_equal(old, new, EPS=1e-6):
    #     for a, b in zip(old, new):
    #         if np.abs(a-b) > EPS:
    #             return False
    #     return True
    # if np.array_equal(old_direction, new_direction):
    #     meta = {}
    #     meta['origin'] = input_image.GetOrigin()
    #     meta['direction'] = old_direction
    #     return input_image, meta

    I, J, K = input_image.GetBufferedRegion().GetSize()
    coords = []
    for i in [0, I-1]:
        for j in [0, J-1]:
            for k in [0, K-1]:
                coords.append([i, j, k])
    coords = [input_image.TransformIndexToPhysicalPoint(t) for t in coords]
    coords = np.array(coords)
    min_i, max_i = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_j, max_j = np.min(coords[:, 1]), np.max(coords[:, 1])
    min_k, max_k = np.min(coords[:, 2]), np.max(coords[:, 2])
    new_origin = [min_i, min_j, min_k]

    new_size = [0, 0, 0]
    new_size[0] = int(round((max_i-min_i)/spacing3d[0]))+1
    new_size[1] = int(round((max_j-min_j)/spacing3d[1]))+1
    new_size[2] = int(round((max_k-min_k)/spacing3d[2]))+1

    # resampleFilter = sitk.ResampleImageFilter()
    # resampleFilter.SetOutputSpacing(spacing3d)
    # resampleFilter.SetOutputOrigin(new_origin)
    # resampleFilter.SetOutputDirection(new_direction) ####
    # resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    # resampleFilter.SetSize(new_size)
    # output_image = resampleFilter.Execute(input_image)

    # resample
    interpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = interpolatorType.New()
    resample2dFilter = itk.ResampleImageFilter[ImageType, ImageType].New()
    resample2dFilter.SetInterpolator(interpolator)
    resample2dFilter.SetSize(new_size)
    resample2dFilter.SetOutputOrigin(new_origin)
    resample2dFilter.SetOutputDirection(new_direction)
    resample2dFilter.SetOutputSpacing(spacing3d)
    resample2dFilter.SetOutputStartIndex(start3d)
    resample2dFilter.SetInput(input_image)
    resample2dFilter.Update()
    output_image = resample2dFilter.GetOutput()

    meta = {}
    meta['origin'] = new_origin
    meta['direction'] = new_direction
    return output_image
    #return output_image, meta

#########################
#####    读写NII     ####
#########################
def writeNii(write_path, Image, imageType):
    # write_path ".nii"
    writer = itk.ImageFileWriter[imageType].New()
    writer.SetInput(Image)
    writer.SetFileName(write_path)
    writer.Update()


#########################
#####   读写DICOM    ####
#########################
def readDicom(dcm_path, imageType):
    # 读取dicom文件
    reader = itk.ImageSeriesReader[imageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(dcm_path)
    reader.Update()
    image3d = reader.GetOutput()

    # 特定的窗宽窗位
    windowWidth, windowLevel, minValue = ImageAutoWidthLevel3DFuction(image3d)
    windowLevelFilter = itk.IntensityWindowingImageFilter[imageType, imageType].New()
    windowLevelFilter.SetInput(image3d)
    windowLevelFilter.SetOutputMaximum(350)
    windowLevelFilter.SetOutputMinimum(0)
    windowLevelFilter.SetWindowLevel(float(windowWidth), float(windowLevel))
    windowLevelFilter.Update()

    return windowLevelFilter.GetOutput()

def readDoseDicom(dcm_path, imageType):
    # 读取dicom文件
    reader = itk.ImageSeriesReader[imageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileName(dcm_path)
    reader.Update()
    image3d = reader.GetOutput()

    return image3d

def ReadRTDicom(rtPath,image,ContourList):
    pointXYZ = []
    UNReadALL = False
    if len(ContourList) != 0 :
        UNReadALL = True
        ContourList = [string.upper() for string in ContourList]

    # origin spacing
    origin  = image.GetOrigin()
    spacing = image.GetSpacing()
    size    = image.GetLargestPossibleRegion().GetSize()
    index   = image.GetLargestPossibleRegion().GetIndex()
    # get the rotation matrix and the inverse
    directionMatrix = itk.GetArrayFromVnlMatrix(image.GetDirection().GetVnlMatrix().as_matrix())
    rotMat    = np.zeros(9)
    rotMatInv = np.zeros(9)
    for i in range(3):
        for j in range(3):
            rotMat[i*3 + j] = directionMatrix[i, j]
            rotMatInv[i*3 +j] = directionMatrix[j, i]

    #read RT
    rtdcm_ds = pydicom.read_file(rtPath, force=True)
    #校验 RT
    try:
        ROIContourSequenceA      = rtdcm_ds[0x3006, 0x0039].value
        StructureSetROISequenceA = rtdcm_ds[0x3006, 0x0020].value
    except Exception:
        print('Problem locating [0x3006,0x0020],[0x3006, 0x0039] - Is this a valid RT Structure file? ')
        return

    # loop through structures
    ROI = {}
    DirOfNumStructureSetROIA = {}
    DirOfNumROIContourA = {}
    for StructureSetROI,ROIContour in zip(StructureSetROISequenceA,ROIContourSequenceA):
        try:
            DirOfNumStructureSetROIA[StructureSetROI[0x3006, 0x0022].value] = StructureSetROI
            DirOfNumROIContourA     [ROIContour     [0x3006, 0x0084].value] = ROIContour
        except Exception:
            print('Problem locating [0x3006, 0x0022],[0x3006, 0x0084] - Is this a valid RT Structure file? ')
            return

    for ROIContour in ROIContourSequenceA:
        try:
            StructureSetROI = DirOfNumStructureSetROIA[ROIContour[0x3006, 0x0084].value]
            OrganName = StructureSetROI[0x3006, 0x0026].value
            if UNReadALL and (OrganName.upper() not in ContourList):
                continue
        except Exception:
            print('Problem locating [0x3006, 0x0026],[0x3006, 0x0084] - Is this a valid RT Structure file? ')
            return

        # vtkPoly contours
        contours = vtk.vtkPolyData()
        # now loop through each item or this structure
        # eg one prostate region on a single slice is an item
        cell = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        pointId = 0

        try:
            # loop for this organ
            for roi in ROIContour[0x3006, 0x0040].value:
                polyLine = vtk.vtkPolyLine()
                data = np.array(roi[0x3006, 0x0050].value)

                npts = len(data) // 3

                for j in range(npts):
                    p = [float(data[j * 3 + 0]), float(data[j * 3 + 1]), float(data[j * 3 + 2])]
                    # consider the direction
                    p = linearTransform(rotMat, origin, p)
                    pointXYZ.append(p)
                    pointId = points.InsertNextPoint(p)
                    polyLine.GetPointIds().InsertNextId(pointId)

                # start connected to end
                p = [float(data[0]), float(data[1]), float(data[2])]
                # consider the direction
                p = linearTransform(rotMat, origin, p)
                pointId = points.InsertNextPoint(p)
                polyLine.GetPointIds().InsertNextId(pointId)
                cell.InsertNextCell(polyLine)
        except Exception as e:
            print(e)
            print('Problem locating [0x3006, 0x0050],[0x3006, 0x0040] - Is this a valid RT Structure file? ')
            return

        contours.SetPoints(points)
        contours.SetLines(cell)
        contours.Modified()

        # get bounds
        bounds = list(contours.GetBounds())
        # roi origin spacing size for X
        Xmin = numpy.floor((bounds[0] - origin[0]) / spacing[0]) - 1
        Xmax = numpy.floor((bounds[1] - origin[0]) / spacing[0]) + 2
        Xextent = Xmax - Xmin
        originX = Xmin * spacing[0] + origin[0]

        # roi origin spacing size for Y
        Ymin = numpy.floor((bounds[2] - origin[1]) / spacing[1]) - 1
        Ymax = numpy.floor((bounds[3] - origin[1]) / spacing[1]) + 2
        Yextent = Ymax - Ymin
        originY = Ymin * spacing[1] + origin[1]

        # roi origin spacing size for Z
        Zmin = numpy.floor((bounds[4] - origin[2]) / spacing[2]) - 1
        Zmax = numpy.floor((bounds[5] - origin[2]) / spacing[2]) + 2
        Zextent = Zmax - Zmin
        originZ = Zmin * spacing[2] + origin[2]

        originRoi = [originX, originY, originZ]
        sizeRoi = [Xextent + 1, Yextent + 1, Zextent + 1]

        # convert the contours to itkImage
        #print('contours points number: ', contours.GetNumberOfPoints())
        #print('contours cells number: ', contours.GetNumberOfCells())
        if contours.GetNumberOfPoints() == 0:
            continue

        # prepare the binary image 's voxel grid
        whiteImage = vtk.vtkImageData()
        whiteImage.SetSpacing(spacing)
        whiteImage.SetExtent(0, int(sizeRoi[0]) - 1, 0, int(sizeRoi[1]) - 1, 0, int(sizeRoi[2]) - 1)
        whiteImage.SetOrigin(originRoi)
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # vtkIdType
        count = whiteImage.GetNumberOfPoints()
        whiteImage.GetPointData().GetScalars().Fill(1)

        # polygonal data --> image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetTolerance(0)  # important if extruder->SetVector(0, 0, 1) !!!
        pol2stenc.SetInputData(contours)
        pol2stenc.SetOutputOrigin(originRoi)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        # cut the corresponding white image and set the background:
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilData(pol2stenc.GetOutput())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()

        imageOrganVtk = imgstenc.GetOutput()

        # vtk image to numpy and to itk
        rows, cols, pages = imageOrganVtk.GetDimensions()
        imageNp = vtk_to_numpy(imageOrganVtk.GetPointData().GetScalars())
        imageNp = numpy.ascontiguousarray(imageNp.reshape(pages, cols, rows) * 255)
        # assert imageNp.shape == imageOrganVtk.GetDimensions()
        imageOrganItk = itk.GetImageFromArray(imageNp)
        imageOrganItk.SetSpacing(imageOrganVtk.GetSpacing())
        # for origin direction is considered
        originRoi = imageOrganVtk.GetOrigin()
        originRoi = linearTransform(rotMatInv, origin, originRoi)
        imageOrganItk.SetOrigin(originRoi)
        # direction
        imageOrganItk.SetDirection(image.GetDirection())
        #  image push back
        ROI[OrganName] = imageOrganItk

    return ROI, pointXYZ


#########################
#####  获取文件地址   ####
#########################
def GetFilePath(ImagePath):
    ctfiles = []
    mrfiles = []
    z = []
    z_mr = []
    patientID = ''
    patientName = ''
    rtfile = ''
    dosefile = ''
    for file_path in os.listdir(ImagePath):
        if not file_path.lower().endswith('dcm'):
            print('Warn, {} is not a dicom file'.format(file_path))
            continue
        dcm = pydicom.read_file(ImagePath + '/' + file_path, force=True)
        try:
            dcm.SOPClassUID
            patientName = dcm.PatientName
        except:
            print('file {} is corrupted, pass'.format(file_path))
            continue
        # RS rt
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            rtfile = ImagePath + '/' + file_path
            patientID = dcm.PatientID
        # ct
        elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            ctfiles.append(ImagePath + '/' + file_path)
            z.append(dcm.SliceLocation)
        # mr
        elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
            mrfiles.append(ImagePath + '/' + file_path)
            z_mr.append(dcm.SliceLocation)
        # RD dose
        elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2':
            dosefile = ImagePath + '/' + file_path
        # RP
        elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.5':
            pass
    if len(ctfiles)>0:
        zipped = zip(z, ctfiles)
        sort_zipped = sorted(zipped, key=lambda x: x[0])
        result = zip(*sort_zipped)
        z, ctfiles = [list(x) for x in result]

    if len(mrfiles)>0:
        zipped2 = zip(z_mr, mrfiles)
        sort_zipped2 = sorted(zipped2, key=lambda x: x[0])
        result2 = zip(*sort_zipped2)
        z_mr, mrfiles = [list(x) for x in result2]

    return ctfiles, rtfile, mrfiles, dosefile, patientID, patientName


#########################
#####  获取窗宽窗位   ####
#########################
def ImageAutoWidthLevel3DFuction(image3d):
    minValue = 0
    maxValue = 1
    windowLevel = 0
    windowWidth = 0

    npImage = itk.GetArrayFromImage(image3d)

    minValue = npImage.min()
    maxValue = npImage.max()

    # if gray level less than 500
    if (maxValue - minValue < 500):
        windowLevel = np.floor((minValue + maxValue) / 2.0)
        windowWidth = maxValue - minValue
        return windowWidth, windowLevel, minValue

    # parameters for histogram
    binNum = 500
    lowPercent = 0.04
    upPercent = 0.999
    lowWindow = minValue
    upWindow = maxValue
    windowLevel = (minValue + maxValue) / 2.0
    windowWidth = maxValue - minValue

    # image information
    imageSize = image3d.GetLargestPossibleRegion().GetSize()

    # ImageToHistogramFilter
    npImage = itk.GetArrayFromImage(image3d)
    histo, edge = np.histogram(npImage, bins=binNum, range=(lowWindow, upWindow))

    # summation to compare
    tempAddSum = 0
    tpHistogramNum = histo.size
    tempTotalNum = imageSize[0] * imageSize[1] * imageSize[2]

    # find the lower value
    for i in range(tpHistogramNum):

        tempAddSum += histo[i]
        if (tempAddSum > tempTotalNum * lowPercent):
            lowWindow = i * (maxValue - minValue) / (1.0 * binNum) + minValue;
            break

    # find the upper value
    tempAddSum = 0
    for i in range(tpHistogramNum):

        tempAddSum += histo[tpHistogramNum - i - 1]
        if (tempAddSum > tempTotalNum * (1 - upPercent)):
            upWindow = maxValue - i * (maxValue - minValue) / (1.0 * binNum);
            break

    # final we get the W/L
    windowWidth = upWindow - lowWindow
    windowLevel = lowWindow + windowWidth / 2
    return windowWidth, windowLevel, minValue


def linearTransform(rotMat, origin, coord):
    """
    Linear transform $3\times1$ vector to a new coordinate system given by
    $3\times3$ rotMat (in row) and a $3\times 1$ origin. If see rotMat's
    rows as base vectors of a new coord. system, this transform is mapping old
    system to new system!
    :param rotMat: $ 3x3 rotation matrix in row expansion
    :param origin: $ 3x1 rotation origin
    :param coord: 3x1 array in old coordinate system
    :return: 3x1 array in new coordinate system
    """
    nodeX = rotMat[0] * (coord[0] - origin[0]) + rotMat[1] * (coord[1] - origin[1]) + \
            rotMat[2] * (coord[2] - origin[2]) + origin[0]
    nodeY = rotMat[3] * (coord[0] - origin[0]) + rotMat[4] * (coord[1] - origin[1]) + \
            rotMat[5] * (coord[2] - origin[2]) + origin[1]
    nodeZ = rotMat[6] * (coord[0] - origin[0]) + rotMat[7] * (coord[1] - origin[1]) + \
            rotMat[8] * (coord[2] - origin[2]) + origin[2]
    return [nodeX, nodeY, nodeZ]


#########################
### 识别单个患者图像信息 ##
#########################
def GetRTPath(ImagePath):
    patientID = ''
    rtfile = ''
    for file_path in os.listdir(ImagePath):
        if not file_path.lower().endswith('dcm'):
            print('Warn, {} is not a dicom file'.format(file_path))
            continue
        dcm = pydicom.read_file(ImagePath + '/' + file_path, force=True)
        try:
            dcm.SOPClassUID
        except:
            print('file {} is corrupted, pass'.format(file_path))
            continue
        # RS rt
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            rtfile = ImagePath + '/' + file_path
            patientID = dcm.PatientID
            patientName = dcm.PatientName
            return rtfile, patientID, patientName
    return None

def list_dcm_files(directory):
    '''
        find all dicom files in a directory

        :param directory: a directory contain dicom files, this directory
            can contain sub-directories
        :return: a list, which each item contrain dcm's SOPInstanceUID, and value contain
            path, SOPClassUID
    '''
    dcm_files = []
    for root, dirs, fns in os.walk(directory):
        for fn in fns:
            dcm_fn = os.path.join(root, fn)
            dcm = pydicom.read_file(dcm_fn, force=True)
            if hasattr(dcm, 'SOPClassUID'):
                if not hasattr(dcm, 'SOPInstanceUID'):
                    print(
                        '{} is a dicom file, but dose not have SOPInstanceUID'.format(dcm_fn))
                    continue
                dcm_files.append({
                    'path': dcm_fn,
                    'SOPClassUID': dcm.SOPClassUID,
                    'SOPInstanceUID': dcm.SOPInstanceUID,
                    'file': dcm,
                })
            else:
                print('{} is not a dicom file'.format(dcm_fn))
    return dcm_files


def get_structure_related_image_files(structure_dcm, dcm_files):
    '''
        find all relevant dicom images of a structure,
        1. if structure file have ContourImageSequence, use ReferencedSOPInstanceUID in structure
        2. else, if structure file have FrameOfReferenceUID, use FrameOfReferenceUID
    '''
    related_image_files = []
    if hasattr(structure_dcm, 'ReferencedFrameOfReferenceSequence'):
        ReferencedFrameOfReferenceSequence = structure_dcm.ReferencedFrameOfReferenceSequence
        if len(ReferencedFrameOfReferenceSequence) > 1:
            print('More than one ReferencedFrameOfReference found')
        if len(ReferencedFrameOfReferenceSequence) == 0:
            print(
                'Cannot find ReferencedFrameOfReference in structure')
        ReferencedFrameOfReference = ReferencedFrameOfReferenceSequence[0]
        if hasattr(ReferencedFrameOfReference, 'RTReferencedStudySequence'):
            try:
                RTReferencedStudy = ReferencedFrameOfReference.RTReferencedStudySequence[0]
                RTReferencedSeriesSequence = RTReferencedStudy.RTReferencedSeriesSequence
                RTRefenecedSeries = RTReferencedSeriesSequence[0]
                for ContourImage in RTRefenecedSeries.ContourImageSequence:
                    ReferencedSOPInstanceUID = ContourImage.ReferencedSOPInstanceUID
                    find_image = False
                    for item in dcm_files:
                        if item['SOPInstanceUID'] == ReferencedSOPInstanceUID:
                            find_image = True
                            related_image_files.append(item)
                            break
                    if not find_image:
                        print('Cannot find {} which referenced by structure'.format(
                            ReferencedSOPInstanceUID))
            except:
                pass
        else:
            if hasattr(ReferencedFrameOfReference, 'FrameOfReferenceUID'):
                FrameOfReferenceUID = ReferencedFrameOfReference.FrameOfReferenceUID
                try:
                    for item in dcm_files:
                        if item['file'].FrameOfReferenceUID == FrameOfReferenceUID:
                            related_image_files.append(item)
                except:
                    pass
    else:
        if hasattr(structure_dcm, 'FrameOfReferenceUID'):
            FrameOfReferenceUID = ReferencedFrameOfReference.FrameOfReferenceUID
            try:
                for item in dcm_files:
                    if item['file'].FrameOfReferenceUID == FrameOfReferenceUID:
                        related_image_files.append(item)
            except:
                pass
    #return related_image_files
    return sorted(related_image_files, key=lambda x: x["file"][0x0008,0x0018].value)

#########################
###  寻找与RT相关的文件  ##
#########################
def archiveFiles(mrDir, mrDir2=None, copyed=False):
    patientNames = []
    patientIDs = []
    id2mrs = {}
    id2rt = {}
    id2dose = {}
    for patient_dir in os.listdir(mrDir):
        print("================================")
        # get RT
        patienDir = os.path.join(mrDir, patient_dir)
        ctfiles, rtfile, mrfiles, dosefile, patientID, patientName = GetFilePath(patienDir)


        print("rtfile ", rtfile)
        print("dosefile ", dosefile)
        print("patientID ", patientID)
        print("patientName ", patientName)
        if rtfile=='' or dosefile=='':
            print("No RT or DOSE found!")
            continue
        # get MRs
        #dcms_all = list_dcm_files(patienDir)
        #dcms_related = get_structure_related_image_files(pydicom.read_file(rtfile), dcms_all)
        #print("MR or CT related #", len(dcms_related))
        # make new dir
        if copyed:
            new_patient_dir = os.path.join(mrDir2, str(patientID))
            os.makedirs(new_patient_dir, exist_ok=True)
            # copy RT file
            shutil.copyfile(rtfile, os.path.join(new_patient_dir, rtfile.split("/")[-1]))
            # copy Dose file
            shutil.copyfile(dosefile, os.path.join(new_patient_dir, dosefile.split("/")[-1]))
            # copy MR files
        if len(ctfiles)>0:
        #if len(dcms_related)>0:
            patientNames.append(patientName)
            patientIDs.append(patientID)
            #id2mrs[patientID] = [ dc["path"] for dc in dcms_related]
            id2mrs[patientID] = [ dc for dc in ctfiles]
            id2rt[patientID] = rtfile
            id2dose[patientID] = dosefile
            if copyed:
                for p in dcms_related:
                    shutil.copyfile(p["path"], os.path.join(new_patient_dir, p["path"].split("/")[-1]))
    return patientNames, patientIDs, id2mrs, id2rt, id2dose


def cropROI(id2mrs, id2rt, roiName, newDir, newSize=None):
    s = roiName
    selected_position_list = [s, s.lower(), s.upper()]
    ContourList = selected_position_list
    idx_done = []
    for pid, mrfiles in id2mrs.items():
        try:
            selected_position = None
            image3d = readDicom(mrfiles, ImageType)
            ROI, pointXYZ = ReadRTDicom(id2rt[pid], image3d, ContourList)

            for i in selected_position_list:
                if i in ROI.keys():
                    selected_position = i
            if selected_position is None:
                print("No roi found!")
                continue
            else:
                print(f"{selected_position} found!")
            mask3d = ROI[selected_position]

            
            origion3d = mask3d.GetOrigin()
            spacing3d = mask3d.GetSpacing()
            region3d = mask3d.GetBufferedRegion()
            start3d = region3d.GetIndex()
            size3d = region3d.GetSize()
            if newSize is None:
                newOrigin = origion3d
            else:
                newOrigin = [0,0,0]
                newOrigin[0] = origion3d[0] + spacing3d[0]*size3d[0]/2 - spacing3d[0]*newSize[0]/2
                newOrigin[1] = origion3d[1] + spacing3d[1]*size3d[1]/2 - spacing3d[1]*newSize[1]/2
                newOrigin[2] = origion3d[2] + spacing3d[2]*size3d[2]/2 - spacing3d[2]*newSize[2]/2
            
            #image3d_reorient = reorientation(image3d)
            

            # resample Mask size
            interpolatorType = itk.LinearInterpolateImageFunction[ImageTypeUC, itk.D]
            interpolator = interpolatorType.New()
            resample2dFilter = itk.ResampleImageFilter[ImageTypeUC, ImageTypeUC].New()
            resample2dFilter.SetInterpolator(interpolator)
            resample2dFilter.SetSize(newSize)
            resample2dFilter.SetOutputOrigin(newOrigin)
            resample2dFilter.SetOutputSpacing(spacing3d)
            resample2dFilter.SetOutputStartIndex(start3d)
            resample2dFilter.SetInput(mask3d)
            resample2dFilter.Update()
            reampleMask3d = resample2dFilter.GetOutput()

            # resample MR according to Mask
            interpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
            interpolator = interpolatorType.New()
            resample2dFilter = itk.ResampleImageFilter[ImageType, ImageType].New()
            resample2dFilter.SetInterpolator(interpolator)
            resample2dFilter.SetSize(newSize)
            resample2dFilter.SetOutputOrigin(newOrigin)
            resample2dFilter.SetOutputSpacing(spacing3d)
            resample2dFilter.SetOutputStartIndex(start3d)
            resample2dFilter.SetInput(image3d)
            resample2dFilter.Update()
            reampleImage3d = resample2dFilter.GetOutput()

            new_patient_dir = os.path.join(newDir, str(pid))
            os.makedirs(new_patient_dir, exist_ok=True)

            # save mask
            mask_nii_path = os.path.join(new_patient_dir, "mask.nii")
            writeNii(mask_nii_path, reampleMask3d, ImageTypeUC)
            
            image_nii_path = os.path.join(new_patient_dir, "mr.nii")
            writeNii(image_nii_path, reampleImage3d, ImageType)
            idx_done.append(pid)
        except Exception as e:
            print(e)
    return idx_done 


def cropDose(id2dose, id2rt, roiName, newDir, newSize=None):
    ImageType = itk.Image[itk.F, 3]
    ImageTypeUC =  itk.Image[itk.UC,3]

    s = roiName
    selected_position_list = [s, s.lower(), s.upper(), s.capitalize(), s.title(), s.replace("-", " ")]
    ContourList = selected_position_list
    idx_done = []
    for pid, rtfile in id2rt.items():
        try:
            selected_position = None
            image3d = readDoseDicom(id2dose[pid], ImageType)
            image3d = readDicom(mrfiles, ImageType)
            ROI, pointXYZ = ReadRTDicom(rtfile, image3d, ContourList)

            for i in selected_position_list:
                if i in ROI.keys():
                    selected_position = i
            if selected_position is None:
                print("No roi found!")
                continue
            else:
                print(f"{selected_position} found!")

            mask3d = ROI[selected_position]
            
            origion3d = mask3d.GetOrigin()
            spacing3d = mask3d.GetSpacing()
            region3d = mask3d.GetBufferedRegion()
            start3d = region3d.GetIndex()
            if newSize is None:
                size3d = region3d.GetSize()
            else:
                size3d = newSize
            
            #image3d_reorient = reorientation(image3d)

            # resample Dose according to Mask
            interpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
            interpolator = interpolatorType.New()
            resample2dFilter = itk.ResampleImageFilter[ImageType, ImageType].New()
            resample2dFilter.SetInterpolator(interpolator)
            resample2dFilter.SetSize(size3d)
            resample2dFilter.SetOutputOrigin(origion3d)
            resample2dFilter.SetOutputSpacing(spacing3d)
            resample2dFilter.SetOutputStartIndex(start3d)
            resample2dFilter.SetInput(image3d)
            resample2dFilter.Update()
            reampleImage3d = resample2dFilter.GetOutput()

            new_patient_dir = os.path.join(newDir, str(pid))
            if not os.path.exists(new_patient_dir):
                os.makedirs(new_patient_dir)

            # save mask
            mask_nii_path = os.path.join(new_patient_dir, "mask.nii")
            writeNii(mask_nii_path, mask3d, ImageTypeUC)
            
            image_nii_path = os.path.join(new_patient_dir, "dose.nii")
            writeNii(image_nii_path, reampleImage3d, ImageType)
            idx_done.append(pid)
        except Exception as e:
            print(e)
    return idx_done 
#########################
###     获取dvh数据     ##
#########################
def get_dvh_of_key(rtssfile, rtdosefile, keyname=None):
    percents = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]
    RTss = dicomparser.DicomParser(rtssfile)
    RTstructures = RTss.GetStructures()

    keyname_list = [keyname, keyname.lower(), keyname.upper(), keyname.capitalize(), keyname.title(), keyname.replace("-", " ")]

    # Generate the calculated DVHs
    # array_counts = []
    # array_bins = []
    data = {}
    for key, structure in RTstructures.items():
        #print(structure)
        roiname = structure["name"]
        if (keyname is not None) and (roiname not in keyname_list):
            continue
        res = dvhcalc.get_dvh(rtssfile, rtdosefile, key)
        if (len(res.counts) and res.counts[0] != 0):
            # print('DVH found for ' + keyname)
            # array_counts.append( res.counts) 
            # array_bins.append( res.bins)
            doses = [res.statistic(f"D{i}").value for i in percents]
            data[roiname] = (doses, res.max, res.min, res.mean)

    return data, percents