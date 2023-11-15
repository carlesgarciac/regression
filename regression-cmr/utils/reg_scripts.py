import numpy as np
import SimpleITK as sitk
import kornia as kornia
import kornia.geometry as KG
import nibabel as nib
import torch
import itk
from skimage import data
import napari 
import matplotlib.pyplot as plt
import os
# import opticalflow3D as op3d


# use_cuda: bool = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

MRI_scan = nib.load('/home/carlesgc/Projects/regression/train_data_regression/images/A0S9V9_sa.nii.gz').get_fdata()

fixed = itk.GetImageFromArray(MRI_scan[:,:,:,7])
moving = itk.GetImageFromArray(MRI_scan[:,:,:,8])

# MRI_scan = torch.Tensor(MRI_scan)

# print(MRI_scan.shape)

# registrator = KG.ImageRegistrator('similarity')
# homo = registrator.register(MRI_scan[:,:,:,0], MRI_scan[:,:,:,1])

# print(homo)

# MRI_scan = itk.imread('/home/carlesgc/Projects/regression/train_data_regression/images/A0S9V9_sa.nii.gz', itk.F)

def itk_registration(fixed_image, moving_image):
    registration_method = sitk.ImageRegistrationMethod()   
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.

    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
        
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-2, numberOfIterations=100, deltaConvergenceTolerance=0.01)

    final_transformation = registration_method.Execute(fixed_image, moving_image)

    return final_transformation

# def command_iteration(filter):
#     print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

# fixed = MRI_scan[0,6,:,:]

# print(MRI_scan.shape)


# moving = MRI_scan[8,6,:,:]

# matcher = sitk.HistogramMatchingImageFilter()
# matcher.SetNumberOfHistogramLevels(1024)
# matcher.SetNumberOfMatchPoints(7)
# matcher.ThresholdAtMeanIntensityOn()
# moving = matcher.Execute(moving, fixed)

# # The basic Demons Registration Filter
# # Note there is a whole family of Demons Registration algorithms included in
# # SimpleITK
# demons = sitk.DemonsRegistrationFilter()
# demons.SetNumberOfIterations(50)
# # Standard deviation for Gaussian smoothing of displacement field
# demons.SetStandardDeviations(1.0)

# demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

# displacementField = demons.Execute(fixed, moving)

# print("-------")
# print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
# print(f" RMS: {demons.GetRMSChange()}")

# outTx = sitk.DisplacementFieldTransform(displacementField)

# # sitk.WriteTransform(outTx, sys.argv[3])

# if "SITK_NOSHOW" not in os.environ:
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(100)
#     resampler.SetTransform(outTx)

#     out = resampler.Execute(moving)
#     simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
#     simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
#     # Use the // floor division operator so that the pixel type is
#     # the same for all three images which is the expectation for
#     # the compose filter.
#     cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
#     sitk.Show(cimg, "DeformableRegistration1 Composition", debugOn=True)


# MRI_scan = np.uint8(MRI_scan)

print(MRI_scan.shape)
## Moving fixed

parameter_object = itk.ParameterObject.New()
parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')

parameter_object.AddParameterMap(parameter_map_rigid)
parameter_object.AddParameterMap(parameter_map_bspline)

transformation, params = itk.elastix_registration_method(fixed, moving, parameter_object=parameter_object)
# deformation_field = itk.transformix_deformation_field( MRI_scan[8,:,:,:], params)

# Load Transformix Object
transformix_object = itk.TransformixFilter.New(transformation)
transformix_object.SetTransformParameterObject(params)
 
# Set advanced options
transformix_object.SetComputeDeformationField(True)

# Set output directory for spatial jacobian and its determinant,
# default directory is current directory.
transformix_object.SetOutputDirectory('exampleoutput/')

# Update object (required)
transformix_object.UpdateLargestPossibleRegion()

# Results of Transformation
result_image_transformix = transformix_object.GetOutput()
deformation_field = transformix_object.GetOutputDeformationField()

# print(params)
# transformation = itk_registration(MRI_scan[0,:,:,:], MRI_scan[8,:,:,:])
# print(transformation.shape)
# print(params)
# deformation_image = sitk.GetImageFromArray(transformation[0,:,:], isVector=True)
# displacement_transform = sitk.DisplacementFieldTransform(deformation_image)

# sitk.show(displacement_transform)
# plt.quiver(transformation)
# plt.show()
# print(transformation.shape)

# napari.view_image(MRI_scan)
# viewernapari.view_image(data.cell())
# viewer.add_image(MRI_scan[1,7,:,:])
# viewer.add_image(transformation)

# reg_field = MRI_scan[0,:,:,:]/transformation
# reg_field = np.nan_to_num(reg_field, nan=0.0)

deformation_field = itk.array_from_image(deformation_field)
deformation_field = deformation_field.transpose(2,1,0,3)


transformation = itk.array_from_image(transformation)
moving = itk.array_from_image(moving)
# print(transformation.shape)
transformation = transformation.transpose(2,1,0)
moving = moving.transpose(2,1,0)

print(transformation.shape)

viewer = napari.Viewer()
viewer.add_image(MRI_scan[:,:,:,7])
viewer.add_image(MRI_scan[:,:,:,8])
# viewer.add_image(deformation_field[:,:,:,0])
# viewer.add_image(deformation_field[:,:,:,1])
# viewer.add_image(deformation_field[:,:,:,2])
viewer.add_image(transformation)
viewer.add_image(transformation-moving)
# viewer.add_image(reg_field)
# viewer.add_image(transformation)
# viewer.add_image(displacementField)
napari.run()



