#!/usr/bin/python3

import sys
import os
import h5py
import numpy as np

def printUsageAndExit():
  print("Usage: {} model=<model-prototxt> solver=<solver-prototxt> id=<string> inputblob=<string> downsample=<int>{{1,<nDims>}} elementsize=<float>{{<nDims>}} pad_input=<int>{{1,<nDims>}} pad_output=<int>{{1,<nDims>}} [name=<string>] [description=<string>] [inputdataset=<string>] [padding=(mirror)|(zero)] [normalization=<0-3>] [additionalArgs=<arg1:value1>[,<arg2:value2>]*] outfile=<outfile-hdf5>".format(sys.argv[0]))
  print("")
  print("Generate an hdf5 model definition file that complies to the Unet Segmentation ImageJ plugin. Strings containing white-spaces in the parameter list must be quoted. If parameters are omitted the default value is chosen.")
  print("")
  print("  model          - The prototxt file containing the network architecture (required)")
  print("  solver         - The prototxt file containing the solver settings (required)")
  print("  id             - A string that uniquely identifies this model (required)")
  print("  inputblob      - The name of the input blob for testing (required)")
  print("  downsample     - The number of downsampling steps in the analysis path. When passing a scalar value, isotropic downscaling is assumed. For anisotropic downsampling give a vector containing downsampling factors for each dimension. (required)")
  print("  elementsize    - The voxel size in micrometers. Pass as many values as your data contains dimensions. (required)")
  print("  pad_input      - The boundary pixel loss due to convolutions in the analysis path. If the loss differs for different input dimensions, pass as many values as there are input dimensions. (required)")
  print("  pad_output     - The boundary pixel loss due to convolutions in the synthesis path. If the loss differs for different input dimensions, pass as many values as there are input dimensions. (required)")
  print("  name           - The network name presented to the user (Default: <id>)")
  print("  description    - A more detailed description of the model (Default: '')")
  print("  inputdataset   - The name of the input hdf5 dataset for testing (Default: 'data')")
  print("  padding        - You can choose between mirroring and zero padding (Default: mirror)")
  print("  normalization  - The value normalization mode. (Default: 0)")
  print("                   0 - no normalization")
  print("                   1 - min-max normalization to [0, 1]")
  print("                   2 - zero mean, unit standard deviation")
  print("                   3 - Unit vector norm (only applicable to multi-channel data)")
  print("  additionalArgs - You can provide an additional set of arguments as comma separated arg:value pairs that will be written to the hdf5 file")
  print("")
  print("  outfile        - The output HDF5 model definition file. If a file with that name already exists it will be replaced. (required)")
  sys.exit(1)

###############################################################################
# Parse command line
###############################################################################

if len(sys.argv) == 1:
  printUsageAndExit()

requiredArguments = ["model", "solver", "id", "inputblob", "downsample", "elementsize", "pad_input", "pad_output", "outfile"]

arguments = { "description": "", "inputdataset": "data", "padding": "mirror", "normalization": 0, "additionalArgs": "" }
for arg in sys.argv[1:]:
  try:
    arguments[arg.split("=")[0]] = arg.split("=")[1]
  except:
    print("Could not parse command line argument '{}'".format(arg))
    print("")
    printUsageAndExit()

for requiredArgument in requiredArguments:
  if requiredArgument not in arguments:
    print("Required argument '{}' missing".format(requiredArgument))
    print("")
    printUsageAndExit()

if "name" not in arguments:
  arguments["name"] = arguments["id"]

try:
  arguments["elementsize"] = np.double(arguments["elementsize"].split(","))
except:
  print("Invalid elementsize given")
  print("")
  printUsageAndExit()

try:
  arguments["normalization"] = np.int32(arguments["normalization"])
  if arguments["normalization"] < 0 or arguments["normalization"] > 3:
    print("Invalid normalization type given")
    print("")
    printUsageAndExit()
except:
  print("Invalid normalization type given")
  print("")
  printUsageAndExit()

try:
  arguments["downsample"] = np.int32(arguments["downsample"].split(","))
except:
  print("Invalid downsample factors given")
  print("")
  printUsageAndExit()

try:
  arguments["pad_input"] = np.int32(arguments["pad_input"].split(","))
except:
  print("Invalid input padding given")
  print("")
  printUsageAndExit()

try:
  arguments["pad_output"] = np.int32(arguments["pad_output"].split(","))
except:
  print("Invalid output padding given")
  print("")
  printUsageAndExit()

###############################################################################
# Create model definition hdf5 file
###############################################################################

# This 'with' statement ensures that the file is closed immediately after
# reading. Not crucial, but good style!
with open(arguments["model"], 'r') as modelprototxtfile:
  modelprototxt = modelprototxtfile.read()
with open(arguments["solver"], 'r') as solverprototxtfile:
  solverprototxt = solverprototxtfile.read()

# Create new model definition file. If a file with that name exists it is
# truncated
outfile = h5py.File(arguments["outfile"], "w")

# Save fixed-length strings for compatibility with MATLAB and libBlitzHdf5
ds = outfile.create_dataset("/.unet-ident", data=np.string_(arguments["id"]))
ds = outfile.create_dataset("/unet_param/name", data=np.string_(arguments["name"]))
ds = outfile.create_dataset("/unet_param/description", data=np.string_(arguments["description"]))
ds = outfile.create_dataset("/unet_param/input_dataset_name", data=np.string_(arguments["inputdataset"]))
ds = outfile.create_dataset("/unet_param/input_blob_name", data=np.string_(arguments["inputblob"]))
ds = outfile.create_dataset("/unet_param/padding", data=np.string_(arguments["padding"]))
ds = outfile.create_dataset("model_prototxt", data=np.string_(modelprototxt))
ds = outfile.create_dataset("solver_prototxt", data=np.string_(solverprototxt))

# Numerical data
outfile.create_dataset("/unet_param/element_size_um", data=arguments["elementsize"])
outfile.create_dataset("/unet_param/normalization_type", data=arguments["normalization"])
outfile.create_dataset("/unet_param/downsampleFactor", data=arguments["downsample"])
outfile.create_dataset("/unet_param/padInput", data=arguments["pad_input"])
outfile.create_dataset("/unet_param/padOutput", data=arguments["pad_output"])

if arguments["additionalArgs"] != "":
  argList = arguments["additionalArgs"].split(",")
  for arg in argList:
    dsName = arg.split(":")[0]
    dsValue = arg.split(":")[1]
    try:
      value = int(dsValue)
      outfile.create_dataset(dsName, data=value)
      continue
    except ValueError:
      pass
    try:
      value = np.float32(dsValue)
      outfile.create_dataset(dsName, data=value)
      continue
    except ValueError:
      pass
    # Fall back to string if the argument could not be converted to a number
    outfile.create_dataset(dsName, data=dsValue)

outfile.close()
