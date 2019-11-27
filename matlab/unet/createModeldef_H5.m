function createModeldef_H5( model_prototxt, solver_prototxt, unet_param, outfileH5, varargin )
% function createModeldef_H5( model_prototxt, solver_prototxt, unet_param, outfileH5, varargin )
%
%   Writes model definition hdf5 file
%       The hdf5 file contains the network definition, the solver definition and
%       additional parameters, such as the number of input channels.

%   Parameters:
%
%       model_prototxt
%           The file containing the network definition prototxt file
%
%       solver_prototxt
%           The file containing the solver definition prototxt file
%
%       unet_param
%           The parameter struct containing the following fields
%
%               ident
%                   Unique model identifier (string)
%
%               name
%                   Model name (string)
%
%               description
%                   Model description (string)
%
%               input_blob_name
%                   Name of input blob in TEST phase (string)
%
%               input_dataset_name
%                   Name of input hdf5 dataset in TEST phase (string)
%                   Default = 'data'
%
%               input_num_channels
%                   Number of channels of the input data
%
%               element_size_um
%                   Input data element sizes in micrometer
%                   [sy sx]     (2D data)
%                   [sz sy sx]  (3D data)
%               --> implicitely define the number of data dimensions
%
%               normalization_type
%                   Input data normalization:
%                   0: no normalization
%                   1: [0, 1]
%                   2: zero mean and unit standard deviation
%                   3: channel norm in [0, 1]
%
%               padding
%                   Input data padding option. Supported options are:
%                   'zero': zero padding
%                   'mirror': mirror padding
%
%               downsampleFactor
%                   Total downsampling factor
%                   Depends on the full cascade of "downscaling" operations
%                   in the network, such as pooling operations for example.
%                   --> Example: 4x max pooling (2x2, stride 2) --> downsampleFactor = 16
%                   [d]:        (scalar) Uniform downsampling factor for all data dimensions
%                   [dy dx]:    Downsampling factors specified for all data dimensions (2d data)
%                   [dz dy dx]: Downsampling factors specified for all data dimensions (3d data)
%
%               padInput
%                   Input padding
%                   [p]:        (scalar) Uniform input padding for all data dimensions
%                   [py px]:    Input padding specified for all data dimensions (2d data)
%                   [pz py px]: Input padding specified for all data dimensions (3d data)
%
%               padOutput
%                   Output padding
%                   [p]:        (scalar) Uniform output padding for all data dimensions
%                   [py px]:    Output padding specified for all data dimensions (2d data)
%                   [pz py px]: Output padding specified for all data dimensions (3d data)
%
%               (Optional) pixelwise_loss_weights
%                   Parameters used for generating pixel-wise loss weights (2d data)
%
%                   The parameter struct contains the following fields
%                       foregroundBackgroundRatio
%                       sigma1_um
%                       borderWeightFactor
%                       borderWeightSigma_um
%
%                   --> If not specified, paramters are not written to the
%                   model definition hdf5 file.
%
%       outfileH5
%           The output hdf5 file
%
%   Optional additional parameters (varargin):
%
%       param - Enables storing of an "input-size --> GPU-memory" map into the hdf5 file
%
%           *********************************************************************************************
%           * THIS FUNCTIONALITY IS EXPERIMENTAL AND TESTED ONLY IN A SPECIFIC ENVIRONMENT AND SETTING! *
%           *********************************************************************************************
%
%           Option 1: Use existing memory map
%
%               mapInputNumPxGPUMemMB - N x 2 matrix
%                   N: number of measurements,
%                   Columns: Input size (total number of input elements), Used GPU-memory by Caffe (in MB)
%
%           Option 2 (2d input data only, LINUX ONLY): Generates memory map, by
%               measuring the GPU-memory used by Caffe for increasing input sizes
%
%               ******************************************
%               * THIS FUNCTIONALITY IS EXPERIMENTAL     *
%               * AND IMPLEMENTED FOR 2D INPUT DATA ONLY *
%               ******************************************
%               An external Linux shell script is generated and called,
%               that runs a MATLAB function to perform the measurements.
%               The standard setting is to omit the optional parameters (see below)
%               and let the function run until the GPU-memory limit is reached and
%               a Caffe segmentation fault forces the process to stop.
%               (requires the tool 'nvidia-smi' to be installed on the Linux system)
%
%               CAFFE_ROOT
%                   Caffe root directory
%
%               CAFFE_LIBRARY
%                   Caffe library directory
%
%               gpu_device
%                   GPU device ID (used by Caffe)
%
%               Optional parameter to set custom MATLAB binary call
%
%                   MATLAB_BIN
%
%               --> If not specified, MATLAB_BIN = 'matlab' is used.
%
%
%               Optional parameters to set explicit stopping criteria
%               (You can use one or both parameters):
%
%                   maxIter
%                       maximal number of iterations
%
%                   maxInputNumPx
%                       maximum input size
%
%               --> If not specified (standard setting), the method will only stop
%               when the GPU-memory limit is reached and a Caffe segmentation fault
%               forces the process to stop.
%
%   Example usage (U-Net with 4 pooling layers):
%   -------------------------------------------------------------------------
%  CAFFE_ROOT = '/path_to_caffe_root';
%  resultdir = '/results_directory';
%  NETNAME = 'examplenet_id';
%  name = 'examplenet name';
%  description = 'examplenet full description';
%
%  addpath([CAFFE_ROOT '/matlab/unet'])
%
%  model_prototxt = [resultdir '/' NETNAME '-train.prototxt'];
%  solver_prototxt = [resultdir '/' NETNAME '-solver.prototxt'];
%  unet_param = struct;
%  unet_param.ident = NETNAME;
%  unet_param.name = name;
%  unet_param.description = description;
%  unet_param.input_blob_name = 'data3';
%  unet_param.input_num_channels = int32(1);
%  unet_param.element_size_um = single([0.5 0.5]);
%  unet_param.normalization_type = int32(1);
%  unet_param.padding = 'mirror';
%  unet_param.downsampleFactor = int32(16);
%  d4a_size = 0;
%  unet_param.padInput =  int32( (((d4a_size * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2);
%  unet_param.padOutput = int32(((((d4a_size     - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2);
%  % UNCOMMENT TO SPECIFY ADDITIONAL PARAMETERS
%  %  unet_param.pixelwise_loss_weights.foregroundBackgroundRatio = 0.1;
%  %  unet_param.pixelwise_loss_weights.sigma1_um = 5.0;
%  %  unet_param.pixelwise_loss_weights.borderWeightFactor = 50.0;
%  %  unet_param.pixelwise_loss_weights.borderWeightSigma_um = 3.0;
%  outfileH5 = [resultdir '/' NETNAME '-modeldef.h5'];
%
%
%  % A) Disabled storing of an "input-size --> GPU-memory" map
%  createModeldef_H5( model_prototxt, solver_prototxt, unet_param, outfileH5);
%
%
%  % B) Enables storing of an "input-size --> GPU-memory" map
%
%    % Option 1: Use existing memory map
%    resultdir = '/misc/lmbraid17/bensch/u-net-3d/2dcellnet';
%    NETNAME = '2dcellnet_v7w5t2l1_mapInputNumPxGPUMemMB';
%    modeldef = readModeldef_H5([resultdir '/' NETNAME '-modeldef.h5'], 0);
%    param.mapInputNumPxGPUMemMB = modeldef.unet_param.mapInputNumPxGPUMemMB;
%    createModeldef_H5( model_prototxt, solver_prototxt, unet_param, outfileH5, param);
%
%    % Option 2 (2d input data only, LINUX ONLY): Generates memory map, by
%    % measuring the GPU-memory used by Caffe for increasing input sizes
%    param.CAFFE_ROOT = CAFFE_ROOT;
%    param.CAFFE_LIBRARY = '/path_to_caffe_library';
%    param.gpu_device = 0;
%    % UNCOMMENT TO SET CUSTOM MATLAB BINARY CALL (STANDARD IS: 'matlab')
%    param.MATLAB_BIN = '/misc/software-lin/matlabR2015a/bin/matlab';
%    % UNCOMMENT TO SET EXPLICIT STOPPING CRITERIA (You can use one or both parameters)
%    %  param.maxIter = 100;
%    %  param.maxInputNumPx = 100*100;
%    createModeldef_H5( model_prototxt, solver_prototxt, unet_param, outfileH5, param);
%
%
%    % Test: Try to read written model definition hdf5 file
%  modeldef = readModeldef_H5(outfileH5);
%   -------------------------------------------------------------------------
%
%   Copyright 2018  Robert Bensch, Thorsten Falk
%
%   Core Facility Image Analysis, Department of Computer Science, University of Freiburg


if (nargin > 4)
  param = varargin{1};
end

if ~isfield(unet_param, 'input_dataset_name')
  unet_param.input_dataset_name = 'data';
end

% measure GPU memory
% *********************************************************************************************
% * THIS FUNCTIONALITY IS EXPERIMENTAL AND TESTED ONLY IN A SPECIFIC ENVIRONMENT AND SETTING! *
% *********************************************************************************************
if (exist('param','var'))
  if (isfield(param, 'mapInputNumPxGPUMemMB'))
	unet_param.mapInputNumPxGPUMemMB = int32(param.mapInputNumPxGPUMemMB);
  else

	%******************************************
	%* THIS FUNCTIONALITY IS EXPERIMENTAL     *
	%* AND IMPLEMENTED FOR 2D INPUT DATA ONLY *
	%******************************************
	assert(length(unet_param.element_size_um)==2)
	assert(length(unet_param.downsampleFactor)==1)
	assert(length(unet_param.padInput)==1)
	assert(length(unet_param.padOutput)==1)

	% set standard parameters, if not specified
	% -- MATLAB_BIN
	if (~isfield(param, 'MATLAB_BIN'))
	  param.MATLAB_BIN = 'matlab';
	end

	param.pwd = pwd;
	bashScriptName = 'tmp-measureGPUMem.sh';
	fid = fopen([param.pwd '/' bashScriptName],'w');
	fprintf(fid, '#!/bin/bash\n');
	fprintf(fid, '\n');
	fprintf(fid, ['matlab_exec=' param.MATLAB_BIN '\n']);
	fprintf(fid, '\n');
	fprintf(fid, 'LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH ${matlab_exec} -nodesktop -nosplash <<''EOF''\n', param.CAFFE_LIBRARY);
	fprintf(fid, 'model_prototxt = ''%s'';\n', model_prototxt);
	fprintf(fid, 'unet_param.input_blob_name = ''%s'';\n', unet_param.input_blob_name);
	fprintf(fid, 'unet_param.input_dataset_name = ''%s'';\n', unet_param.input_dataset_name);
	fprintf(fid, 'unet_param.input_num_channels = %d;\n', unet_param.input_num_channels);
	fprintf(fid, 'unet_param.padding = ''%s'';\n', unet_param.padding);
	fprintf(fid, 'unet_param.downsampleFactor = %d;\n', unet_param.downsampleFactor);
	fprintf(fid, 'unet_param.padInput = %d;\n', unet_param.padInput);
	fprintf(fid, 'unet_param.padOutput = %d;\n', unet_param.padOutput);
	fprintf(fid, 'unet_param.gpu_device = %d;\n', param.gpu_device);
	if (isfield(param, 'maxIter'))
	  fprintf(fid, 'unet_param.maxIter = %d;\n', param.maxIter);
	end
	if (isfield(param, 'maxInputNumPx'))
	  fprintf(fid, 'unet_param.maxInputNumPx = %d;\n', param.maxInputNumPx);
	end
	fprintf(fid, 'outfileMat = ''%s'';\n', [param.pwd '/' 'tmp-measureGPUMem.mat']);
	fprintf(fid, 'addpath(''%s/matlab'');\n', param.CAFFE_ROOT);
	fprintf(fid, 'addpath(''%s/matlab/unet'');\n', param.CAFFE_ROOT);
	fprintf(fid, 'measureGPUMem( model_prototxt, unet_param, outfileMat);\n');
	fprintf(fid, 'exit\n');
	fprintf(fid, 'EOF');
	fclose(fid);
	[status, cmdout] = system(['chmod ugo+rx ' bashScriptName]);

	% calling external script for measuring GPU memory
	disp('************************************************************')
	disp('* Calling external script for measuring GPU memory...      *')
	disp('*                                                          *')
	disp('*   Note: The external script is supposed to crash         *')
	disp('*   when maximum GPU memory is reached, so don''t worry ;)  *')
	disp('************************************************************')
	disp('This takes some time, please wait...')
	[status, cmdout] = system(['./' bashScriptName],'-echo')

	ld = load([param.pwd '/' 'tmp-measureGPUMem.mat']);
	offset_caffe_GPU_memory = ld.caffe_GPU_memory(1,1);
	unet_param.mapInputNumPxGPUMemMB = int32([ld.input_sizes(:,1).*ld.input_sizes(:,2) ld.caffe_GPU_memory(:,1)-offset_caffe_GPU_memory]);

	unet_param.mapInputNumPxGPUMemMB

	disp('*******************************************************')
	disp('* Measuring GPU memory successfully completed.        *')
	disp('*******************************************************')

	% delete temporary files
	delete([param.pwd '/' bashScriptName]);
	delete([param.pwd '/' 'tmp-measureGPUMem.mat']);
  end
end


% read model definition file
disp(['reading ' model_prototxt]);
fileID = fopen(model_prototxt);
model_def = fread(fileID,'*char')';
fclose(fileID);

% read solver definition file
disp(['reading ' solver_prototxt]);
fileID = fopen(solver_prototxt);
solver_def = fread(fileID,'*char')';
fclose(fileID);


% delete existing file
delete(outfileH5);

% write numerical datasets
disp(['writing ' outfileH5]);
h5create(outfileH5, '/unet_param/input_num_channels', [1], 'Datatype', 'int32');
h5write(outfileH5, '/unet_param/input_num_channels', unet_param.input_num_channels);
h5create(outfileH5, '/unet_param/element_size_um', [numel(unet_param.element_size_um)], 'Datatype', 'single');
h5write(outfileH5, '/unet_param/element_size_um', unet_param.element_size_um);
h5create(outfileH5, '/unet_param/normalization_type', [1], 'Datatype', 'int32');
h5write(outfileH5, '/unet_param/normalization_type', unet_param.normalization_type);
h5create(outfileH5, '/unet_param/downsampleFactor', [length(unet_param.downsampleFactor)], 'Datatype', 'int32');
h5write(outfileH5, '/unet_param/downsampleFactor', unet_param.downsampleFactor);
h5create(outfileH5, '/unet_param/padInput', [length(unet_param.padInput)], 'Datatype', 'int32');
h5write(outfileH5, '/unet_param/padInput', unet_param.padInput);
h5create(outfileH5, '/unet_param/padOutput', [length(unet_param.padOutput)], 'Datatype', 'int32');
h5write(outfileH5, '/unet_param/padOutput', unet_param.padOutput);
% -- pixelwise_loss_weights
if (isfield(unet_param, 'pixelwise_loss_weights'))
  h5create(outfileH5, '/unet_param/pixelwise_loss_weights/foregroundBackgroundRatio', [1], 'Datatype', 'single');
  h5write(outfileH5, '/unet_param/pixelwise_loss_weights/foregroundBackgroundRatio', unet_param.pixelwise_loss_weights.foregroundBackgroundRatio);
  h5create(outfileH5, '/unet_param/pixelwise_loss_weights/sigma1_um', [1], 'Datatype', 'single');
  h5write(outfileH5, '/unet_param/pixelwise_loss_weights/sigma1_um', unet_param.pixelwise_loss_weights.sigma1_um);
  h5create(outfileH5, '/unet_param/pixelwise_loss_weights/borderWeightFactor', [1], 'Datatype', 'single');
  h5write(outfileH5, '/unet_param/pixelwise_loss_weights/borderWeightFactor', unet_param.pixelwise_loss_weights.borderWeightFactor);
  h5create(outfileH5, '/unet_param/pixelwise_loss_weights/borderWeightSigma_um', [1], 'Datatype', 'single');
  h5write(outfileH5, '/unet_param/pixelwise_loss_weights/borderWeightSigma_um', unet_param.pixelwise_loss_weights.borderWeightSigma_um);
end
% -- mapInputNumPxGPUMemMB
if (isfield(unet_param, 'mapInputNumPxGPUMemMB'))
  h5create(outfileH5, '/unet_param/mapInputNumPxGPUMemMB', [size(unet_param.mapInputNumPxGPUMemMB)], 'Datatype', 'int32');
  h5write(outfileH5, '/unet_param/mapInputNumPxGPUMemMB', unet_param.mapInputNumPxGPUMemMB);
end

% write string datasets using the simple interface
hdf5write(outfileH5, '/.unet-ident', unet_param.ident, 'WriteMode', 'append');
hdf5write(outfileH5, '/unet_param/name', unet_param.name, 'WriteMode', 'append');
hdf5write(outfileH5, '/unet_param/description', unet_param.description, 'WriteMode', 'append');
hdf5write(outfileH5, '/unet_param/input_blob_name', unet_param.input_blob_name, 'WriteMode', 'append');
hdf5write(outfileH5, '/unet_param/input_dataset_name', unet_param.input_dataset_name, 'WriteMode', 'append');
hdf5write(outfileH5, '/unet_param/padding', unet_param.padding, 'WriteMode', 'append');
hdf5write(outfileH5, '/model_prototxt', model_def, 'WriteMode', 'append');
hdf5write(outfileH5, '/solver_prototxt', solver_def, 'WriteMode', 'append');

end
