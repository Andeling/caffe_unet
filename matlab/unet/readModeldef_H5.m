function modeldef = readModeldef_H5( infileH5, varargin )
% function modeldef = readModeldef_H5( infileH5 )
%
%   Reads model definition hdf5 file
%
%   Parameters:
%
%       infileH5
%           The model definition hdf5 file
%
%   Optional additional parameter (varargin):
%
%       ensureRequiredParameters - Binary switch to ensure that
%       all required parameters are read from the hdf5 file
%           0: all available parameters are read from the hdf5 file (catches errors and displays error messages)
%           1: ensure that all required parameters are read (throws error otherwise)
%       --> If not specified, ensureRequiredParameters = 1
%
%
%   Return value:
%
%       modeldef - The model definition struct containing the following fields
%           (maybe just a subset or empty in case the optional parameter is set to ensureRequiredParameters = 0)
%
%           unet_param
%               The parameter struct containing the fields defined in
%               createModeldef_H5.m
%
%           model_prototxt
%               String containing the network definition prototxt file
%
%           solver_prototxt
%               String containing the solver definition prototxt file
%
%   Robert Bensch
%
%   Core Facility Image Analysis, Department of Computer Science, University of Freiburg


if (nargin > 1)
  ensureRequiredParameters = varargin{1};
  assert(ensureRequiredParameters==0 || ensureRequiredParameters==1);
else
  ensureRequiredParameters = 1;
end

modeldef = struct;

switch ensureRequiredParameters

  % All available parameters are read from the hdf5 file
  %	(THIS CODE IS A REPLICATE OF CODE FOR 'case 1' (SEE BELOW),
  %	WITH ADDED TRY CATCH BLOCKS)
  case 0

	% Read numerical datasets
	try
	  modeldef.unet_param.input_num_channels = double(h5read( infileH5, '/unet_param/input_num_channels'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.element_size_um =  double(h5read( infileH5, '/unet_param/element_size_um'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.normalization_type =  double(h5read( infileH5, '/unet_param/normalization_type'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.downsampleFactor =  double(h5read( infileH5, '/unet_param/downsampleFactor'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.padInput =  double(h5read( infileH5, '/unet_param/padInput'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.padOutput =  double(h5read( infileH5, '/unet_param/padOutput'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.pixelwise_loss_weights = struct;
	  modeldef.unet_param.pixelwise_loss_weights.foregroundBackgroundRatio = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/foregroundBackgroundRatio'));
	  modeldef.unet_param.pixelwise_loss_weights.sigma1_um = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/sigma1_um'));
	  modeldef.unet_param.pixelwise_loss_weights.borderWeightFactor = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/borderWeightFactor'));
	  modeldef.unet_param.pixelwise_loss_weights.borderWeightSigma_um = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/borderWeightSigma_um'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.mapInputNumPxGPUMemMB = h5read( infileH5, '/unet_param/mapInputNumPxGPUMemMB');
	catch err
	  disp(err.identifier);
	end

	% Read string datasets
	try
	  h5string  = hdf5read( infileH5, '/.unet-ident');
	  modeldef.unet_param.ident = h5string.Data;
	catch err
	  disp(err.identifier);
	end
	try
	  h5string  = hdf5read( infileH5, '/unet_param/name');
	  modeldef.unet_param.name = h5string.Data;
	catch err
	  disp(err.identifier);
	end
	try
	  h5string  = hdf5read( infileH5, '/unet_param/description');
	  modeldef.unet_param.description = h5string.Data;
	catch err
	  disp(err.identifier);
	end
	try
	  h5string  = hdf5read( infileH5, '/unet_param/input_blob_name');
	  modeldef.unet_param.input_blob_name = h5string.Data;
	catch err
	  disp(err.identifier);
	end
	try
	  h5string  = hdf5read( infileH5, '/unet_param/padding');
	  modeldef.unet_param.padding = h5string.Data;
	catch err
	  disp(err.identifier);
	end
	try
	  h5string  = hdf5read( infileH5, '/model_prototxt');
	  modeldef.model_prototxt = h5string.Data;
	catch err
	  disp(err.identifier);
	end
	try
	  h5string  = hdf5read( infileH5, '/solver_prototxt');
	  modeldef.solver_prototxt = h5string.Data;
	catch err
	  disp(err.identifier);
	end

  % Ensure that all required parameters are read
  case 1

	% Read numerical datasets
	modeldef.unet_param.input_num_channels = double(h5read( infileH5, '/unet_param/input_num_channels'));
	modeldef.unet_param.element_size_um =  double(h5read( infileH5, '/unet_param/element_size_um'));
	modeldef.unet_param.normalization_type =  double(h5read( infileH5, '/unet_param/normalization_type'));
	modeldef.unet_param.downsampleFactor =  double(h5read( infileH5, '/unet_param/downsampleFactor'));
	modeldef.unet_param.padInput =  double(h5read( infileH5, '/unet_param/padInput'));
	modeldef.unet_param.padOutput =  double(h5read( infileH5, '/unet_param/padOutput'));
	try
	  modeldef.unet_param.pixelwise_loss_weights = struct;
	  modeldef.unet_param.pixelwise_loss_weights.foregroundBackgroundRatio = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/foregroundBackgroundRatio'));
	  modeldef.unet_param.pixelwise_loss_weights.sigma1_um = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/sigma1_um'));
	  modeldef.unet_param.pixelwise_loss_weights.borderWeightFactor = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/borderWeightFactor'));
	  modeldef.unet_param.pixelwise_loss_weights.borderWeightSigma_um = double(h5read( infileH5, '/unet_param/pixelwise_loss_weights/borderWeightSigma_um'));
	catch err
	  disp(err.identifier);
	end
	try
	  modeldef.unet_param.mapInputNumPxGPUMemMB = h5read( infileH5, '/unet_param/mapInputNumPxGPUMemMB');
	catch err
	  disp(err.identifier);
	end

	% Read string datasets
	h5string  = hdf5read( infileH5, '/.unet-ident');
	modeldef.unet_param.ident = h5string.Data;
	h5string  = hdf5read( infileH5, '/unet_param/name');
	modeldef.unet_param.name = h5string.Data;
	h5string  = hdf5read( infileH5, '/unet_param/description');
	modeldef.unet_param.description = h5string.Data;
	h5string  = hdf5read( infileH5, '/unet_param/input_blob_name');
	modeldef.unet_param.input_blob_name = h5string.Data;
	h5string  = hdf5read( infileH5, '/unet_param/padding');
	modeldef.unet_param.padding = h5string.Data;
	h5string  = hdf5read( infileH5, '/model_prototxt');
	modeldef.model_prototxt = h5string.Data;
	h5string  = hdf5read( infileH5, '/solver_prototxt');
	modeldef.solver_prototxt = h5string.Data;

end

end