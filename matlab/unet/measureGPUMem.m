function measureGPUMem( model_prototxt, unet_param, outfileMat )
% function measureGPUMem( model_prototxt, unet_param, outfileMat )
%
%   Generates memory map, by measuring the GPU-memory used by Caffe for increasing input sizes
%       The measurement is specific to the input data, the network architecture,
%       the Caffe setting (cpu+gpu, cpu+gpu+cuDNN) and the GPU type. The function
%       is supposed to crash with a Caffe segmentation fault in case the
%       maximum GPU-memory is reached. Measurements are continuously written
%       to the output file after each iteration.
%
%   Parameters:
%
%       model_prototxt
%           The file containing the network definition prototxt file
%
%       unet_param
%           The parameter struct containing the following fields
%
%               input_blob_name
%                   Name of input blob in TEST phase
%
%               input_num_channels
%                   Number of channels of the input data
%
%               downsampleFactor
%                   Total downsampling factor
%                   Depends on the full cascade of "downscaling" operations
%                   in the network, such as pooling operations for example.
%                   --> Example: 4x max pooling (2x2, stride 2) --> downsampleFactor = 16
%                   [d]:        (scalar) Uniform downsampling factor for all data dimensions
%                   [dy dx]:    Downsampling factors specified for all data dimensions (2D data)
%                   [dz dy dx]: Downsampling factors specified for all data dimensions (3D data)
%
%               padInput
%                   Input padding
%                   [p]:        (scalar) Uniform input padding for all data dimensions
%                   [py px]:    Input padding specified for all data dimensions (2D data)
%                   [pz py px]: Input padding specified for all data dimensions (3D data)
%
%               padOutput
%                   Output padding
%                   [p]:        (scalar) Uniform output padding for all data dimensions
%                   [py px]:    Output padding specified for all data dimensions (2D data)
%                   [pz py px]: Output padding specified for all data dimensions (3D data)
%
%               gpu_device
%                    GPU device ID (used by Caffe)
%
%               Optional parameters (You can use one or both parameters):
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
%       outfileMat
%           The MAT file containing the measured information. Is continuously stored after each iteration.
%
%
%   Requirements:
%
%       Requires a LINUX system and the tool 'nvidia-smi' to be installed.
%
%
%   Robert Bensch
%
%   Core Facility Image Analysis, Department of Computer Science, University of Freiburg


%  CAFFE_ROOT=/path_to_caffe_root
%  addpath([CAFFE_ROOT '/matlab'])
%  addpath([CAFFE_ROOT '/matlab/unet'])

% to double
unet_param.downsampleFactor = double(unet_param.downsampleFactor);
unet_param.padInput = double(unet_param.padInput);
unet_param.padOutput = double(unet_param.padOutput);
unet_param.input_num_channels = double(unet_param.input_num_channels);

% get CUDA_VISIBLE_DEVICES
disp(['unet_param.gpu_device = ' num2str(unet_param.gpu_device)]);
[status, cuda_visible_devices_str] = system('echo $CUDA_VISIBLE_DEVICES');
if ( length( str2num( cuda_visible_devices_str)) > 0)
  cuda_visible_devices = str2num( cuda_visible_devices_str);
  cuda_visible_device = cuda_visible_devices(unet_param.gpu_device + 1);
  disp(['use cuda_visible_device = ' num2str(cuda_visible_device)]);
else
  cuda_visible_device = unet_param.gpu_device;
end

% initialization
input_sizes = [];
output_sizes = [];
caffe_GPU_memory = [];

% offset (GPU not in use by this caffe process)
ci = 0;
[status, gpuID_memTotal] = system(['nvidia-smi -i ' num2str(cuda_visible_device) ' -q -d MEMORY | grep Total -m 1']);
if (status ~= 0)
  error('Requires the tool ''nvidia-smi'' to be installed.');
end
[status, gpuID_memUsed] = system(['nvidia-smi -i ' num2str(cuda_visible_device) ' -q -d MEMORY | grep Used -m 1']);
idx1 = strfind(gpuID_memTotal, ':');
idx2 = strfind(gpuID_memTotal, 'MiB');
gpu_memTotal = str2num( gpuID_memTotal(idx1+2:idx2-2) );
idx1 = strfind(gpuID_memUsed, ':');
idx2 = strfind(gpuID_memUsed, 'MiB');
gpu_memUsed = str2num( gpuID_memUsed(idx1+2:idx2-2) );
input_sizes(ci+1,[1:2]) = [0 0];
output_sizes(ci+1,[1:2]) = [0 0];
caffe_GPU_memory(ci+1,[1:2]) = [gpu_memUsed, gpu_memTotal];
%  if (gpu_memUsed > 100)
%    disp('*')
%    disp(['* Used GPU memory is already above 100MiB: ' num2str(gpu_memUsed) 'MiB']);
%    disp(['* GPU memory measurement cannot make use of full GPU capacity!']);
%    disp(['* Try to get a free GPU.']);
%    disp('*')
%    disp('To continue anyway, press any key...')
%    pause;
%  end

fid = fopen(model_prototxt);
trainPrototxt = fread( fid);
fclose(fid);

% iteratively increase input image size and measure GPU memory
condition = true;
while (condition)

  %
  %  increment size on deepest layer
  %
  ci = ci + 1;
  if (ci==1)
	% initialization (minimum size on deepest layer)
	d4a_size = ceil(([1 1] - unet_param.padOutput)/unet_param.downsampleFactor);
  else
	% increment size on deepest layer (alternate between increasing x and y size)
	d4a_size = d4a_size + [mod(ci+1,2) mod(ci,2)];
  end

  %
  %  compute input and output sizes
  %
  input_size = unet_param.downsampleFactor*d4a_size + unet_param.padInput;
  output_size = unet_param.downsampleFactor*d4a_size + unet_param.padOutput;
%    disp(['d4a_size = ' num2str(d4a_size) ' --> insize = ' num2str(input_size) ...
%  		', outsize = ' num2str(output_size)])

  input_sizes(ci+1,[1:2]) = input_size;
  output_sizes(ci+1,[1:2]) = output_size;

  %
  %  create Network with fitting dimensions
  %
  model_def_file = 'tmp-test.prototxt';
  fid = fopen(model_def_file,'w');
%    fprintf(fid, 'input: "%s"\n', unet_param.input_blob_name);
%    fprintf(fid, 'input_shape: { dim: %g dim: %g dim: %g dim: %g}', ...
%        1, unet_param.input_num_channels, input_size(2), input_size(1));
%    fprintf(fid, 'state: { phase: TEST }');
  fprintf(fid, 'state: { phase: TEST }\n');
  fprintf(fid, 'layer {\n');
  fprintf(fid, 'name: "%s"\n', unet_param.input_blob_name);
  fprintf(fid, 'type: "Input"\n');
  fprintf(fid, 'top: "%s"\n', unet_param.input_blob_name);
  fprintf(fid, 'input_param { shape: { dim: %g dim: %g dim: %g dim: %g } }\n', ...
		  1, unet_param.input_num_channels, input_size(2), input_size(1));
  fprintf(fid, '}\n');
  fwrite(fid, trainPrototxt);
  fclose(fid);

  while ( ~exist(model_def_file,'file'))
	pause(0.1);
  end

  caffe.set_mode_gpu();
  if( isfield( unet_param, 'gpu_device'))
	caffe.set_device(unet_param.gpu_device)
  end

  %
  %  initialize network
  %
  net = caffe.Net(model_def_file, 'test');

  % delete temorary file
  delete(model_def_file);

  %
  % forward
  %
  paddedInputSlice = zeros([input_size unet_param.input_num_channels], 'single');
  % TODO: replace by other function because matlab net.forward()
  % uses deprecated caffe forward() function?
  scores_caffe = net.forward( {paddedInputSlice});

  nReps = 20;
  elapsed = zeros(1, nReps);
  for i=1:nReps;
      tic;
      scores_caffe = net.forward( {paddedInputSlice});
      elapsed(i) = toc;
  end

  pause(0.1);

  %
  % measure GPU memory usage
  %
  [status, gpuID_memTotal] = system(['nvidia-smi -i ' num2str(cuda_visible_device) ' -q -d MEMORY | grep Total -m 1']);
  [status, gpuID_memUsed] = system(['nvidia-smi -i ' num2str(cuda_visible_device) ' -q -d MEMORY | grep Used -m 1']);
  idx1 = strfind(gpuID_memTotal, ':');
  idx2 = strfind(gpuID_memTotal, 'MiB');
  gpu_memTotal = str2num( gpuID_memTotal(idx1+2:idx2-2) );
  idx1 = strfind(gpuID_memUsed, ':');
  idx2 = strfind(gpuID_memUsed, 'MiB');
  gpu_memUsed = str2num( gpuID_memUsed(idx1+2:idx2-2) );

  caffe_GPU_memory(ci+1,[1:2]) = [gpu_memUsed, gpu_memTotal];
  caffe_forward_time(ci+1,1) = mean(elapsed);
  caffe_forward_time(ci+1,2) = std(elapsed);

  % reset caffe
  caffe.reset_all();

  [input_sizes caffe_GPU_memory round(100*caffe_GPU_memory(:,1)./caffe_GPU_memory(:,2))]

  delete(outfileMat);
  save( outfileMat, 'input_sizes', 'output_sizes', 'caffe_GPU_memory', ...
        'caffe_forward_time');
  while ( ~exist(outfileMat,'file'))
	pause(0.1);
  end

  curr_inputNumPx = input_sizes(ci+1,1)*input_sizes(ci+1,2);
  if (isfield(unet_param, 'maxIter') && isfield(unet_param, 'maxInputNumPx'))
	condition = ci < unet_param.maxIter && curr_inputNumPx < unet_param.maxInputNumPx;
  elseif (isfield(unet_param, 'maxIter') && ~isfield(unet_param, 'maxInputNumPx'))
	condition = ci < unet_param.maxIter;
  elseif (~isfield(unet_param, 'maxIter') && isfield(unet_param, 'maxInputNumPx'))
	condition = curr_inputNumPx < unet_param.maxInputNumPx;
  else
	condition = true;
  end

end

end
