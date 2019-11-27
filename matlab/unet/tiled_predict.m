function scores = tiled_predict(data, modeldef, modelfile, param)
% function scores = tiled_predict(data, modeldef, modelfile, param)
%
%  Overlap-tile strategy for passing large images through caffe (for U-Net)
%
%  Parameters:
%
%    data
%      The test blob to segment. The function assumes that the
%      dimensions are ordered [x y (z) c n], where x=width, y=height,
%      z=depth, c=#channels, n=#samples. If the model definition is
%      2D, the z-dimension is omitted. Trailing singleton
%      dimensions are removed by MATLAB, therefore the array may be
%      lower-dimensional.
%
%      The data layout corresponds to the layout obtained when
%      loading an hdf5 dataset using hdf5read which reverses the
%      dataset dimensions!
%
%   modeldef
%     The hdf5 file containing the network architecture and
%     additional parameters as the number of input channels,
%     ... This file should be created using createModeldef_H5
%
%   modelfile
%     The file containing the trained network weights. The net is
%     initialized without weight file, if the modelfile = []
%
%   param
%     The parameter struct containing the following fields
%
%       gpu_or_cpu
%         Set GPU, or CPU mode
%         'cpu': CPU mode
%         'gpu': GPU mode
%
%       gpu_device
%         The GPU to use for segmentation
%
%     Average options (You can choose only one):
%
%       average_mirror
%         Use the average over mirrored versions of the input tiles
%         for the final segmentation. Mirroring only in xy-plane for
%         2d and 3d data.
%
%       average_rotate
%         Use the average over rotated versions of the input tiles
%         for the final segmentation. Rotation only in xy-plane for
%         2d and 3d data in 90 degree steps. Currently only
%         available for tiling option 'tile_size' where
%         tile shape in y equals tile shape in x.
%
%     Tiling options (You can choose only one):
%
%       gpu_mem_available_MB (GPU mode only; 2d input data only)
%         In GPU mode you can give the available GPU memory in
%         MB. Internally this will be transformed to
%         mem_available_px via the memory usage table that was
%         recorded during creation of the modelfile. The modelfile
%         can be created using createModeldef_H5.m. If no such
%         table is available, the function will fail!
%
%       mem_available_px (2d input data only)
%         You can give the number of input pixels that can be
%         maximally input to the network. This value must be
%         given including required padding.
%
%       n_tiles (2d and 3d input data)
%         (2d input data only) You can give the total number of
%         tiles as scalar value. The tile shape will be determined
%         to minimize memory consumption.
%         (2d and 3d input data) Alternatively you can give a vector
%         and directly specify the tile layout [(nz) ny nx].
%
%       tile_size (2d and 3d input data)
%         The tile shape [(z) y x] in pixels
%
%
%  Return value:
%
%    scores - The prediction scores with shape [x y (z) cl n], where x,
%      y, z and n equal the input dimensions. cl is the number of
%      classes (for binary segmentation c=2
%      (1=background/2=foreground)).
%
%  Example usage:
%  -------------------------------------------------------------------------
%  % Set paths
%  CAFFE_ROOT = '/path_to_caffe_root';
%  resultdir = '/results_directory';
%  NETNAME = 'examplenet_id';
%  addpath([CAFFE_ROOT '/matlab'])
%  addpath([CAFFE_ROOT '/matlab/unet'])
%
%  % Read input data
%  data = hdf5read([resultdir '/' 'labelledtest_scaled+weights3/Fluo-N2DL-HeLa_01_labelledtest_scaled.h5'], 'data');
%  data = reshape(data, [size(data,1) size(data,2) 1 size(data,3)]);
%  % Specify caffe model file
%  modelfile = [resultdir '/' NETNAME '/' NETNAME '_snapshot_iter_200000.caffemodel.h5'];
%  % Read model definition hdf5 file
%  modeldef = readModeldef_H5([resultdir '/' NETNAME '-modeldef.h5']);
%
%  % Set tiled prediction runtime parameters
%  param = struct;
%  param.gpu_or_cpu = 'gpu';
%  param.gpu_device = 0;
%  param.average_mirror = false;
%  param.average_rotate = false;
%  % Tiling options (You can choose only one):
%  param.gpu_mem_available_MB = 12000;         % GPU mode only; 2d input data only
%  %param.mem_available_px = prod([540 540]);  % 2d input data only
%  %param.n_tiles = 12;                        % 2d input data only
%  %param.n_tiles = [3 4];                     % 2d and 3d input data
%  %param.tile_size = [540 540];               % 2d and 3d input data
%
%  % Run tiled prediction
%  scores = tiled_predict( data, modeldef, modelfile, param );
%
%  % Show results
%  [~, labels] = max(scores,[],3);
%  for t=1:size(scores,4)
%   figure(1);imshow( labels(:,:,1,t)-1);
%   drawnow
%   figure(2);imshow( scores(:,:,2,t)-scores(:,:,1,t),[]);colormap jet; colorbar
%   drawnow
%   pause
%  end
%  -------------------------------------------------------------------------
%
% History:
%
%   2016-06-28, Thorsten Falk
%     Cloned from tiled_predict2.m
%
%   Robert Bensch, Thorsten Falk, Olaf Ronneberger
%
%   Core Facility Image Analysis, Department of Computer Science, University of Freiburg


% set option for computing tiling layout
setOptions = [isfield(param, 'gpu_mem_available_MB'), ...
              isfield(param, 'mem_available_px'), ...
              isfield(param, 'n_tiles'), ...
              isfield(param, 'tile_size')];

if (param.gpu_or_cpu == 'cpu')
    if (setOptions(1))
	error('"Option "gpu_mem_available_MB" is not supported in CPU mode.');
    end
    if ( sum(setOptions(2:end)) ~= 1 )
	error(['Exactly one of the alternative options ' ...
               '"mem_available_px", "n_tiles", or "tile_size" ' ...
               'has to be specified in CPU mode!']);
    end
elseif (param.gpu_or_cpu == 'gpu')
    if ( sum(setOptions) ~= 1 )
	error(['Exactly one of the alternative options ' ...
               '"gpu_mem_available_MB", "mem_available_px", ' ...
               '"n_tiles", or "tile_size" has to be specified in GPU mode!']);
    end
else
    error('param.gpu_or_cpu must be set to either "cpu" or "gpu".');
end

if (param.average_mirror && param.average_rotate)
  error('Only one of the options "average_mirror" or "average_rotate" can be chosen.');
end
if (param.average_rotate && setOptions(4)==0)
  error('Option "average_rotate" currently only available with tiling option "tile_size". In this case tile shape in y must equal tile shape in x.')
end

nDims = length(modeldef.unet_param.element_size_um);
assert(nDims >= 2 && nDims <= 3, [num2str(nDims) '-D not supported']);
inshape = size(data);
assert(length(inshape) >= nDims, ...
       ['The input Array is only ' num2str(length(inshape)) ...
        '-D. The model requires at least ' nDims '-D.']);
shape = inshape(1:nDims);
nChannels = size(data, nDims + 1);
nSamples = size(data, nDims + 2);

disp(['Data shape (x/w y/h [z/d] [c [n]]) = [' num2str(inshape) ']']);
disp([num2str(nDims) '-D model selected: image size = [' ...
      num2str(shape) '], #channels = ' num2str(nChannels) ...
      ', #samples = ' num2str(nSamples)]);

padIn = modeldef.unet_param.padInput(end:-1:1)';
if length(padIn) == 1;
    padIn = padIn .* ones(1, nDims);
end
assert(length(padIn) == nDims, ...
       ['padInput has incompatible length ' num2str(length(size(padIn)))]);
padOut = modeldef.unet_param.padOutput(end:-1:1)';
if length(padOut) == 1;
    padOut = padOut .* ones(1, nDims);
end
assert(length(padOut) == nDims, ...
       ['padOutput has incompatible length ' num2str(length(size(padOut)))]);
dsFactor = modeldef.unet_param.downsampleFactor(end:-1:1)';
if length(dsFactor) == 1;
    dsFactor = dsFactor .* ones(1, nDims);
end
assert(length(dsFactor) == nDims, ...
       ['downsampleFactor has incompatible length ' ...
        num2str(length(size(dsFactor)))]);

setOption = find(setOptions);
switch setOption

  % given memory available (param.gpu_mem_available_MB or
  %                         param.mem_available_px)
  case {1, 2}

    assert(nDims == 2, ...
           ['Options "gpu_mem_available_MB" and "mem_available_px" ' ...
            'currently only available for 2d input data.'])

    % set 'mem_available_px' based on 'gpu_mem_available_MB'
    % (using map from modeldef.unet_param.mapInputNumPxGPUMemMB)
    if setOption == 1
        assert(isfield(modeldef.unet_param, 'mapInputNumPxGPUMemMB'), ...
               'Empty field "modeldef.unet_param.mapInputNumPxGPUMemMB".');
        map = modeldef.unet_param.mapInputNumPxGPUMemMB;
        ix_found = find( map(:,2) <= param.gpu_mem_available_MB, 1, 'last');
        assert(~isempty(ix_found));
        param.mem_available_px = map(ix_found, 1);
        disp(['Set mem_available_px: ' num2str(param.mem_available_px) ...
              ' (based on map entry [' num2str(map(ix_found, 1)) ', ' ...
              num2str(map(ix_found, 2)) ...
              '] from /unet_param/mapInputNumPxGPUMemMB)']);
    end

    assert(~isempty(param.mem_available_px), ...
           'Empty field "param.mem_available_px".');

    % check minimum input size
    d4a_s = ceil((ones(1, nDims) - padOut) ./ dsFactor);
    min_input_size = dsFactor .* d4a_s + padIn;
    assert(param.mem_available_px >= prod(min_input_size), ...
           ['GPU memory available must be at least the minimum ' ...
            'input size of the network, which is [' ...
            num2str(min_input_size) ']!'])
    disp(['param.mem_available_px = ' num2str(param.mem_available_px)]);

    memAvailPx = param.mem_available_px;

    outShapeMax = -ones(1, nDims);
    border_px = round((padIn - padOut) / 2);

    % find maximum reasonable number of tiles in rows
    curr_ntiles = [1 0];
    while (outShapeMax(1) <= 0)
        curr_ntiles(2) = curr_ntiles(2) + 1;
        d4a_s = ceil((ceil(shape ./ curr_ntiles) - padOut) ./ dsFactor);
        input_size = dsFactor .* d4a_s + padIn;
        outShapeMax(1) = floor(memAvailPx / input_size(2)) - 2 * border_px(2);
        d4a_s = floor((outShapeMax(1) * ones(1, nDims) - padOut) ./ dsFactor);
        outShapeMax(1) = dsFactor(2) .* d4a_s(2) + padOut(2);
    end

    % find maximum reasonable number of tiles in columns
    curr_ntiles = [0 1];
    while (outShapeMax(2) <= 0)
        curr_ntiles(1) = curr_ntiles(1) + 1;
        d4a_s = ceil((ceil(shape ./ curr_ntiles) - padOut) ./ dsFactor);
        input_size = dsFactor .* d4a_s + padIn;
        outShapeMax(2) = floor(memAvailPx / input_size(1)) - 2 * border_px(1);
        d4a_s = floor((outShapeMax(2) * ones(1, nDims) - padOut) ./ dsFactor);
        outShapeMax(2) = dsFactor(1) .* d4a_s(1) + padOut(1);
    end

    outShapeMax

    % find optimal tiling layout
    % 	solution with minimum total input size
    max_ntiles = ceil(shape ./ outShapeMax);
    M_tile = 0;				% memory per tile in input image pixels
    Neff_tiles = 0;			% number of effective tiles
    min_M_total = uint64(+Inf);
    ntiles = [1 1];
    for nx = 1:max_ntiles(1)
        for ny = 1:max_ntiles(2)
            tiling = [nx ny];
            d4a_s = ceil((ceil(shape ./ tiling) - padOut) ./ dsFactor);
            input_size = dsFactor .* d4a_s + padIn;
            output_size = dsFactor .* d4a_s + padOut;
            M_tile = prod(input_size);
            Neff_tiles = prod(ceil(shape ./ output_size));
            if M_tile <= param.mem_available_px && ...
                         M_tile * Neff_tiles < min_M_total
                min_M_total = M_tile * Neff_tiles;
                ntiles = tiling;
            end
        end
    end

    % given number of tiles or tile layout
  case 3

    assert(~isempty(param.n_tiles), 'Empty field "param.n_tiles".');
    assert(length(param.n_tiles) == 1 || length(param.n_tiles) == nDims, ...
           'n_tiles vector has incompatible length');

    % number of tiles (e.g. param.n_tiles = 4)
    if length(param.n_tiles) == 1

        assert(nDims == 2, ...
               ['Option "n_tiles" specifying the total number of tiles ' ...
                'as scalar value currently only available for 2d input data.'])

        disp(['param.n_tiles = ' num2str(param.n_tiles)]);
        % find optimal tiling layout
        %   solution with minimum tile input size / minimum total
        %   input size
        max_ntiles = param.n_tiles .* ones(1, nDims);
        M_tile = 0;
        Neff_tiles = 0;
        max_Neff_tiles = 0;
        min_M_tile = +Inf;
        ntiles = ones(1, nDims);
        for tileIdx = 0:prod(max_ntiles) - 1
            tiling = ones(1, nDims);
            tmp = tileIdx;
            for d = nDims:-1:1
                tiling(d) = mod(tmp, max_ntiles(d)) + 1;
                tmp = floor(tmp / max_ntiles(d));
            end
            if prod(tiling) <= param.n_tiles
                d4a_s = ceil((ceil(shape ./ tiling) - padOut) ./ dsFactor);
                input_size = dsFactor .* d4a_s + padIn;
                output_size = dsFactor .* d4a_s + padOut;
                M_tile = prod(input_size);
                Neff_tiles = prod(ceil(shape ./ output_size));
                if Neff_tiles > max_Neff_tiles
                    max_Neff_tiles = Neff_tiles;
                    min_M_tile = M_tile;
                    ntiles = tiling;
                end
                if Neff_tiles == max_Neff_tiles
                    if M_tile < min_M_tile
                        min_M_tile = M_tile;
                        ntiles = tiling;
                    end
                end
            end
        end
        assert(max_Neff_tiles > 0);

	% tile layout in (z,) y and x (e.g. param.n_tiles = [2 2 2])
    else

        assert(length(param.n_tiles) == nDims, ...
               'n_tiles has incompatible dimensionality');
        disp(['param.n_tiles = [' num2str(param.n_tiles) ']']);
        ntiles = param.n_tiles(end:-1:1);

        % round to effective number of tiles in z, y and x
        d4a_s = ceil((ceil(shape ./ ntiles) - padOut) ./ dsFactor);
        output_size = dsFactor .* d4a_s + padOut;
        ntiles = ceil(shape ./ output_size);

    end

    % given tile size in (z,) y and x in pixels (tile_size)
  case 4

    assert(~isempty(param.tile_size), 'Empty field "param.tile_size".');
    assert(length(param.tile_size) == nDims, ...
           'tile_size has incompatible dimensionality');
    disp(['param.tile_size = [' num2str(param.tile_size) ']']);

    % round to next possible tile size
    output_size = param.tile_size(end:-1:1);
    d4a_s = ceil((output_size - padOut) ./ dsFactor);
    output_size = dsFactor .* d4a_s + padOut;

end


%
%  compute input and output sizes (for u-shaped network)
%
if (setOption~=4)
  d4a_s = ceil((ceil(shape ./ ntiles) - padOut) ./ dsFactor);
else
  d4a_s = (output_size - padOut) ./ dsFactor;
  ntiles = ceil(shape ./ output_size);
end
input_size = dsFactor .* d4a_s + padIn
output_size = dsFactor .* d4a_s + padOut
disp(['d4a_size = ' num2str(d4a_s) ' --> insize = ' num2str(input_size) ...
	  ', outsize = ' num2str(output_size)]);

if (param.average_rotate)
  assert(input_size(1)==input_size(2), 'Currently with option "average_rotate" the input size in y must equal the input size in x!');
end

n_tiles = prod(ntiles);
disp(['ntiles (x y [z]) = [' num2str(ntiles) ']']);


%
%  create padded volume with maximum border
%
border = round((input_size - output_size) / 2);
paddedFullVolume = zeros([shape + 2 * border nChannels nSamples], ...
    'single');
if nDims == 2
    paddedFullVolume(border(1) + 1:border(1) + shape(1), ...
                     border(2) + 1:border(2) + shape(2),:,:) = data;
else
    paddedFullVolume(border(1) + 1:border(1) + shape(1), ...
                     border(2) + 1:border(2) + shape(2), ...
                     border(3) + 1:border(3) + shape(3),:,:) = data;
end

if strcmp(modeldef.unet_param.padding, 'mirror')
    pad = border;
    from = border + 1;
    to = border + shape;

    paddedFullVolume(1:from(1) - 1,:,:,:,:) = ...
        paddedFullVolume(from(1) + pad(1):-1:from(1) + 1,:,:,:,:);
    paddedFullVolume(to(1) + 1:end,:,:,:,:) = ...
        paddedFullVolume(to(1) - 1:-1:to(1) - pad(1),:,:,:,:);

    paddedFullVolume(:,1:from(2) - 1,:,:,:) = ...
        paddedFullVolume(:,from(2) + pad(2):-1:from(2) + 1,:,:,:);
    paddedFullVolume(:,to(2) + 1:end,:,:,:) = ...
        paddedFullVolume(:,to(2) - 1:-1:to(2) - pad(2),:,:,:);

    if nDims == 3
        paddedFullVolume(:,:,1:from(3) - 1,:,:) = ...
            paddedFullVolume(:,:,from(3) + pad(3):-1:from(3) + 1,:,:);
        paddedFullVolume(:,:,to(3) + 1:end,:,:) = ...
            paddedFullVolume(:,:,to(3) - 1:-1:to(3) - pad(3),:,:);
    end
end

%
%  create Network with corresponding dimensions
%
model_def_file = [tempname '-caffe_unet-test.prototxt']
fid = fopen(model_def_file,'w');
fprintf(fid, 'state: { phase: TEST }\n');
fprintf(fid, 'layer {\n');
fprintf(fid, 'name: "%s"\n', modeldef.unet_param.input_blob_name);
fprintf(fid, 'type: "Input"\n');
fprintf(fid, 'top: "%s"\n', modeldef.unet_param.input_blob_name);
if nDims == 2
    fprintf(...
        fid, 'input_param { shape: { dim: %g dim: %g dim: %g dim: %g } }\n', ...
        1, nChannels, input_size(2), input_size(1));
else
    fprintf(...
        fid, ['input_param { shape: { dim: %g dim: %g dim: %g dim: ' ...
              '%g dim: %g } }\n'], ...
        1, nChannels, input_size(3), input_size(2), input_size(1));
end
fprintf(fid, '}\n');
fwrite(fid, modeldef.model_prototxt);
fclose(fid);

if param.gpu_or_cpu == 'gpu'
    caffe.set_mode_gpu();
    if isfield(param, 'gpu_device')
	caffe.set_device(param.gpu_device)
    end
else
    caffe.set_mode_cpu();
end

% Initialize a network
if (~isempty(modelfile))
  net = caffe.Net(model_def_file, modelfile, 'test');
else
  warning('Initialize net without weight file!');
  net = caffe.Net(model_def_file, 'test');
end

% delete temorary file
delete(model_def_file);

%
%  do the classification (tiled)
%  average over flipped images
if (param.average_mirror == true)
  numAveragePasses = 4;   % all xy-flips for 2d data; only xy-inplane flips for 3d data
  %numAveragePasses = 2^nDims;
else
  numAveragePasses = 1;
end
%  average over rotated images
if (param.average_rotate == true)
  numAveragePasses = 4;   % all k*90 degree rotations, where k = 0...3, in xy-plane for 2d and 3d data
else
  numAveragePasses = 1;
end
for num = 1:nSamples
    disp(['processing image ' num2str(num)])

    for tileIdx = 0:(prod(ntiles) - 1)

        tile = ones(1, nDims);
        tmp = tileIdx;
        for d = nDims:-1:1
            tile(d) = mod(tmp, ntiles(d));
            tmp = floor(tmp / ntiles(d));
        end

        if any(tile .* output_size + 1 > shape + 2 * border)
            disp(['----> skip [' num2str(tile) '] (out of bounds)']);
            continue;
        else
            disp(['----> tile [' num2str(tile) ']']);
        end

        paddedInputTile = zeros([input_size nChannels], 'single');
        validReg = min(...
            input_size, shape + 2 * border - tile .* output_size);
        if nDims == 2
            paddedInputTile(1:validReg(1),1:validReg(2),:) = ...
                paddedFullVolume( ...
                    tile(1) * output_size(1) + 1:tile(1) * ...
                    output_size(1) + validReg(1), ...
                    tile(2) * output_size(2) + 1:tile(2) * ...
                    output_size(2) + validReg(2),:,num);
        else
            paddedInputTile(1:validReg(1),1:validReg(2),1:validReg(3),:) = ...
                paddedFullVolume( ...
                    tile(1) * output_size(1) + 1:tile(1) * ...
                    output_size(1) + validReg(1), ...
                    tile(2) * output_size(2) + 1:tile(2) * ...
                    output_size(2) + validReg(2), ...
                    tile(3) * output_size(3) + 1:tile(3) * ...
                    output_size(3) + validReg(3),:,num);
        end

        scores_caffe = net.forward({paddedInputTile});
        scoreTile = scores_caffe{1};

        if param.average_mirror == true
            for d = 1:numAveragePasses - 1
                f = zeros(1, nDims);
                tmp = d;
                for d2 = nDims:-1:1
                    f(d2) = mod(tmp, 2);
                    tmp = floor(tmp./2);
                end
                flipIdx = find(f);
                switch length(flipIdx)
                  case 1
                    scores_caffe = net.forward(...
                        {flip(paddedInputTile, flipIdx(1))});
                    scoreTile = scoreTile + ...
                        flip(scores_caffe{1}, flipIdx(1));
                  case 2
                    scores_caffe = net.forward(...
                        {flip(flip(paddedInputTile, flipIdx(1)), flipIdx(2))});
                    scoreTile = scoreTile + ...
                        flip(flip(scores_caffe{1}, flipIdx(2)), flipIdx(1));
                  case 3
                    scores_caffe = net.forward(...
                        {flip(flip(flip(paddedInputTile, flipIdx(1)), ...
                                   flipIdx(2)), flipIdx(3))});
                    scoreTile = scoreTile + ...
                        flip(flip(flip(scores_caffe{1}, flipIdx(3)), ...
                                  flipIdx(2)), flipIdx(1));
                  otherwise
                    error('Unexpected case!');
                end
            end
            scoreTile = scoreTile / numAveragePasses;
        end

        if param.average_rotate == true
          for d = 1:numAveragePasses - 1
			  scores_caffe = net.forward(...
				  {rot90(paddedInputTile, d)});
			  scoreTile = scoreTile + ...
				  rot90(scores_caffe{1}, -d);
          end
          scoreTile = scoreTile / numAveragePasses;
        end

        if num == 1 && tileIdx == 0
            nClasses = size(scoreTile, nDims + 1);
            scores = zeros([shape nClasses nSamples]);
        end

        validReg = min(output_size, shape - tile .* output_size);
        if nDims == 2
            scores(tile(1) * output_size(1) + 1:tile(1) * output_size(1) + ...
                   validReg(1), ...
                   tile(2) * output_size(2) + 1:tile(2) * output_size(2) + ...
                   validReg(2),:,num) = ...
                scoreTile(1:validReg(1),1:validReg(2),:);
        else
            scores(tile(1) * output_size(1) + 1:tile(1) * output_size(1) + ...
                   validReg(1), ...
                   tile(2) * output_size(2) + 1:tile(2) * output_size(2) + ...
                   validReg(2), ...
                   tile(3) * output_size(3) + 1:tile(3) * output_size(3) + ...
                   validReg(3),:,num) = ...
                scoreTile(1:validReg(1),1:validReg(2),1:validReg(3),:);
        end
    end
end
caffe.reset_all();
