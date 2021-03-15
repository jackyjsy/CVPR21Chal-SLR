function I = isWrite(I, fName)
% function I = isWrite(I, fName)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

	typ = whos('I');
	typ = typ.class;
	f = fopen(fName, 'w');
	getMatlabStr(typ);
	fwrite(f, [1, size(I,2), size(I,1), size(I, 3), getMatlabStr(typ)], 'uint32');
	I = permute(I, [4 3 2 1]);
	fwrite(f, I(:), typ);
	fclose(f);
end

function i = getMatlabStr(str)
	switch str
		case 'single', i = 0;
		case 'double', i = 1;
		case 'uint8', i = 2;
    	case 'int8', i = 3;
    	case 'uint16', i = 4;
    	case 'int16', i = 5;
    	case 'uint32', i = 6;
    	case 'int32', i = 7;
    	case 'uint64', i = 8;
    	case 'int64', i = 9;
	end
end
