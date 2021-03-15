function I = isRead(fName)
% function I = isRead(fName)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

	f = fopen(fName, 'r');
	p = fread(f, 5, 'uint32');
	I = fread(f, prod(p(1:4)), getMatlabStr(p(5)));
	eval(sprintf('I = %s(I);', getMatlabStr(p(5))));
	I = reshape(I, [p(4) p(2) p(3)]);
	I = permute(I, [3 2 1]);
	fclose(f);
end

function str = getMatlabStr(i)
	switch i
		case 0, str = 'single';
		case 1, str = 'double';
		case 2, str = 'uint8';
    	case 3, str = 'int8';
    	case 4, str = 'uint16';
    	case 5, str = 'int16';
    	case 6, str = 'uint32';
    	case 7, str = 'int32';
    	case 8, str = 'uint64';
    	case 9, str = 'int64';
	end
end
