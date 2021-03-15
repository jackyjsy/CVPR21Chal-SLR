function [N b] = computeNormals(X, Y, Z, disparity, missingMap, superpixels, sigmaSupport, sigmaDisparity, sigmaSuperpixel)
% function [N b] = computeNormals(X, Y, Z, disparity, missingMap, superpixels, sigmaSupport, sigmaDisparity, sigmaSuperpixel)
% The normal at pixel (x,y) is N(x, y, :)'pt + b(x,y) = 0

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

	N = NaN([size(Z), 3]);
	Xf = X; Yf = Y; Zf = Z;

	X(missingMap) = NaN;
	Y(missingMap) = NaN;
	Z(missingMap) = NaN;
	
	one_Z = 1./Z; 
	X_Z = X./Z;
	Y_Z = Y./Z;
	one = Z; one(~isnan(one)) = 1;
	X_ZZ = X./(Z.*Z);
	Y_ZZ = Y./(Z.*Z);

	AtARaw = cat(3, X_Z.^2, X_Z.*Y_Z, X_Z, Y_Z.^2, Y_Z, one);
	AtbRaw = cat(3, X_ZZ, Y_ZZ, one_Z);
	AtARaw(isnan(AtARaw)) = 0;
	AtbRaw(isnan(AtbRaw)) = 0;

	AtA = filterItChopOffIS(cat(3, AtARaw, AtbRaw), superpixels, disparity, sigmaSupport, sigmaSuperpixel, sigmaDisparity);
	Atb = AtA(:, :, (size(AtARaw,3)+1):end);
	AtA = AtA(:, :, 1:size(AtARaw,3));

	[AtA_1 detAtA] = invertIt(AtA);

	N = mutiplyIt(AtA_1, Atb);
	b = N(:,:,1);
	b(:) = -detAtA;
	b = bsxfun(@rdivide, b, sqrt(sum(N.^2,3)));
	N = bsxfun(@rdivide, N, sqrt(sum(N.^2,3)));

	%Reorient the normals to point out from the scene.
	SN = sign(N(:,:,3));
	SN(SN == 0) = 1;
	N = N.*repmat(SN, [1 1 3]);
	b = b.*SN;
	sn = sign(sum(N.*cat(3, Xf, Yf, Zf),3));
	sn(isnan(sn)) = 1;
	sn(sn == 0) = 1;
	N = repmat(sn,[1 1 3]).*N;
	b = b.*sn;
end

function fFilt = filterItChopOffIS(f, sp, disparity, sigmaSupport, sigmaSuperpixel, sigmaDisparity)
	sp = double(sp);
	%sp1 is superpixel boundaries, sp2,3 is the sigma support direction, sp4 is sigma disparity..
	[sp(:,:,2) sp(:,:,3)] = ndgrid(1:size(sp,1), 1:size(sp,2));
	sp(:,:,2:3) = sp(:,:,2:3)/sigmaSupport;
	sp(:,:,1) = sp(:,:,1)./sigmaSuperpixel;
	sp(:,:,4) = disparity./sigmaDisparity;
	fFilt = jointBilateral(sp, f, 1, 1000);
end

function x = mutiplyIt(AtA_1, Atb)
	a = @(k) AtA_1(:,:,k);
	b = @(k) Atb(:,:,k);
	x1 = a(1).*b(1) + a(2).*b(2) + a(3).*b(3);
	x2 = a(2).*b(1) + a(4).*b(2) + a(5).*b(3);
	x3 = a(3).*b(1) + a(5).*b(2) + a(6).*b(3);
	x = cat(3, x1, x2, x3);
end

function [AtA_1 detAtA]= invertIt(AtA)
	a = @(k) AtA(:,:,k);

	A = a(4).*a(6) - a(5).*a(5);
	D = -(a(2).*a(6 )- a(3).*a(5));
	G = a(2).*a(5) - a(3).*a(4);
	E = a(1).*a(6) - a(3).*a(3);
	H = -(a(1).*a(5) - a(2).*a(3));
	K = a(1).*a(4) - a(2).*a(2);

	detAtA = a(1).*A + a(2).*D + a(3).*G;

	AtA_1 = cat(3, A, D, G, E, H, K);
	%AtA_1 = bsxfun(@rdivide, AtA_1, detAtA);
end

function fFilt = filterItChopOff(f, r, sp)
	f(isnan(f)) = 0;
	[H W d] = size(f);
	B = ones(2*r+1,2*r+1);
	
	minSP = ordfilt2(sp, 1, B);
	maxSP = ordfilt2(sp, numel(B), B);
	ind = find(minSP(:) ~= sp(:) | maxSP(:) ~= sp(:));
	spInd = sp; spInd(:) = 1:numel(sp);

	delta = zeros(size(f));
	delta = reshape(delta, [H*W, d]);
	f = reshape(f, [H*W, d]);

	% Calculate the delta...
	fprintf('Need to recompute for %d/%d...\n', length(ind), numel(sp));
	[I, J] = ind2sub([H W],  ind);
	for i = 1:length(ind),
		x = I(i); y = J(i);
		clipInd = spInd(max(1, x-r):min(H, x+r), max(1, y-r):min(W, y+r));
		diffInd = clipInd(sp(clipInd) ~= sp(x,y));
		delta(ind(i), :) = sum(f(diffInd, :),1);
	end

	delta = reshape(delta, [H W d]);
	f = reshape(f, [H W d]);	
	for i = 1:size(f,3),
		fFilt(:,:,i) = filter2(B, f(:,:,i));
	end
	fFilt = fFilt-delta;
end

function fFilt = filterIt(f, r)
	B = ones(2*r+1,2*r+1);
	f(isnan(f)) = 0;
	for i = 1:size(f,3),
		fFilt(:,:,i) = filter2(B, f(:,:,i));
	end
end
