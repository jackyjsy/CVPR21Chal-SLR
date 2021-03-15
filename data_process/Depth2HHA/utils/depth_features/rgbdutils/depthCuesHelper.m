function [mcD mcN smcN] = depthCuesHelper(pc, pcf, rr, sigmaSpace, qzc, nori, sigmaDisparity, tmpDir, binName)
%function [mcD mcN smcN] = depthCuesHelper(pc, pcf, rr, sigmaSpace, qzc, nori, sigmaDisparity)
% Output:
%   mcD oriented depth gradient based on the distance between the planes at the point being considered
%	  mcN is th eangle betweent he 2 planes
%	  smcN is the sign of the N gradient
% Input:
%	  pc is the point cloud, 
%   pcf is the filled in point cloud, 
%   rr determines the offset at which we want to look up the normals to compute the angle between them 
%   sigmaSpace is the sigma for the gaussian to estimate the normals
%   qzc is used to convert the Z value at a pixel into disparity, and to estimate the error at a particular depth.
%	  nori is the number of orientations
%	  sigmaDisparity is the value sigma for disparity in plane fitting.
%   tmpDir - directory to use to comunicate with image stack library
%   binName - name and location of the image stack binary

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

	X = pc(:,:,1); Y = pc(:,:,2); Z = pc(:,:,3);
	Xf = pcf(:,:,1); Yf = pcf(:,:,2); Zf = pcf(:,:,3);

	[h w] = size(Z);
	mcN = zeros([size(Z) nori]);
	mcD = zeros([size(Z) nori]);

	thetaQ = -([0:1:(nori-1)]./nori*pi - pi/2);
	
	N{1} = NaN([size(Z) 3]);
	N{2} = NaN([size(Z) 3]);
	D{1} = NaN([size(Z) 3]);
	D{2} = NaN([size(Z) 3]);
	cntr{1} = NaN([h w 3]);
	cntr{2} = NaN([h w 3]);
	xyz = cat(3, Xf, Yf, Zf);
	
	% qZ1 = qzc*ordfilt2(Zf,1,ones(2*rr+1)).^2;
	se = strel(ones(ceil(2*rr+1)));
	qZ2 = qzc*((-imdilate(-Zf, se)).^2);
	% assert(isequal(qZ1(rr+1:end-rr, rr+1:end-rr), qZ2(rr+1:end-rr, rr+1:end-rr)), 'Oops something is wrong with dilation as a substitue for max filter.');
	qZ = qZ2;

	for k = 1:nori,

		% For computing the signal at different orientations
		[pos(:,:,1) pos(:,:,2)] = ndgrid(1:size(Z,1), 1:size(Z,2));
		[theta, r] = cart2pol(pos(:,:,1), pos(:,:,2));
		theta = theta-thetaQ(k);
		[pos(:,:,1) pos(:,:,2)] = pol2cart(theta, r);

		% Point offsets at which we want to compare the features at 
		anchorI = [-(rr+1)/2, (rr+1)/2];
		anchorJ = [0 0];
		[theta, r] = cart2pol(anchorI, anchorJ);
		theta = theta+thetaQ(k);
		[anchorI, anchorJ] = pol2cart(theta, r);
		Rpad = ceil(max(abs([anchorI, anchorJ])))+1;

		pos(:,:,2) = pos(:,:,2)/2; %sqrt(2);

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
    
    tt = tic();
		AtA = filterItChopOffIS(cat(3, AtARaw, AtbRaw), pos, Zf, qzc, sigmaDisparity, sigmaSpace);	
    % fprintf('Time for filterItChopOffIS: %0.3f\n', toc(tt));
		Atb = AtA(:, :, (size(AtARaw,3)+1):end);
		AtA = AtA(:, :, 1:size(AtARaw,3));

		[AtA_1 detAtA] = invertIt(AtA);

		Nboth = mutiplyIt(AtA_1, Atb);
		filteredOne = AtA(:,:,end);

		% to ignore places with very few points.
		badPts = (filteredOne./max(filteredOne(:))) < 0.25;
		Nboth(repmat(badPts, [1 1 3])) = NaN;

		bboth = Nboth(:,:,1);
		bboth(:) = -detAtA;
		bboth = bsxfun(@rdivide, bboth, sqrt(sum(Nboth.^2,3)));
		Nboth = bsxfun(@rdivide, Nboth, sqrt(sum(Nboth.^2,3)));
		%Dboth  = xyz - bsxfun(@times, (sum(Nboth.*xyz, 3) + b), Nboth);
	
		Nboth = padarray(Nboth, [Rpad Rpad 0], 'replicate', 'both'); 
		bboth = padarray(bboth, [Rpad Rpad 0], 'replicate', 'both'); 
		XYZ = padarray(xyz, [Rpad Rpad 0], 'replicate', 'both'); 
	
		stI = round(Rpad+1+anchorI(1));
		enI = stI+(h-1);
		stJ = round(Rpad+1+anchorJ(1));
		enJ = stJ+(w-1);
		N{1} = Nboth(stI:enI, stJ:enJ, :);
		b{1} = bboth(stI:enI, stJ:enJ, :);
		D{1} = xyz - bsxfun(@times, (sum(N{1}.*xyz, 3) + b{1}), N{1});
		cntr{1} = XYZ(stI:enI, stJ:enJ, :);
		
		stI = round(Rpad+1+anchorI(2));
		enI = stI+(h-1);
		stJ = round(Rpad+1+anchorJ(2));
		enJ = stJ+(w-1);
		N{2} = Nboth(stI:enI, stJ:enJ, :);
		b{2} = bboth(stI:enI, stJ:enJ, :);
		D{2} = xyz - bsxfun(@times, (sum(N{2}.*xyz, 3) + b{2}), N{2});
		cntr{2} = XYZ(stI:enI, stJ:enJ, :);

		mcN(:,:,k) = 1-abs(sum(N{1}.*N{2}, 3));
		mcD(:,:,k) = sqrt(sum((D{1} - D{2}).^2, 3));
		% fprintf('   .');
		
		% Making the normals consistent.
		for i = 1:2,
			N{i} = N{i}.*repmat(sign(N{i}(:,:,3)),[1 1 3]);
			sn = sign(sum(N{i}.*cat(3, Xf, Yf, Zf),3));
			sn(isnan(sn)) = 1;
			N{i} = repmat(sn,[1 1 3]).*N{i};
		end

		% Computing the orientation for the normal discontinuity
		smcN(:,:,k) = sign(sum((N{1}-N{2}).*(cntr{1}-cntr{2}),3));
		mcD(:,:,k) = mcD(:,:,k).*(mcD(:,:,k) > qZ*1.05);	
		
		% fprintf('%d ',k);
	end

	fprintf('\n');
	mcN(isnan(mcN)) = 0;
	mcD(isnan(mcD)) = 0;
end

function fFilt = filterItChopOffIS(f, pos, Zf, qzc, sigmaDisparity, r)
	sp(:,:,1:2) = pos;
	sp(:,:,1:2) = sp(:,:,1:2)/r;
	sp(:,:,4) = (1./(qzc*Zf))/sigmaDisparity;

  % tt = tic();
	fFilt = jointBilateral(sp, f, 1, 1000);
  % fprintf('jointBilateral filtering matlab time: %0.2f\n', toc(tt));
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


