z = getImage('img_5001', 'depth'); %imread(inName);
z = double(z)./10;
C = cropCamera(getCameraParam('color'));
  
[dt.ng1, dt.ng2, dt.dg] = normalCues(z, C, 1);
