function C = getCameraParam(colorOrZ)
%function C = getCameraParam(colorOrZ,nyuOrX)
%gets the camera matrix, C
%Input: 
% colorOrZ 'color', 'depth'
  if(nargin < 1)
    error('Insufficient arguments\n');
  end

  switch colorOrZ,

  case 'color',
    % RGB Intrinsic Parameters
    fx_rgb = 5.1885790117450188e+02;
    fy_rgb = 5.1946961112127485e+02;
    cx_rgb = 3.2558244941119034e+02;
    cy_rgb = 2.5373616633400465e+02;
    fc_rgb = [fx_rgb,fy_rgb];
    cc_rgb = [cx_rgb,cy_rgb];
    C = [fc_rgb(1) 0 cc_rgb(1);
      0 fc_rgb(2) cc_rgb(2);
      0 0 1];

  case 'depth',
    % Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02;
    fy_d = 5.8269103270988637e+02;
    cx_d = 3.1304475870804731e+02;
    cy_d = 2.3844389626620386e+02;
    fd_d = [fx_d,fy_d];
    cd_d = [cx_d,cy_d];
    C = [fd_d(1) 0 cd_d(1);
      0 fd_d(2) cd_d(2);
      0 0 1];
  end

end
