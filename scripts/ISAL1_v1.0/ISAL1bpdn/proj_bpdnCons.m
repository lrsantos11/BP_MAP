function sol = proj_bpdnCons(x, A, At, b, delta, nu, useFISTA, tol, maxit, tight)
%function sol = proj_bpdnCons(x, A, At, b, delta, nu, useFISTA, tol, maxit, tight)
% ================================================================================================
% This file is a simplified version of the proj_b2.m function from the UnLocBox package, 
% see    http://unlocbox.sourceforge.net/    for the original file by G. Puy and N. Perraudin.
% Various aspects were removed because they are not needed for the application within ISAL1bpdn,
% moreover, the method now uses nu/2 instead of nu (which still allows for convergence of ISTA
% and at least empirically also increases convergence speed within FISTA).
% [ Original header, in particular containing author information, follows... ]
% --- last modified: 09/28/2013, Andreas M. Tillmann, TU Darmstadt, Germany
% ================================================================================================
%
% Projection onto a Basis Pursuit Denoising constraint ||Ax-b||_2 <= delta
%
%   Usage:  sol=proj_b2(x, ~, param)
%           [sol, infos]=proj_b2(x, ~, param)
%
%   Input parameters:
%         x     : Input signal.
%         param : Structure of optional parameters.
%   Output parameters:
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   proj_bpdnCons(x,~,param) solves:
%
%      sol = argmin_{z} ||x - z||_2^2   s.t.  ||A z - b||_2 <= delta
%
%
%   Remark: the projection is the proximal operator of the indicator 
%   function of y - A z||_2 < epsilon. So it can be written:
%      prox_{f, gamma }(x)    where     f= i_c(||A z - b||_2 <= delta)
%
%
% INPUT PARAMETER:
%
%    x : point to be projected
%
%    A : Forward operator / matrix, given as function handle
%
%    At : Adjoint operator / transposed matrix, given as function handle
%
%    delta : bound (on the noise level) from the BPDN constraint
%
%    nu : bound on the squared norm of the operator A, i.e. 
%       ||A x||^2 <= nu * ||x||^2 
%
%    useFISTA : TRUE if the problem shall be solved using 'FISTA';
%       otherwise, ISTA is used (default: true)
%
%    tol : tolerance for the projection onto the L2 ball  (default: 1e-6). 
%       The algorithm stops if   
%       delta/(1-tol) <= ||A z - b||_2 <= delta/(1+tol)
%
%    maxit : max. number of iterations (default: 200)
%
%    tight : TRUE if A is a tight frame (default: false) 
%
% OUTPUT:
%
%    sol : the (approximate) solution to the projection problem for given x
%
% ================================================== 1:1 from original: ===
%
%   References:
%     M. Fadili and J. Starck. Monotone operator splitting for optimization
%     problems in sparse recovery. In Image Processing (ICIP), 2009 16th 
%     IEEE International Conference on, pages 1461-1464. IEEE, 2009.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/proj_b2.php
%
% Copyright (C) 2012-2013 Nathanael Perraudin.
% This file is part of LTFAT version 1.1.97
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%
% Author: Gilles Puy, Nathanael Perraudin
% Date: Feb 20, 2013
%
% =========================================================================
% Set defaults for optional input arguments:
if ~exist('useFISTA', 'var'), useFISTA = true; end
if ~exist('tol', 'var'), tol = 1e-6; end
if ~exist('maxit', 'var'), maxit = 200; end
if ~exist('tight', 'var'), tight = false; end

% Useful functions for the projection
sc = @(z) z*min(delta/norm(z(:)), 1); % scaling

% Projection
if tight % TIGHT FRAME CASE
    
    temp = A(x) - b;
    sol = x + 1/nu * At(sc(temp)-temp);
        
else % NON TIGHT FRAME CASE
    
    % Initializations
    sol = x; u = zeros(size(b));
    iter = 1; proceed = true;    
        
    if useFISTA
        v = u;
        told = 1;
    else
      % MODIFICATION: replace nu by nu/2, as (for ISTA) in the Fadili/Starck ref.
        nu = nu/2;
    end
    
    % Set feasibility tolerance (sol must be on the boundary of feas. set)
    delta_low = delta/(1+tol);
    delta_up = delta/(1-tol);
    
    % Check if x is already in the constraint set
    norm_res = norm(A(sol) - b, 2); % was 'fro' originally
    if norm_res <= delta_up
        proceed = false;
    end
    
    % Projection onto the BPDN constraint:
    while proceed
        
        % Residual
        res = A(sol) - b; norm_res = norm(res(:), 2);
        
        % Scaling for the projection
        res = u*nu + res; norm_proj = norm(res(:), 2);
                
        % Stopping criteria
        if( (norm_res >= delta_low && norm_res <= delta_up) || (iter >= maxit) )
            break;
        end
               
        ratio = min(1, delta/norm_proj);
            
        if useFISTA 
            t = (1+sqrt(1+4*told^2))/2;
            u = v;
            v = 1/nu * (res - res*ratio);
            u = v + (told-1)/t * (v - u);
        
            % Update number of iteration
            told = t;     
            
        else % ISTA
            u = 1/nu * (res - res*ratio);    
        end
        
        % Current estimate
        sol = x - At(u);
        
        % Update number of iteration
        iter = iter + 1;
        
    end
end

end
