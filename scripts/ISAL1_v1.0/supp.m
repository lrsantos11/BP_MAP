function [S,sizeS] = supp(x,tol)
%function [S,sizeS] = supp(x,tol)
% =========================================================================
%  FILE:           supp.m 
%  AUTHOR:         Andreas M. Tillmann, TU Darmstadt, Germany             
%  LAST MODIFIED:  09/28/2013                                   
% =========================================================================
%  DESCRIPTION:    This function calculates the (approximate) support of  
%                  a given vector.                                        
%                                                                         
%  INPUT:          x       The vector to be considered                    
%                  tol     A nonnegative tolerance/threshold: entries of x 
%                          with abs. value below tol are interpreted as
%                          zero and not counted as part of the support
%                          [ optional; default is tol=0 ]     
%
%  OUTPUT:         S       The support itself, i.e., the index set of     
%                          nonzero entries in x (w.r.t. tolerance)
%                  sizeS   The size of the (approx.) support                 
% =========================================================================
% error(nargchk(1, 2, nargin, 'struct')); % check number of arguments
if( nargin < 2 )
    tol = 0;
end
S = find( abs(x) > tol ); % support w.r.t. (nonnegative) tolerance
if( nargout > 1 )
    sizeS = numel(S);     % (approx.) support size ||x||_0 := |S|
end
