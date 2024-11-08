function dlb = ISAL1bpdn_lowbo(A,b,delta,x,h,mode)
%function dlb = ISAL1bpdn_lowbo(A,b,delta,x,h,mode)
% =========================================================================
%  FILE:           ISAL1bpdn_lowbo.m
%                  ( This function is part of the ISAL1bpdn program, v1.0 )
%  AUTHOR:         Andreas M. Tillmann, TU Darmstadt, Germany
%  LAST MODIFIED:  09/28/2013
% =========================================================================
%  DESCRIPTION:    This function computes a valid lower bound for the
%                  objective function value of the L1-norm minimization
%                  problem (BPDN):  min ||x||_1  s.t.  ||Ax-b||_2 <= delta
%                  by generating a feasible dual point y, i.e., y s.t.
%                  ||A'*y||_inf <= 1, and evaluating the dual ob-
%                  jective function -b'y-delta*||y||_2. 
%                  Several constructions of y are
%                  possible to choose from; see description of 'mode'.
%
%                  Copyright (C) Andreas M. Tillmann, 2013
%
%  INPUT:          A    The constraint system matrix of the (P1) instance
%                  b    The right hand side vector of the constraints
%                  x    A vector interpreted as a primal solution; may be
%                       an arbitrary (infeas.) vector (of correct length)
%                  delta Tolerance for deviation from Ax-b (w.r.t. 2-norm)
%                  h    A (stabilized) subgradient of ||.||_1 at x
%                  mode A parameter specifying how the dual lower bound is
%                       computed. Feasible values for mode are:
%                       0 = Constant lower bound of zero
%                       1 = Dual bound -b'b/||A'b||_inf - delta*||b||_2},
%                           obtained from scaling -b to become feasible
%                           w.r.t. the dual constraint ||A'y||_inf <= 1
%                       2 = A valid dual lower bound is obtained by (ap-
%                           proximately) solving the system 
%                                       A*A'*y = -A*sign(x)
%                           and then scaling y to achieve dual feasilibity
%                       3 = A variation of 2, sometimes yielding better
%                           bounds: Instead of the subgradient h=sign(x),
%                           only the sign of the x-part corresp. to entries
%                           with abs.val. >=1 is used, while for the remai-
%                           ning entries y(i) the values x(i)<1 are used.
%                       4 = Another variation of 2 using a user-supplied 
%                           (stab. subgrad.?) h instead of sign(x).
%                       5 = A variation of 2 which uses a Camerini/Maffioli/
%                           Fratta-type stabilized subgradient. When used
%                           to estimate the duality gap in an algorithm, the
%                           function argument h should be the CMF-type stab. 
%                           subgradradient of the previous iterate 
%                       6 = A variation of 2, related to HOC:
%                           Instead of taking the full vector h, only h-
%                           entries that correspond to the support of the 
%                           thresholded vector x are used, i.e., entries of
%                           x with absolute values smaller than 1e-9 are
%                           treated as zeros. The dual variable is then 
%                           computed using only the reduced-dimensional
%                           system A(:,S)'*y=-h(S), where S is the
%                           support of the thresholded x-vector. Scaling
%                           y to be dual feasible yields the lower bound
%                           -b'*y. 
%                       [ optional argument; default is mode=5 ]
%
%  OUTPUT:         dlb  The (dual) lower bound on the (primal) objective
%                       function value, i.e., dlb <= ||x||_1 for x: Ax=b
% =========================================================================
if( nargin < 5 )
    mode = 5;
end

switch mode
    
    case 0, % 0 is always a feasible lower bound for ||x||_1;
            % x,h,A,b,delta not needed (can be empty)
        dlb = 0;

    case 1, % Simple dual lower bound; x, h not needed (can be empty).
        y = -b./norm(-A'*b,'inf'); 
        dlb = -b'*y-delta*norm(y);

    case 2, % h not needed (sign is used, h can be empty)
        y = CG( A, [], -A*sign(x), 1e-5, 3, 1 );
        y = y./norm(A'*y,'inf');
        dlb = -b'*y-delta*norm(y);

    case 3, % variation of case 2: if |x(i)|<1, init. ylb(i)=x(i);
            % h not needed (can be empty)
        idx = find(abs(x)<1); s = sign(x); s(idx) = x(idx);
        y = CG( A, [], -A*s, 1e-5, 3, 1 );
        y = y./norm(A'*y,'inf');
        dlb = -b'*y-delta*norm(y);

    case 4, % variation of case 2: h is user-provided subgradient of x
        y = CG( A, [], -A*h, 1e-5, 3, 1 );        
        y = y./norm(A'*y,'inf');
        dlb = -b'*y-delta*norm(y);

    case 5, % variation of 2 inspired by CMF-style subgradient 
            % stabilization: h should be stab. subgradient of the
            % iterate previous to x when used in an algorithm
            % (h could be just any subgradient, as well)
        s = sign(x);
        s = s + norm(s)/norm(h)*h;
        y = CG( A, [], -A*s, 1e-5, 3, 1 );
        y = y./norm(A'*y,'inf');
        dlb = -b'*y-delta*norm(y);
        % current default

    case 6, % variation of 2 using approximate supports as in HOC
            % h not needed (can be empty)
         [Sx,~] = supp(x,1e-9);
         ASx = A(:,Sx);
         w = CG( ASx, [], -ASx*h(Sx), 1e-9, 20, 1 );
         y = w./norm(A'*w,'inf');
         dlb = -b'*y-delta*norm(y);

    otherwise,
        disp('Invalid mode. Choose mode between 0 and 6.\n');
        dlb = [];
end

% negative bounds are worse than trivial bounds (case 0 or 1);
% in this case, we override dlb by 0:
dlb = max(dlb,0);
% dlb = max(dlb,b'*b/norm(A'*b,'inf')-delta*norm(b./norm(A'*b,'inf')));  % could also use case 1

if( isnan(dlb) ) % this might happen due to numerical stability problems
    dlb = 0;
    disp('Computation of dual lower bound returned NaN. dLB set to zero.');
end

if( issparse(dlb) )  % fprintf will not work with sparse input
    dlb = full(dlb);
end
