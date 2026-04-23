%% N=5：先筛"最大PSR"，再按循环移位+常数倍合并
N = 5;
roots = coprimes_to(N, 1:N-1);
gals  = coprimes_to(N, 1:N-1);
gbs   = 0:N-1;

paramList = [];  PSR = [];  Y = {};
k = 0:N-1;

for u0 = roots
  for u1 = roots
    for ga = gals
      for gc = gals
        a = mod(u0*ga^2 - u1*gc^2, N);
        if gcd(a, N) ~= 1, continue; end
        for gb = gbs
          for gd = gbs
            y = spi_zc_seq(N, u0, u1, ga, gb, gc, gd);
            a_int = u0*(ga^2) - u1*(gc^2);
            b_int = u0*(ga + 2*ga*gb) - u1*(gc + 2*gc*gd);
            gamma = exp(1j*pi/N * (u0*(gb^2 + gb) - u1*(gd^2 + gd)));
            S = sum( exp(1j*pi/N * (a_int*(k.^2) + b_int*k)) );
            PSR(end+1,1) = abs(N + gamma*S)/sqrt(N);
            paramList(end+1,:) = [u0 ga gb u1 gc gd];
            Y{end+1} = y;
          end
        end
      end
    end
  end
end

maxPSR = max(PSR);
idxMax = find(abs(PSR - maxPSR) <= 1e-12);
Pmax = paramList(idxMax,:);  Ymax = Y(idxMax);
fprintf('N=%d: 最大 PSR = %.12f，达到最大值的参数条数 = %d\n', N, maxPSR, size(Pmax,1));

% 分组：循环移位 + 常数倍
assigned = false(numel(Ymax),1); groups = {};
for i = 1:numel(Ymax)
  if assigned(i), continue; end
  grp = i; assigned(i) = true;
  for j = i+1:numel(Ymax)
    if assigned(j), continue; end
    if is_equiv_shift_scale(Ymax{i}, Ymax{j})
      grp(end+1) = j; assigned(j) = true; %#ok<AGROW>
    end
  end
  groups{end+1} = grp; %#ok<AGROW>
end

% 仅输出组数量
fprintf('%d\n', numel(groups));

% fprintf('等价类（唯一序列）数量 = %d\n\n', numel(groups));
% for g = 1:numel(groups)
%   rows = groups{g}(:);
%   fprintf('组 %2d（大小 %d）:\n', g, numel(rows));
%   disp([rows, Pmax(rows,:)])   % 列: [索引 u0 ga gb u1 gc gd]
% end

function y = spi_zc_seq(N, u0, u1, ga, gb, gc, gd)
  k = 0:N-1;
  x0 = exp( -1j*pi/N * u0 * (k.*(k+1)) );
  x1 = exp( -1j*pi/N * u1 * (k.*(k+1)) );
  g0 = mod(ga*(0:N-1) + gb, N) + 1;
  g1 = mod(gc*(0:N-1) + gd, N) + 1;
  y  = x0(g0) + x1(g1);
end

function tf = is_equiv_shift_scale(y1, y2)
  N  = numel(y1);
  y1 = y1(:); n1 = norm(y1); tf = false;
  for d = 0:N-1
    z = circshift(y2, [0,d]); z = z(:);
    c = (z' * y1) / (z' * z);
    if norm(y1 - c*z)/n1 <= 1e-12, tf = true; 
        return;
    end
  end
end

function out = coprimes_to(N, arr)
  out = arr(arrayfun(@(x) gcd(x, N)==1, arr));
end
