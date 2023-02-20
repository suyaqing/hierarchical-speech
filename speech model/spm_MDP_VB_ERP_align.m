function [u, v, w] = spm_MDP_VB_ERP_align(x, y, d, f)
% u: realigned upper level posterior expectation
% v: realigned lower level posterior
% w: upper level without the gradient descend, i.e. the GD steps are
% replaced by the last step of the previous update

Ne = size(x, 1);
    
% synchronise responses
%----------------------------------------------------------------------
u = {};
w = {};
if f==1 % second input is the unprocessed response y{}
    v   = {};
else 
    v = y;
    Nsub = size(v, 1)/Ne;
end
uu  = spm_cat(x(1,:));
for k = 1:Ne

    % low-level
    %------------------------------------------------------------------
    if f==1
        v{end + 1,1} = spm_cat(y{k}); % new epoch
        if k > 1
            prev = spm_cat(x(k-1,:));
            u{k, 1} = ones((size(v{end,:},1)-size(spm_cat(x(k-1,:)),1)),1) ...
                *prev(end, :); % start from repeating the previous epoch
            w{k, 1} = u{k, 1};
            filler = prev(end, :);
        else
%             u{1,1} = ones(size(v{end,:},1),1)*uu(1,:);
            u{1,1} = ones(size(v{end,:},1),1)*(d'/sum(d));
            w{1, 1} = u{1, 1};
            filler = (d'/sum(d));
        end
    else
        vv = spm_cat(v((k-1)*Nsub+1:k*Nsub));
        if k > 1
            prev = spm_cat(x(k-1,:));
            u{k, 1} = ones((size(vv,1)-size(spm_cat(x(k-1,:)),1)),1) ...
                *prev(end, :); % start from repeating the previous epoch
            w{k, 1} = u{k, 1};
            filler = prev(end, :);
        else
%             u{1,1} = ones(size(vv,1),1)*uu(1,:);
            u{1, 1} = ones(size(vv,1),1)*(d'/sum(d));
            w{1, 1} = u{1, 1};
            filler = (d'/sum(d));
        end
    end


    % high-level
    %------------------------------------------------------------------
    u{k, 1} = [u{k, 1}; spm_cat(x(k, :))]; % the real epoch at the end of the epoch
    w{k, 1} = [w{k, 1}; ones(size(spm_cat(x(k, :)),1), 1)*filler];
%     if k==Ne
%         v{end + 1,1} = ones(size(spm_cat(x(k,:)),1),1)*v{end,1}(end,:); % repeat
%     end


end

