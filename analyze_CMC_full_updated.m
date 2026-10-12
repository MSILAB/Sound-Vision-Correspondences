%% analyze_CMC_full.m
% 跨模态对应（CMC）完整分析管线：评分偏好 + 分类干扰 + 大五人格 + 贝叶斯建模
%
% 整合并取代 analyze_CMC_ratings.m / analyze_CMC_test.m / analyze_CMC_personality.m，
% 并将原有 R/brms 分析（层级 RT 模型、累积序数评分模型、人格回归、
% remove_unconverged 链裁剪）翻译为 MATLAB 实现：
%   - 贝叶斯估计（BEST 式 t 模型）替代逐参 t 检验：后验均值、95% HDI、
%     方向概率 P(>0)、Savage-Dickey 近似 BF10、ROPE 比例；
%   - 随机游走 Metropolis MCMC（6 链），含 R-hat 与有效样本量(ESS)诊断；
%   - removeUnconverged：与 R 版同名函数等价的链裁剪/样本调配工具，
%     附 autoConverge 自动收敛点检测（滑动窗均值稳定准则）；
%   - 人格多元回归（倾向性 ~ N+E+O+A+C，标准化），逐系数 BF10；
%   - 参与者自定义一致性（按个体评分方向翻转 congruent 编码）与
%     实验者定义编码的对比。
%
% 用法：将 CMC_rate_merged.csv、CMC_test_merged.csv 与
%       FFI_merged_dimension_scores.csv 放入本脚本同级 data 文件夹后运行。
%       若 data/ 不存在但三张合并表位于当前目录，则自动从当前目录读取。
% 依赖：Statistics and Machine Learning Toolbox
%       （ttest, tcdf, tinv, corr, ksdensity, gammaln）。
% 注意：本脚本不在图内添加图注文字；全部图注见配套 Word 文稿。
%       局部函数统一定义在文件末尾。

clear; clc
rng(20260802, 'twister');          % 固定随机种子，保证 MCMC 可复现

%% 参数
dataDir = fullfile(pwd, 'data');
if ~exist(fullfile(dataDir, 'CMC_rate_merged.csv'), 'file')
    dataDir = pwd;
end
outDir  = fullfile(pwd, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end
rtMin = 0.2; rtMax = 2.5; catchCutoff = 0.70;
traitNames = {'Neuroticism', 'Extraversion', 'Openness', ...
    'Agreeableness', 'Conscientiousness'};
pairNames = {'shape-timbre', 'shape-pitch', 'lightness-pitch', 'lightness-timbre'};
orientVec = [-1, 1, 1, -1];        % 规范配对在左(-1)/右(+1)，顺序同 pairNames
ropeRT = 10;                       % RT 效应 ROPE：±10 ms
ropeAC = 1;                        % 正确率效应 ROPE：±1 个百分点
nChain = 6; nIter = 8000; nWarm = 1500;   % MCMC 设置（t 模型）

%% ============ 第一部分：评分任务（15 对，N = 49） ============
[rateIds, RT15, pairLbl15, leftLbl, rightLbl, alpha15, gender, keyMat3] = loadRating(dataDir);
nR = numel(rateIds);
stats15 = nan(15, 7);              % M, SD, CIlo, CIhi, t, p, dz
for p = 1:15
    x = RT15(:, p);
    [~, pv, ci, st] = ttest(x, 4);
    stats15(p, :) = [mean(x), std(x), ci(1), ci(2), st.tstat, pv, (mean(x)-4)/std(x)];
end
pHolm15 = holmBonferroni(stats15(:, 6));
resRating = table(pairLbl15(:), leftLbl(:), rightLbl(:), ...
    stats15(:, 1), stats15(:, 2), stats15(:, 3), stats15(:, 4), ...
    stats15(:, 5), stats15(:, 6), pHolm15, stats15(:, 7), alpha15(:), ...
    'VariableNames', {'pair', 'left', 'right', 'M', 'SD', 'CIlo', 'CIhi', ...
    't48', 'p', 'pHolm', 'dz', 'CronbachAlpha'});
writetable(resRating, fullfile(outDir, 'T1_rating_tests.csv'));
fprintf('评分：5 对在 Holm 校正后显著；mean alpha = %.2f。\n', ...
    nnz(pHolm15 < .05), mean(alpha15));

% 重复测量 ANOVA（15 对）+ Greenhouse-Geisser 校正
[F15, pF15, epsGG] = rmAnovaGG(RT15);
dfG = epsGG * 14;
fprintf('rmANOVA: F(14, 672) = %.2f, p = %.3g, GG eps = %.2f (df1 ~ %.1f)。\n', ...
    F15, pF15, epsGG, dfG);

% 含听觉特征的组合 vs 纯视觉组合（偏好强度 |M-4|）
% 六个特征的 15 对中：8 对视觉-听觉、6 对视觉-视觉、1 对听觉-听觉。
% 旧稿中“9 对 cross-modal”实际是“9 对含至少一个听觉特征的组合”
% （包含 timbre-pitch）；这里显式按该比较复现稿件中的 1.39 vs 0.99。
hasAuditory = false(15, 1);
withinVisual = false(15, 1);
for p = 1:15
    f = strsplit(pairLbl15{p}, '-');
    hasAuditory(p) = any(ismember(f, {'timbre', 'pitch'}));
    withinVisual(p) = all(ismember(f, {'shape', 'complexity', 'hue', 'lightness'}));
end
assert(nnz(hasAuditory) == 9 && nnz(withinVisual) == 6, '特征组合分类异常。');
strAbs = abs(RT15 - 4);
audStrength = mean(strAbs(:, hasAuditory), 2);
visStrength = mean(strAbs(:, withinVisual), 2);
[~, pvCW, ~, stCW] = ttest(audStrength, visStrength);
dzCW = mean(audStrength - visStrength) / std(audStrength - visStrength);
fprintf('含听觉特征 vs 纯视觉偏好强度：M=%.2f vs %.2f, t(%d)=%.2f, p=%.3g, dz=%.2f。\n', ...
    mean(audStrength), mean(visStrength), nR-1, stCW.tstat, pvCW, dzCW);

% TOST 等价检验（未显著的对，dz = ±0.4）
nonsig = find(pHolm15 >= .05);
tostTbl = nan(numel(nonsig), 5);   % 90% CI, boundLo/Hi, TOST p
decision = strings(numel(nonsig), 1);
for i = 1:numel(nonsig)
    x = RT15(:, nonsig(i));
    [tostTbl(i, 5), bnd] = tostP(x, 4, 0.4*std(x));
    se = std(x) / sqrt(numel(x));
    hw90 = tinv(.95, numel(x)-1) * se;
    tostTbl(i, 1:2) = [mean(x)-hw90, mean(x)+hw90];
    tostTbl(i, 3:4) = bnd;
    if tostTbl(i, 5) < .05, decision(i) = "equivalent"; else, decision(i) = "inconclusive"; end
end
resTost = table(pairLbl15(nonsig), tostTbl(:, 1), tostTbl(:, 2), ...
    tostTbl(:, 3), tostTbl(:, 4), tostTbl(:, 5), decision, ...
    'VariableNames', {'pair', 'CI90lo', 'CI90hi', 'boundLo', 'boundHi', 'TOSTp', 'decision'});
writetable(resTost, fullfile(outDir, 'T2_rating_TOST.csv'));

% 性别差异（探索性）
male = strcmp(gender, 'male') | strcmp(gender, 'm') | strcmp(gender, 'M') | strcmp(gender, '1');
pG = nan(15, 1);
for p = 1:15
    [~, pG(p)] = ttest2(RT15(male, p), RT15(~male, p));
end
fprintf('性别：最小未校正 p = %.3f（Holm 后均不显著）。\n', min(pG));

% 4 个跨模态规范偏好（正值 = 偏好规范方向），供相关与贝叶斯分析
prefCanon = nan(nR, 4);
for p = 1:4
    prefCanon(:, p) = (RT15(:, strcmp(pairLbl15, pairNames{p})) - 4) * orientVec(p);
end
prefMean = mean(prefCanon, 2);

%% ============ 第二部分：分类任务（N = 47） ============
[testIds, rtEff, acEff, main, exclIds] = loadTest(dataDir, pairNames, rtMin, rtMax, catchCutoff);
nT = numel(testIds);
statsCls = nan(4, 12);             % RT: M CIlo CIhi t p dz | ACC: same
for p = 1:4
    [~, pv, ci, st] = ttest(rtEff(:, p), 0);
    statsCls(p, 1:6) = [mean(rtEff(:, p)), ci(1), ci(2), st.tstat, pv, mean(rtEff(:, p))/std(rtEff(:, p))];
    [~, pv2, ci2, st2] = ttest(acEff(:, p), 0);
    statsCls(p, 7:12) = [mean(acEff(:, p)), ci2(1), ci2(2), st2.tstat, pv2, mean(acEff(:, p))/std(acEff(:, p))];
end
pHolmRT = holmBonferroni(statsCls(:, 5));
pHolmAC = holmBonferroni(statsCls(:, 11));
tostRT = nan(4,1); tostAC = nan(4,1);
for p = 1:4
    tostRT(p) = tostP(rtEff(:,p), 0, 0.4*std(rtEff(:,p)));
    tostAC(p) = tostP(acEff(:,p), 0, 0.4*std(acEff(:,p)));
end
resCls = table(pairNames(:), statsCls(:, 1), statsCls(:, 2), statsCls(:, 3), ...
    statsCls(:, 4), pHolmRT, statsCls(:, 6), tostRT, statsCls(:, 7), statsCls(:, 8), ...
    statsCls(:, 9), statsCls(:, 10), pHolmAC, statsCls(:, 12), tostAC, ...
    'VariableNames', {'pair', 'dRTms', 'CIlo', 'CIhi', 't46', 'pHolm_RT', 'dz_RT', 'TOSTp_RT', ...
    'dAccpp', 'CIlo_A', 'CIhi_A', 't46_A', 'pHolm_ACC', 'dz_ACC', 'TOSTp_ACC'});
writetable(resCls, fullfile(outDir, 'T3_classification_effects.csv'));
fprintf('分类：剔除 catch<.70 被试 %s；仅 lightness-timbre 正确率效应校正后显著。\n', mat2str(exclIds));

% 总体 RT 效应与 TOST（dz = ±0.4）
ovRT = mean(rtEff, 2);
[~, pvOv, ciOv, stOv] = ttest(ovRT, 0);
[tostOv, ~] = tostP(ovRT, 0, 0.4*std(ovRT));
fprintf('总体 RT 效应 = %+.1f ms, t(%d) = %.2f, p = %.3f; TOST p = %.3f。\n', ...
    mean(ovRT), nT-1, stOv.tstat, pvOv, tostOv);

% 任务模态分解（视觉任务 vs 听觉任务）
main.taskmod = repmat({''}, height(main), 1);
main.taskmod(ismember(main.task, {'shape', 'lightness'})) = {'visual'};
main.taskmod(ismember(main.task, {'pitch', 'timbre'})) = {'auditory'};
dirRes = cell(8, 7); r = 0;
for p = 1:4
    for tm = 0:1
        tmLbl = 'visual'; if tm, tmLbl = 'auditory'; end
        r = r + 1;
        eRT = nan(nT, 1); eAC = nan(nT, 1);
        for s = 1:nT
            idx = main.id == testIds(s) & strcmp(main.pairLbl, pairNames{p}) & strcmp(main.taskmod, tmLbl);
            eAC(s) = (mean(main.score(idx & main.isCon), 'omitnan') - ...
                      mean(main.score(idx & ~main.isCon), 'omitnan')) * 100;
            ok = idx & main.score == 1 & main.rt > rtMin & main.rt < rtMax;
            eRT(s) = (mean(main.rt(ok & ~main.isCon), 'omitnan') - ...
                      mean(main.rt(ok & main.isCon), 'omitnan')) * 1000;
        end
        [~, p1, ~, s1] = ttest(eRT, 0); [~, p2, ~, s2] = ttest(eAC, 0);
        dirRes(r, :) = {pairNames{p}, tmLbl, mean(eRT), s1.tstat, p1, mean(eAC), p2};
    end
end
writetable(cell2table(dirRes, 'VariableNames', ...
    {'pair', 'taskmod', 'dRTms', 't', 'p', 'dAccpp', 'p_A'}), ...
    fullfile(outDir, 'S1_classification_by_taskmodality.csv'));

% 评分偏好 × 干扰效应 跨测量相关
xr = nan(4, 4);                    % r_RT, p_RT, r_AC, p_AC per pair
for p = 1:4
    [ir, ia] = ismember(testIds, rateIds);
    y1 = prefCanon(ia(ir), p);
    [xr(p, 1), xr(p, 2)] = corr(y1, rtEff(ir, p));
    [xr(p, 3), xr(p, 4)] = corr(y1, acEff(ir, p));
end
fprintf('跨测量相关：|r| <= %.2f，均不显著。\n', max(abs(xr(:, [1 3])), [], 'all'));

% 参与者自定义一致性（按个体评分方向翻转）
[rtEffP, acEffP] = participantDefined(main, testIds, rateIds, prefCanon, pairNames, rtMin, rtMax);
pdefRes = nan(4, 6);
for p = 1:4
    [~, p1, ~, s1] = ttest(rtEffP(:, p), 0);
    [~, p2, ~, s2] = ttest(acEffP(:, p), 0);
    pdefRes(p, :) = [mean(rtEffP(:, p)), s1.tstat, p1, mean(acEffP(:, p)), s2.tstat, p2];
end
writetable(array2table(pdefRes, 'VariableNames', ...
    {'dRTms', 't', 'p', 'dAccpp', 't_A', 'p_A'}, 'RowNames', pairNames), ...
    fullfile(outDir, 'S2_participant_defined_congruency.csv'));

%% ============ 第三部分：大五人格相关 ============
[ffiIds, ffiScores] = readFFI(dataDir, traitNames);
outNames = [strcat('Rating: ', pairNames), {'Rating: mean of four'}, ...
    strcat('dRT: ', pairNames), strcat('dAcc: ', pairNames)];
outcomeIds = [repmat({rateIds}, 1, 5), repmat({testIds}, 1, 8)];
outcomeMat = [prefCanon, prefMean, rtEff, acEff];
nOut = numel(outNames);
R = nan(nOut, 5); pV = nan(nOut, 5); pH = nan(nOut, 5); Ns = nan(nOut, 1);
for o = 1:nOut
    [cmn, ia, ib] = intersect(outcomeIds{o}, ffiIds);
    Ns(o) = numel(cmn);
    y = outcomeMat(ia, o); X = ffiScores(ib, :);
    for t = 1:5
        ok = ~isnan(y) & ~isnan(X(:, t));
        [R(o, t), pV(o, t)] = corr(X(ok, t), y(ok));
    end
    pH(o, :) = holmBonferroni(pV(o, :));
end
resPers = array2table([Ns, R, pV, pH], 'VariableNames', ...
    [{'N'}, strcat('r_', traitNames), strcat('p_', traitNames), strcat('pHolm_', traitNames)]);
resPers.outcome = outNames(:);
resPers = movevars(resPers, 'outcome', 'Before', 1);
writetable(resPers, fullfile(outDir, 'T4_personality_correlations.csv'));
fprintf('人格：无相关在 Holm 校正后显著（最小校正 p = %.3f）。\n', min(pH(:)));

%% ============ 第四部分：贝叶斯建模（MCMC） ============
% BEST 式 t 模型：y/sd(y) ~ t(mu, sigma, nu)，mu ~ N(0,1)，
% log sigma 平坦先验，nu-1 ~ Exp(1/29)。Savage-Dickey 近似 BF10（mu=0）。
bNames = [strcat('dRT: ', pairNames), strcat('dAcc: ', pairNames), ...
    strcat('Rating: ', pairNames), {'Rating: mean of four'}];
nPad = max(nR, nT);
bData = nan(nPad, 13);                     % 行数补齐，缺失为 NaN（贝叶斯模型只用样本向量）
bData(1:nT, 1:8) = [rtEff, acEff];
bData(1:nR, 9:13) = [prefCanon, prefMean];
bRope = [repmat(ropeRT, 1, 4), repmat(ropeAC, 1, 4), nan(1, 5)];
nB = numel(bNames);
Bres = nan(nB, 8);                 % mean, HDIlo, HDIhi, P(>0), BF10, Rhat, ESS, ROPEin
chainsMu = cell(nB, 1);            % 每模型 mu 链（原量纲）
for b = 1:nB
    y = bData(:, b); y = y(~isnan(y));
    sd = std(y);
    chains = mcmcT(y / sd, nIter, nChain);           % [chain x iter x 3]
    mu = chains(:, nWarm+1:end, 1);                   % 标准化尺度
    [rh, ess] = rhatEss(mu);
    post = mu(:) * sd;                                % 回到原量纲
    [lo, hi] = hdi95(post);
    bf = savageDickey(mu(:));                         % z 尺度，N(0,1) 先验
    if isnan(bRope(b)), ropeIn = nan;
    else, ropeIn = mean(post > -bRope(b) & post < bRope(b)); end
    Bres(b, :) = [mean(post), lo, hi, mean(post > 0), bf, rh, ess, ropeIn];
    chainsMu{b} = mu * sd;
    fprintf('  %-22s post %+7.2f [%+7.2f, %+7.2f]  P(>0) = %.3f  BF10 = %6.2f  Rhat %.3f  ESS %5.0f\n', ...
        bNames{b}, mean(post), lo, hi, mean(post > 0), bf, rh, ess);
end
resBayes = array2table(Bres, 'VariableNames', ...
    {'postMean', 'HDIlo', 'HDIhi', 'Pgt0', 'BF10', 'Rhat', 'ESS', 'ROPEin'});
resBayes.outcome = bNames(:);
resBayes = movevars(resBayes, 'outcome', 'Before', 1);
writetable(resBayes, fullfile(outDir, 'T5_bayesian_estimates.csv'));

% removeUnconverged 演示：自动检测各链收敛点，裁剪并调配样本使链等长
demoChain = chainsMu{1};                             % dRT: shape-timbre（6 x iter）
starts = autoConverge(demoChain);
[eqChains, info] = removeUnconverged(demoChain, starts);
fprintf('removeUnconverged：收敛点 = %s；调配后每链 %d 样本（调配明细 %s）。\n', ...
    mat2str(starts), size(eqChains, 2), mat2str(info));
[rhatAfter, essAfter] = rhatEss(eqChains);
fprintf('  调配后 Rhat = %.3f, ESS = %.0f。\n', rhatAfter, essAfter);

% 人格多元回归：倾向性（4 对 ΔRT 均值，标准化）~ z(N+E+O+A+C)
prop = mean(rtEff, 2);
[cmn, ia, ib] = intersect(testIds, ffiIds);
yz = zscore(prop(ia)); Xz = zscore(ffiScores(ib, :));
chainsReg = mcmcReg(Xz, yz, 12000, 4);
regRes = nan(6, 6);                                  % mean HDIlo HDIhi P(>0) BF10 Rhat
regNames = [{'Intercept'}, traitNames];
for j = 1:6
    mu = chainsReg(:, 1501:end, j);
    [rh, ~] = rhatEss(mu);
    post = mu(:); [lo, hi] = hdi95(post);
    bf = nan; if j > 1, bf = savageDickey(post); end
    regRes(j, :) = [mean(post), lo, hi, mean(post > 0), bf, rh];
    fprintf('  回归 %-16s beta = %+.2f [%+.2f, %+.2f]  P(>0) = %.3f  BF10 = %5.2f  Rhat %.3f\n', ...
        regNames{j}, mean(post), lo, hi, mean(post > 0), bf, rh);
end
resReg = array2table(regRes, 'VariableNames', ...
    {'betaMean', 'HDIlo', 'HDIhi', 'Pgt0', 'BF10', 'Rhat'}, 'RowNames', regNames);
writetable(resReg, fullfile(outDir, 'T6_personality_regression.csv'), 'WriteRowNames', true);

%% ============ 第五部分：图形（PNG 输出；图注见 Word 文稿） ============
figDir = fullfile(outDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
PLUM = [0.557 0.384 0.471]; GREEN = [0.290 0.404 0.255]; GOLD = [0.722 0.525 0.043];

% 图 1：15 对组均值
f1 = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 1100 520]);
[m15, ord15] = sort(stats15(:, 1));
sem15 = stats15(:, 2) / sqrt(nR);
bar(1:15, m15, 0.6, 'FaceColor', PLUM); hold on
errorbar(1:15, m15, sem15(ord15), 'k', 'LineStyle', 'none', 'LineWidth', 1);
yline(4, '--k', 'LineWidth', 1);
sig = find(pHolm15(ord15) < .05);
for k = 1:numel(sig)
    plot(sig(k), m15(sig(k)) + 0.35*sign(m15(sig(k))-4) + sem15(ord15(sig(k))), '*k');
end
set(gca, 'XTick', 1:15, 'XTickLabel', pairLbl15(ord15), 'XTickLabelRotation', 45);
ylabel('Mean rating (1–7)'); xlim([0.4 15.6]);
exportgraphics(f1, fullfile(figDir, 'fig1_group_ratings.png'), 'Resolution', 200);
close(f1);

% 图 2：个体均值分布
f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 1100 520]); hold on
for p = 1:15
    y = RT15(:, p);
    x = p + (rand(nR, 1) - 0.5) * 0.5;
    plot(x, y, 'o', 'MarkerSize', 4, 'Color', [PLUM 0.35]);
    plot(p, mean(y), 'd', 'MarkerFaceColor', GOLD, 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
end
yline(4, '--k');
set(gca, 'XTick', 1:15, 'XTickLabel', pairLbl15, 'XTickLabelRotation', 45);
ylabel('Individual mean rating'); xlim([0.4 15.6]);
exportgraphics(f2, fullfile(figDir, 'fig2_individual.png'), 'Resolution', 200);
close(f2);

% 图 3：一致性效应（RT 与 ACC）
f3 = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 900 420]);
subplot(1, 2, 1); hold on
bar(1:4, statsCls(:, 1), 0.6, 'FaceColor', PLUM);
errorbar(1:4, statsCls(:, 1), (statsCls(:, 1)-statsCls(:, 2)), (statsCls(:, 3)-statsCls(:, 1)), 'k', 'LineStyle', 'none');
yline(0, '--k'); set(gca, 'XTick', 1:4, 'XTickLabel', pairNames, 'XTickLabelRotation', 30);
ylabel('\DeltaRT (ms)'); title('A');
subplot(1, 2, 2); hold on
bar(1:4, statsCls(:, 7), 0.6, 'FaceColor', GREEN);
errorbar(1:4, statsCls(:, 7), (statsCls(:, 7)-statsCls(:, 8)), (statsCls(:, 9)-statsCls(:, 7)), 'k', 'LineStyle', 'none');
yline(0, '--k'); set(gca, 'XTick', 1:4, 'XTickLabel', pairNames, 'XTickLabelRotation', 30);
ylabel('\DeltaAcc (pp)'); title('B');
plot(4, statsCls(4, 7) - 0.25, '*k');
exportgraphics(f3, fullfile(figDir, 'fig3_congruency.png'), 'Resolution', 200);
close(f3);

% 图 4：ΔRT 后验密度（4 面板，HDI 加深）
f4 = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 900 620]);
for p = 1:4
    subplot(2, 2, p); hold on
    post = chainsMu{p}(:);
    [f, xs] = ksdensity(post);
    fill([xs fliplr(xs)], [f zeros(size(f))], PLUM, 'FaceAlpha', 0.30, 'EdgeColor', PLUM);
    lo = Bres(p, 2); hi = Bres(p, 3);
    inH = xs >= lo & xs <= hi;
    fill([xs(inH) fliplr(xs(inH))], [f(inH) zeros(size(xs(inH)))], PLUM, 'FaceAlpha', 0.85, 'EdgeColor', 'none');
    xline(0, '--k');
    title(strrep(pairNames{p}, '-', '–')); xlabel('\DeltaRT (ms)'); ylabel('Posterior density');
end
exportgraphics(f4, fullfile(figDir, 'fig4_posterior_rt.png'), 'Resolution', 200);
close(f4);

% 图 5：链轨迹（示例参数，含自动收敛点竖线）
f5 = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 900 420]); hold on
c6 = lines(6);
for c = 1:nChain
    plot(demoChain(c, :), 'Color', [c6(c, :) 0.7], 'LineWidth', 0.4);
    if starts(c) > 0
        xline(starts(c), ':', 'Color', c6(c, :), 'LineWidth', 1.2);
    end
end
xlabel('Iteration'); ylabel('\mu (ms)');
exportgraphics(f5, fullfile(figDir, 'fig5_traces.png'), 'Resolution', 200);
close(f5);

% 图 6：人格回归系数后验（山脊式）
f6 = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 900 480]); hold on
for j = 2:6
    post = chainsReg(:, 1501:end, j); post = post(:);
    [f, xs] = ksdensity(post);
    y0 = 7.5 - j;
    f = f / max(f) * 0.38;
    fill([xs fliplr(xs)], [y0 + f, repmat(y0, size(xs))], GREEN, 'FaceAlpha', 0.45, 'EdgeColor', GREEN);
    lo = regRes(j, 2); hi = regRes(j, 3);
    plot([lo hi], [y0 y0], 'k', 'LineWidth', 2);
    plot(regRes(j, 1), y0, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
end
xline(0, '--k');
set(gca, 'YTick', 1.5:1:5.5, 'YTickLabel', fliplr(traitNames));
xlabel('Standardized coefficient (\beta)'); ylim([1 6.2]);
exportgraphics(f6, fullfile(figDir, 'fig6_posterior_traits.png'), 'Resolution', 200);
close(f6);

fprintf('\n完成。统计结果 CSV 与图形 PNG 已写入 %s\n', outDir);

%% -----------------------------------------------------------------------
%  局部函数
%  -----------------------------------------------------------------------
function [ids, RT15, pairLbl, leftLbl, rightLbl, alpha15, gender, key3] = loadRating(dataDir)
%LOADRATING 读取合并后的 CMC_rate_merged.csv，返回 15 对个体均值、标签、信度与性别
fn = fullfile(dataDir, 'CMC_rate_merged.csv');
assert(exist(fn, 'file') == 2, '未找到 %s。', fn);
T = readtable(fn, 'TextType', 'string');
T.id = toDouble(T.id);
T.key = toDouble(T.key);
assert(numel(unique(T.id)) == 49, '评分合并表被试数不是 49。');
assert(height(T) == 49*45, '评分合并表行数应为 2205，实际 %d。', height(T));

n = height(T);
lbl = cell(n, 1);
for i = 1:n
    lbl{i} = featOrderLabel(char(T.task(i)), char(T.inducer(i)));
end
T.pairLbl = string(lbl);
pairLbl = unique(lbl, 'stable');
assert(numel(pairLbl) == 15, '评分数据含 %d 个组合（应为 15）。', numel(pairLbl));

ids = unique(T.id);
nP = numel(pairLbl);
RT15 = nan(numel(ids), nP);
leftLbl = cell(nP, 1); rightLbl = cell(nP, 1);
key3 = nan(numel(ids), 3, nP);
gender = cell(numel(ids), 1);
for si = 1:numel(ids)
    for p = 1:nP
        idx = T.id == ids(si) & T.pairLbl == string(pairLbl{p});
        k = T.key(idx);
        assert(numel(k) == 3, '被试 %g / %s 的评分重复数不是 3。', ids(si), pairLbl{p});
        RT15(si, p) = mean(k, 'omitnan');
        key3(si, :, p) = k(1:3);
    end
    ii = find(T.id == ids(si), 1);
    gender{si, 1} = char(T.gender(ii));
end
for p = 1:nP
    ii = find(T.pairLbl == string(pairLbl{p}), 1);
    leftLbl{p} = char(T.left(ii)); rightLbl{p} = char(T.right(ii));
end
alpha15 = nan(nP, 1);
for p = 1:nP
    alpha15(p) = cronbach(key3(:, :, p));
end
end

function a = cronbach(X)
%CRONBACH 3 次重复的内部一致性 alpha（标准样本方差公式）
X = X(~any(isnan(X), 2), :);
k = size(X, 2);
if size(X, 1) < 3 || k < 2, a = nan; return, end
a = k / (k - 1) * (1 - sum(var(X, 0, 1)) / var(sum(X, 2), 0));
end

function [F, pGG, epsGG] = rmAnovaGG(X)
%RMANOVAGG 单因素重复测量 ANOVA + Greenhouse-Geisser 校正
[n, k] = size(X);
gm = mean(X, 'all');
ssC = n * sum((mean(X, 1) - gm).^2);
ssE = sum((X - mean(X, 1) - mean(X, 2) + gm).^2, 'all');
F = (ssC / (k - 1)) / (ssE / ((n - 1) * (k - 1)));
S = cov(X);
C = eye(k) - ones(k) / k;          % 对条件空间做中心化投影
Sg = C * S * C;                    % GG epsilon 必须基于中心化协方差
epsGG = trace(Sg)^2 / ((k - 1) * sum(Sg.^2, 'all'));
epsGG = min(1, max(1/(k-1), epsGG));
pGG = 1 - fcdf(F, epsGG * (k - 1), epsGG * (n - 1) * (k - 1));
end

function [p, bnd] = tostP(x, center, bound)
%TOSTP 双单侧等价检验（对 center ± bound）
x = x(~isnan(x)); n = numel(x);
se = std(x) / sqrt(n);
t1 = (mean(x) - (center - bound)) / se;
t2 = ((center + bound) - mean(x)) / se;
p = max(tcdf(-t1, n - 1), tcdf(-t2, n - 1));
bnd = [center - bound, center + bound];
end

function pHolm = holmBonferroni(p)
%HOLMBONFERRONI Holm-Bonferroni 逐步校正
p = p(:);
[pSort, ord] = sort(p);
m = numel(p);
pAdj = (m:-1:1)' .* pSort;
pAdj = cummax(pAdj);
pHolm = min(1, pAdj);
pHolm(ord) = pHolm;
end

function [ids, rtEff, acEff, main, exclIds] = loadTest(dataDir, pairNames, rtMin, rtMax, catchCutoff)
%LOADTEST 读取合并后的 CMC_test_merged.csv，编码一致性并计算个体 RT/ACC 效应
fn = fullfile(dataDir, 'CMC_test_merged.csv');
assert(exist(fn, 'file') == 2, '未找到 %s。', fn);
T = readtable(fn, 'TextType', 'string');
T.id = toDouble(T.id);
assert(numel(unique(T.id)) == 49, '分类合并表被试数不是 49。');

expT = T(T.phase == "exp", :);
isCatch = toDouble(expT.catch) == 1;
catchScore = toDouble(expT.catch_score);
allIds = unique(expT.id);
catchAcc = nan(numel(allIds), 1);
for si = 1:numel(allIds)
    catchAcc(si) = mean(catchScore(expT.id == allIds(si) & isCatch), 'omitnan');
end
exclIds = allIds(catchAcc < catchCutoff);
main = expT(~isCatch & ~ismember(expT.id, exclIds), :);
main.score = toDouble(main.score);
main.rt = toDouble(main.rt);
[isCon, pairLbl] = congruentFlags(cellstr(main.task), cellstr(main.inducer), cellstr(main.CMC_code));
main.isCon = isCon; main.pairLbl = pairLbl;
ids = unique(main.id);

nPair = numel(pairNames);
rtEff = nan(numel(ids), nPair); acEff = nan(numel(ids), nPair);
for si = 1:numel(ids)
    for p = 1:nPair
        idx = main.id == ids(si) & strcmp(main.pairLbl, pairNames{p});
        assert(any(idx), '被试 %g 缺少 %s 分类试次。', ids(si), pairNames{p});
        acEff(si, p) = (mean(main.score(idx & main.isCon), 'omitnan') - ...
                        mean(main.score(idx & ~main.isCon), 'omitnan')) * 100;
        ok = idx & main.score == 1 & main.rt > rtMin & main.rt < rtMax;
        rtEff(si, p) = (mean(main.rt(ok & ~main.isCon), 'omitnan') - ...
                        mean(main.rt(ok & main.isCon), 'omitnan')) * 1000;
    end
end
end

function [rtEffP, acEffP] = participantDefined(main, testIds, rateIds, prefCanon, pairNames, rtMin, rtMax)
%PARTICIPANTDEFINED 按个体评分方向定义一致性后的 RT/ACC 效应
nPair = numel(pairNames);
rtEffP = nan(numel(testIds), nPair); acEffP = nan(numel(testIds), nPair);
for s = 1:numel(testIds)
    rIdx = find(rateIds == testIds(s));
    for p = 1:nPair
        idx = main.id == testIds(s) & strcmp(main.pairLbl, pairNames{p});
        isConP = main.isCon;
        if ~isempty(rIdx) && prefCanon(rIdx, p) < 0
            isConP = ~isConP;   % 该被试偏好反规范方向：翻转编码
        end
        acEffP(s, p) = (mean(main.score(idx & isConP), 'omitnan') - ...
                        mean(main.score(idx & ~isConP), 'omitnan')) * 100;
        ok = idx & main.score == 1 & main.rt > rtMin & main.rt < rtMax;
        rtEffP(s, p) = (mean(main.rt(ok & ~isConP), 'omitnan') - ...
                        mean(main.rt(ok & isConP), 'omitnan')) * 1000;
    end
end
end

function [ids, scores] = readFFI(dataDir, traitNames)
%READFFI 读取合并后的 FFI_merged_dimension_scores.csv
fn = fullfile(dataDir, 'FFI_merged_dimension_scores.csv');
assert(exist(fn, 'file') == 2, '未找到 %s。', fn);
T = readtable(fn, 'TextType', 'string');
ids = toDouble(T.subject_id);
assert(numel(ids) == 49 && numel(unique(ids)) == 49, 'FFI 合并表应包含 49 名唯一被试。');
scores = nan(height(T), numel(traitNames));
for j = 1:numel(traitNames)
    assert(ismember(traitNames{j}, T.Properties.VariableNames), 'FFI 缺少列 %s。', traitNames{j});
    scores(:, j) = toDouble(T.(traitNames{j}));
end
[ids, ord] = sort(ids);
scores = scores(ord, :);
end

function lbl = featOrderLabel(f1, f2)
%FEATORDERLABEL 按六特征固定顺序生成唯一组合标签
featOrder = {'shape', 'complexity', 'hue', 'lightness', 'timbre', 'pitch'};
[tf1, ia] = ismember(f1, featOrder);
[tf2, ib] = ismember(f2, featOrder);
assert(tf1 && tf2, '未知特征标签：%s / %s', f1, f2);
if ia <= ib, lbl = [f1 '-' f2]; else, lbl = [f2 '-' f1]; end
end

function [isCon, pairLbl] = congruentFlags(task, inducer, code)
%CONGRUENTFLAGS 按先验 CMC 方向编码一致性，并给出组合标签
canonical = {'smooth:low', 'pointy:high', 'smooth:sine', 'pointy:square', ...
    'dark:low', 'light:high', 'dark:sine', 'light:square'};
n = numel(code);
isCon = false(n, 1); pairLbl = cell(n, 1);
for i = 1:n
    parts = strsplit(code{i}, ':');
    isCon(i) = ismember([parts{1} ':' parts{2}], canonical) || ...
               ismember([parts{2} ':' parts{1}], canonical);
    pairLbl{i} = featOrderLabel(task{i}, inducer{i});
end
end

function v = toDouble(v)
%TODOUBLE 文本列转数值（空串 -> NaN）
if iscell(v) || isstring(v)
    v = str2double(string(v));
end
end

% --------------------------- MCMC：BEST 式 t 模型 ---------------------------
function chains = mcmcT(yz, nIter, nChain)
%MCMCT 随机游走 Metropolis：yz ~ t(mu, sigma, nu)，mu ~ N(0, 1)
% 返回 [nChain x nIter x 3]，参数为 (mu, log sigma, log(nu-1))
chains = nan(nChain, nIter, 3);
m0 = median(yz); s0 = std(yz) * 1.2 + eps;
for c = 1:nChain
    th = [m0 + randn * s0, log(s0), log(30)] + randn(1, 3) * 0.3;
    lp = logpostT(th, yz);
    scale = exp(th(2)) / 25 + 0.5;
    step = [0.5, 0.15, 0.10] * scale;
    for i = 1:nIter
        prop = th + randn(1, 3) .* step;
        lpp = logpostT(prop, yz);
        if log(rand) < lpp - lp
            th = prop; lp = lpp;
        end
        chains(c, i, :) = th;
    end
end
end

function lp = logpostT(th, yz)
%LOGPOSTT t 似然 + 先验（mu~N(0,1)；p(log sigma) 平坦；nu-1~Exp(1/29)）
mu = th(1); sig = exp(th(2)); nu = exp(th(3)) + 1;
n = numel(yz);
z = (yz - mu) / sig;
ll = n * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * log(nu * pi) - th(2)) ...
    - (nu + 1) / 2 * sum(log1p(z.^2 / nu));
lp = ll - 0.5 * mu^2 - (nu - 1) / 29;
end

% --------------------------- MCMC：多元回归 ---------------------------
function chains = mcmcReg(Xz, yz, nIter, nChain)
%MCMCREG y ~ X*b + N(0, sigma)，b ~ N(0, 1)，含截距；返回 [nChain x nIter x 7]
Xd = [ones(numel(yz), 1), Xz];
k = size(Xd, 2) + 1;
chains = nan(nChain, nIter, k);
for c = 1:nChain
    th = [randn(1, k - 1) * 0.5, log(std(yz))];
    lp = logpostReg(th, Xd, yz);
    for i = 1:nIter
        prop = th + randn(1, k) * 0.16;
        lpp = logpostReg(prop, Xd, yz);
        if log(rand) < lpp - lp
            th = prop; lp = lpp;
        end
        chains(c, i, :) = th;
    end
end
end

function lp = logpostReg(th, Xd, y)
%LOGPOSTREG 正态回归对数后验
b = th(1:end - 1)'; sig = exp(th(end));
r = y - Xd * b;
n = numel(y);
lp = -n * th(end) - 0.5 * (r' * r) / sig^2 - 0.5 * (b' * b) - th(end);
end

% --------------------------- 收敛诊断与裁剪 ---------------------------
function [rhat, ess] = rhatEss(chains)
%RHATESS Gelman-Rubin R-hat 与有效样本量（链 [C x N]）
[C, N] = size(chains);
cm = mean(chains, 2); cv = var(chains, 0, 2);
W = mean(cv); B = N * var(cm);
vh = (N - 1) / N * W + B / N;
rhat = sqrt(vh / W);
acs = 0;
for lag = 1:120
    a = 0;
    for c = 1:C
        cc = corrcoef(chains(c, 1:end - lag), chains(c, 1 + lag:end));
        a = a + cc(1, 2);
    end
    a = a / C;
    if a < 0.05, break, end
    acs = acs + a;
end
ess = C * N / (1 + 2 * acs);
end

function starts = autoConverge(chains, W, tol)
%AUTOCONVERGE 自动检测各链收敛点：滑动窗均值稳定准则
% 返回每条链需丢弃的初始样本数（0 起）
if nargin < 2, W = 400; end
if nargin < 3, tol = 0.15; end
[C, N] = size(chains);
starts = zeros(C, 1);
for c = 1:C
    x = chains(c, :);
    fm = mean(x(end - 999:end)); sd = std(x(end - 999:end));
    t = 1;
    while t <= N - 3 * W
        ok = true;
        for k = 0:2
            seg = mean(x(t + k * W:t + (k + 1) * W - 1));
            if abs(seg - fm) >= tol * sd, ok = false; break, end
        end
        if ok, break, end
        t = t + 50;
    end
    starts(c) = min(t - 1, N - 1);
end
end

function [eq, nDonate] = removeUnconverged(chains, starts)
%REMOVEUNCONVERGED 与 R 版 remove_unconverged 等价的链裁剪/调配
% chains: [C x N]；starts: 每链需跳过的初始样本数
% 先按收敛点裁剪，再按比例在链间调配样本使各链长度统一为 n_goal：
% 富裕链把多余样本捐入样本池，短缺链从池尾领取（处理顺序按 starts 升序）。
[C, N] = size(chains);
nGoal = floor(mean(N - starts));
kept = cell(C, 1);
for c = 1:C
    kept{c} = chains(c, starts(c) + 1:end);
end
pool = [];
nDonate = zeros(C, 1);
[~, order] = sort(starts);
for ci = order(:).'
    S = kept{ci};
    nDonate(ci) = numel(S) - nGoal;
    if nDonate(ci) > 0
        pool = [pool, S(end - nDonate(ci) + 1:end)]; %#ok<AGROW>
        S = S(1:end - nDonate(ci));
    elseif nDonate(ci) < 0
        need = -nDonate(ci);
        take = min(need, numel(pool));
        S = [S, pool(end - take + 1:end)]; %#ok<AGROW>
        pool = pool(1:end - take);
    end
    kept{ci} = S(1:nGoal);
end
eq = nan(C, nGoal);
for c = 1:C
    eq(c, :) = kept{c}(1:nGoal);
end
end

% --------------------------- 后验摘要 ---------------------------
function [lo, hi] = hdi95(x)
%HDI95 最小宽度 95% 最高密度区间
x = sort(x(:)); n = numel(x);
k = floor(0.95 * n);
w = x(k + 1:end) - x(1:end - k - 1);
[~, i] = min(w);
lo = x(i); hi = x(i + k);
end

function bf10 = savageDickey(post)
%SAVAGEDICKEY mu = 0 处的 Savage-Dickey 密度比（先验 N(0,1)）
dPost = ksdensity(post, 0);
dPrior = normpdf(0, 0, 1);
bf10 = dPrior / dPost;
end
