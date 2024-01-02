using Random
using CSV
using DataFrames

# ランダムに3つのPCMを選択する関数
function random_select_pcms(pcms, count)
    return pcms[shuffle(1:length(pcms))[1:count]]
end

# 結果を保存するためのデータフレームを初期化
results = DataFrame(
    W_PerfectIncorporation = String[],
    d_PerfectIncorporation = Float64[],
    W_tPerfectIncorporation2 = String[],
    d_tPerfectIncorporation2 = Float64[],
    s_tPerfectIncorporation2 = Float64[],
    # ここに他の手法の結果用の列を追加
)

# 1000回の実験を実行
for _ in 1:1000
    selected_pcms = random_select_pcms(pcms, 3)

    # PerfectIncorporationの計算
    PerfectIncorporation = solveonePerfectIncorporationLP(selected_pcms)
    tPerfectIncorporation2 = solvetPerfectIncorporationLP2(selected_pcms)

    # CommonGroundの計算
    CommonGround = solveoneCommonGroundLP(selected_pcms)
    tCommonGround2 = solvetCommonGroundLP2(selected_pcms)

    # PartialIncorporationの計算
    PartialIncorporation = solveonePartialIncorporationLP(selected_pcms)
    tPartialIncorporation2 = solvetPartialIncorporationLP2(selected_pcms)

    # 結果をデータフレームに追加
    push!(results, (
        W_PerfectIncorporation = join(PerfectIncorporation.W, ","),
        d_PerfectIncorporation = PerfectIncorporation.optimalValue,
        W_tPerfectIncorporation2 = join(tPerfectIncorporation2.W, ","),
        d_tPerfectIncorporation2 = tPerfectIncorporation2.optimalValue,
        s_tPerfectIncorporation2 = tPerfectIncorporation2.s,
        W_CommonGround = join(CommonGround.W, ","),
        d_CommonGround = CommonGround.optimalValue,
        W_tCommonGround2 = join(tCommonGround2.W, ","),
        d_tCommonGround2 = tCommonGround2.optimalValue,
        s_tCommonGround2 = tCommonGround2.s,
        W_PartialIncorporation = join(PartialIncorporation.W, ","),
        d_PartialIncorporation = PartialIncorporation.optimalValue,
        W_tPartialIncorporation2 = join(tPartialIncorporation2.W, ","),
        d_tPartialIncorporation2 = tPartialIncorporation2.optimalValue,
        s_tPartialIncorporation2 = tPartialIncorporation2.s,
    ))
end

# 結果をCSVファイルに保存
CSV.write("results.csv", results)
