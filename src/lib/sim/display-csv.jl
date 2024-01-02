using Random
using CSV
using DataFrames

# ランダムに5つのPCMを選択する関数
function random_select_pcms(pcms, count)
    return pcms[shuffle(1:length(pcms))[1:count]]
end

# 結果を保存するためのデータフレームを初期化
results = DataFrame(
    #W_PerfectIncorporation = String[],
    d_PerfectIncorporation = Float64[],
    #W_tPerfectIncorporation2 = String[],
    d_tPerfectIncorporation2 = Float64[],
    #W_CommonGround = String[],
    d_CommonGround = Float64[],
    #W_tCommonGround2 = String[],
    d_tCommonGround2 = Float64[],
    #W_PartialIncorporation = String[],
    d_PartialIncorporation = Float64[],
    #W_tPartialIncorporation2 = String[],
    d_tPartialIncorporation2 = Float64[]
)

# 1000回の実験を実行
for _ in 1:1000
    selected_pcms = random_select_pcms(pcms, 5)

    # PerfectIncorporationの計算
    PerfectIncorporation = solveonePerfectIncorporationLP(selected_pcms)
    if PerfectIncorporation === nothing
        PerfectIncorporation = (W = "", optimalValue = -100)
    end
    tPerfectIncorporation2 = solvetPerfectIncorporationLP2(selected_pcms)
    if tPerfectIncorporation2 === nothing
        tPerfectIncorporation2 = (W = "", optimalValue = -100)
    end

    # CommonGroundの計算
    CommonGround = solveoneCommonGroundLP(selected_pcms)
    if CommonGround === nothing
        CommonGround = (W = "", optimalValue = -100)
    end
    tCommonGround2 = solvetCommonGroundLP2(selected_pcms)
    if tCommonGround2 === nothing
        tCommonGround2 = (W = "", optimalValue = -100)
    end

    # PartialIncorporationの計算
    PartialIncorporation = solveonePartialIncorporationLP(selected_pcms)
    if PartialIncorporation === nothing
        PartialIncorporation = (W = "", optimalValue = -100)
    end
    tPartialIncorporation2 = solvetPartialIncorporationLP2(selected_pcms)
    if tPartialIncorporation2 === nothing
        tPartialIncorporation2 = (W = "", optimalValue = -100)
    end

    # 結果をデータフレームに追加
    push!(results, (
        #W_PerfectIncorporation = join(PerfectIncorporation.W, ","),
        d_PerfectIncorporation = PerfectIncorporation.optimalValue,
        #W_tPerfectIncorporation2 = join(tPerfectIncorporation2.W, ","),
        d_tPerfectIncorporation2 = tPerfectIncorporation2.optimalValue,
        #W_CommonGround = join(CommonGround.W, ","),
        d_CommonGround = CommonGround.optimalValue,
        #W_tCommonGround2 = join(tCommonGround2.W, ","),
        d_tCommonGround2 = tCommonGround2.optimalValue,
        #W_PartialIncorporation = join(PartialIncorporation.W, ","),
        d_PartialIncorporation = PartialIncorporation.optimalValue,
        #W_tPartialIncorporation2 = join(tPartialIncorporation2.W, ","),
        d_tPartialIncorporation2 = tPartialIncorporation2.optimalValue
    ))
end

# 結果をCSVファイルに保存
CSV.write("results.csv", results)
