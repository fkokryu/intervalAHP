using Random
using CSV
using DataFrames
using Base.Threads

# Mutexの初期化
mutex = Threads.Mutex()

# ランダムにPCMを選択する関数
function random_select_pcms(pcms, count)
    return pcms[shuffle(1:length(pcms))[1:count]]
end

# 結果を保存するためのデータフレームを初期化
results = DataFrame(
    d_PerfectIncorporation = Float64[],
    d_tPerfectIncorporation2 = Float64[],
    d_CommonGround = Float64[],
    d_tCommonGround2 = Float64[],
    d_PartialIncorporation = Float64[],
    d_tPartialIncorporation2 = Float64[]
)

# 1000回の実験を実行
Threads.@threads for _ in 1:1000
    selected_pcms = random_select_pcms(pcms, 3)

    # 各手法の計算とエラーチェック

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

    # スレッドごとの結果を一時的に保存
    local_results = (
        d_PerfectIncorporation = PerfectIncorporation.optimalValue,
        d_tPerfectIncorporation2 = tPerfectIncorporation2.optimalValue,
        d_CommonGround = CommonGround.optimalValue,
        d_tCommonGround2 = tCommonGround2.optimalValue,
        d_PartialIncorporation = PartialIncorporation.optimalValue,
        d_tPartialIncorporation2 = tPartialIncorporation2.optimalValue
    )

    # Mutexを使用してスレッドセーフに結果を追加
    lock(mutex) do
        push!(results, local_results)
    end
end

# 結果をCSVファイルに保存
CSV.write("results.csv", results)