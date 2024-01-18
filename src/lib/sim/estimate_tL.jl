using CSV
using DataFrames
using IntervalArithmetic

include("../ttimes/optimal-value.jl")

function calculate_and_save_pcm_results(pcm_file::String, output_file::String, n::Int, k::Int)
    # PCMファイルを読み込む
    df_pcm = CSV.read(pcm_file, DataFrame)

    # 結果を格納するためのDataFrameを作成
    results_df = DataFrame(
        Trial = Int[],
        PCM = Int[],
        t_L = Float64[]
    )

    # PCMデータの列名を動的に生成
    pcm_columns = ["pcm_$i" for i in 1:n]

    # 各TrialごとにPCMデータと区間重要度データを格納するための辞書を作成
    trial_data = Dict()

    # CSVから読み込んだデータに対して、各Trialごとに処理を行う
    for trial_number = 1:1000
        # 各PCMごとのデータを格納するための辞書
        pcm_data = Dict()
        
        # 対応するTrialのデータを取得
        trial_df = filter(row -> row.Trial == trial_number, df_pcm)
        
        for pcm_number in unique(trial_df.PCM)
            # 対応するPCMのデータを取得
            pcm_df = filter(row -> row.PCM == pcm_number, trial_df)
            
            # PCMデータと区間重要度データを取得
            pcm_values = Matrix{Float64}(pcm_df[:, pcm_columns])
            w_values = [interval(row.w_L, row.w_U) for row in eachrow(pcm_df)]
            
            # PCMデータ辞書に格納
            pcm_data[pcm_number] = (pcm_values, w_values)
        end
        
        # 最終的なTrialデータ辞書に格納
        trial_data[trial_number] = pcm_data
    end

    # 計算とDataFrameへのデータ追加
    for i = 1:1000
        for m =1:k
            pcm, w = trial_data[i][m]

            w_center1 = solveCrispAHPLP(pcm)
    
            # `candidate` 配列の初期化
            candidate = zeros(Float64, n)
    
            for l = 1:n
                wᵢᴸ_check = w_center1.wᴸ_center_1[l]
                ∑wⱼᵁ = sum(map(j -> w_center1.wᵁ_center_1[j], filter(j -> l != j, 1:n)))
                candidate[l] = ∑wⱼᵁ + wᵢᴸ_check
            end
    
            t_L = 1 / minimum(candidate) #式(5)の解に対するt_L
    
            # 結果をDataFrameに追加
            push!(results_df, (
                Trial = i,
                PCM = m,
                t_L = t_L,
            ))
        end
    end
    
    # 計算とDataFrameへのデータ追加
for i = 1:1000
    if haskey(trial_data, i)
        for m = 1:k
            if haskey(trial_data[i], m)
                pcm, w = trial_data[i][m]
                w_center1 = solveCrispAHPLP(pcm)
                 # `candidate` 配列の初期化
                candidate = zeros(Float64, n)
        
                for l = 1:n
                    wᵢᴸ_check = w_center1.wᴸ_center_1[l]
                    ∑wⱼᵁ = sum(map(j -> w_center1.wᵁ_center_1[j], filter(j -> l != j, 1:n)))
                    candidate[l] = ∑wⱼᵁ + wᵢᴸ_check
                end
        
                t_L = 1 / minimum(candidate) #式(5)の解に対するt_L
        
                # 結果をDataFrameに追加
                push!(results_df, (
                    Trial = i,
                    PCM = m,
                    t_L = t_L,
                ))
            else
                # PCMデータが存在しない場合の処理
            end
        end
    else
        # Trialデータが存在しない場合の処理
    end
end


    # DataFrameをCSVファイルとして保存
    CSV.write(output_file, results_df)
end