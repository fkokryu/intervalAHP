using CSV
using DataFrames
using IntervalArithmetic

include("../interval-ahp.jl")
include("../ttimes/optimal-value.jl")
include("../entani/common-ground.jl")
include("../ttimes-center/t-common-center.jl")

function process_pcm_data_common_all(input_file1::String, input_file2::String, output_file::String, n::Int, k::Int)
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)
    df2 = CSV.read(input_file2, DataFrame)

    # 全てのTrialのPCM_Identifiersを抽出
    extracted_trials = df1[:, :PCM_Identifiers]

    # 不要な部分を削除して必要な文字列のみを抽出
    cleaned_vector = String[]
    for item in extracted_trials
        # 不要な文字列を取り除く
        cleaned_item = replace(item, r"ny\[\"" => "")
        cleaned_item = replace(cleaned_item, "\"" => "")
        
        # 結果のベクターに追加
        push!(cleaned_vector, cleaned_item)
    end

    # Trialの番号を抽出
    trial_numbers = Int[]
    for item in cleaned_vector
        # "Trial-"に続く数字を抽出
        match_obj = match(r"Trial-(\d+)_PCM", item)
        if match_obj !== nothing
            trial_number = parse(Int, match_obj.captures[1])
            push!(trial_numbers, trial_number)
        end
    end

    # 重複を削除してユニークなリストを作成
    unique_trial_numbers = unique(trial_numbers)

    # PCMデータの列名を動的に生成
    pcm_columns = ["pcm_$i" for i in 1:n]

    # 各TrialごとにPCMデータと区間重要度データを格納するための辞書を作成
    trial_data = Dict()

    # CSVから読み込んだデータに対して、各Trialごとに処理を行う
    for trial_number in unique_trial_numbers
        # 各PCMごとのデータを格納するための辞書
        pcm_data = Dict()
        
        # 対応するTrialのデータを取得
        trial_df = filter(row -> row.Trial == trial_number, df2)
        
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

    # 結果を格納するためのDataFrameを作成
    results_df = DataFrame(
        Trial = Int[],
        PCM = Int[],
        式1の解の中心の総和 = Float64[],
        式10の解に対するt_L = Float64[],
        式10の解に対するt_U = Float64[],
        式1の最適値 = Float64[],
        式10の最適値 = Float64[],
        式10の最適値とt_Lの積 = Float64[],
        式10の最適値とt_Uの積 = Float64[],
        式2の最適値 = Vector{Union{Float64, Missing}}(),
        式20の最適値 = Vector{Union{Float64, Missing}}(),
        式20のt = Vector{Union{Float64, Missing}}()
    )

    # 計算とDataFrameへのデータ追加
    for i in unique_trial_numbers
        A = [zeros(Float64, n, n) for _ in 1:3]
        for m =1:k
            pcm, w = trial_data[i][m]
            A[m] = pcm
    
            # 区間の中心の値を計算
            centers = [(a.lo + a.hi) / 2 for a in w]
    
            # 中心の値の総和を計算
            sum_of_centers = sum(centers) #式(1)の解の中心の総和
            
            w_center1 = solveCrispAHPLP(pcm)
    
            # `candidate` 配列の初期化
            candidate = zeros(Float64, n)
            candidate2 = zeros(Float64, n)
    
            for l = 1:n
                wᵢᴸ_check = w_center1.wᴸ_center_1[l]
                ∑wⱼᵁ = sum(map(j -> w_center1.wᵁ_center_1[j], filter(j -> l != j, 1:n)))
                candidate[l] = ∑wⱼᵁ + wᵢᴸ_check

                wᵢᵁ_check = w_center1.wᵁ_center_1[l]
                ∑wⱼᴸ = sum(map(j -> w_center1.wᴸ_center_1[j], filter(j -> l != j, 1:n)))
                candidate2[l] = ∑wⱼᴸ + wᵢᵁ_check
            end
    
            t_L = 1 / minimum(candidate) #式(10)の解に対するt^L        
            t_U = 1 / maximum(candidate2) #式(10)の解に対するt^U     
    
            optimalval = solveIntervalAHPLP(pcm).optimalValue #式(1)の最適値
            optimalval_center1 = w_center1.optimalValue_center_1 #式(10)の最適値
    
            tl_times_optval_center1 = t_L * optimalval_center1 #式(10)の最適値をt_L倍したもの
            tu_times_optval_center1 = t_U * optimalval_center1 #式(10)の最適値をt_L倍したもの
    
            if(m!=k)
                # 結果をDataFrameに追加
                push!(results_df, (
                    Trial = i,
                    PCM = m,
                    式1の解の中心の総和 = sum_of_centers,
                    式10の解に対するt_L = t_L,
                    式10の解に対するt_U = t_U,
                    式1の最適値 = optimalval,
                    式10の最適値 = optimalval_center1,
                    式10の最適値とt_Lの積 = tl_times_optval_center1,
                    式10の最適値とt_Uの積 = tu_times_optval_center1,
                    式2の最適値 = missing,
                    式20の最適値 = missing,
                    式20のt = missing
                ))
            else
                commonbefore = solveCommonGroundLP([A[1], A[2], A[3]])
                if commonbefore === nothing
                    optimalval_common_entani = missing
                else
                    optimalval_common_entani = commonbefore.optimalValue_common_entani
                end

                tcommon = solvetCommonGroundLP2([A[1], A[2], A[3]])
                if tcommon === nothing
                    optimalval_tcommon = missing
                    t_tcommon = missing
                else
                    optimalval_tcommon = tcommon.optimalValue_tcommon_center_1
                    t_tcommon = tcommon.s_tcommon_center_1
                end

                # 結果をDataFrameに追加
                push!(results_df, (
                    Trial = i,
                    PCM = m,
                    式1の解の中心の総和 = sum_of_centers,
                    式10の解に対するt_L = t_L,
                    式10の解に対するt_U = t_U,
                    式1の最適値 = optimalval,
                    式10の最適値 = optimalval_center1,
                    式10の最適値とt_Lの積 = tl_times_optval_center1,
                    式10の最適値とt_Uの積 = tu_times_optval_center1,
                    式2の最適値 = optimalval_common_entani,
                    式20の最適値 = optimalval_tcommon,
                    式20のt = t_tcommon
                ))
            end
        end
    end

    # DataFrameをCSVファイルとして保存
    CSV.write(output_file, results_df)
end