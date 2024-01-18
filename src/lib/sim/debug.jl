using CSV
using DataFrames
using DataStructures
using IntervalSets

# CSVファイルを読み込む
df1 = CSV.read("5_3_0.65.csv", DataFrame)
df2 = CSV.read("PCM_5_3_0.65.csv", DataFrame)

# `entani`論文のPerfectIncorporationが解の非唯一性考慮のPerfectIncorporationより小さいTrial番号を抽出
selected_trials = df1[df1[:, :entani論文のPerfectIncorporation] .< df1[:, :解の非唯一性考慮のPerfectIncorporation], :PCM_Identifiers]

# 文字列を処理して"Trial-X_PCM-Y"形式に分解
extracted_trials = String[]
for trial in selected_trials
    # 文字列を分解して個々の要素を抽出
    trial_parts = split(trial[2:end-1], "\", \"")  # "Any["と"]"を除去して分割
    append!(extracted_trials, trial_parts)
end

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

# Trialごとのデータを格納するための辞書を作成
trial_data = OrderedDict{Int, Tuple{Array{Float64,2}, Array{Interval{Float64},1}}}()

# 指定されたTrial番号ごとにデータを抽出して辞書に格納
for trial_number in trial_numbers
    trial_rows = filter(row -> row.Trial == trial_number, df2)
    if !isempty(trial_rows)
        pcm_data = convert(Array{Float64,2}, trial_rows[:, ["pcm_1", "pcm_2", "pcm_3", "pcm_4", "pcm_5"]])
        w_data = [Interval{Float64}(row.w_L, row.w_U) for row in eachrow(trial_rows)]
        trial_data[trial_number] = (pcm_data, w_data)
    end
end

# 結果の確認（例：Trial 2のデータを表示）
trial2_pcm, trial2_w = trial_data[2]
println("PCM Data for Trial 2:")
println(trial2_pcm)
println("Interval Importance for Trial 2:")
println(trial2_w)