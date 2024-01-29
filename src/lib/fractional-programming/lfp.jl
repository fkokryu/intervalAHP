using JuMP, Ipopt

# 問題のパラメータ
pcm = [
    1.0	5.0	1.0	1.0	2.0
    0.2	1.0	1.0	1.0	2.0
    1.0	1.0	1.0	1.0	3.0
    1.0	1.0	1.0	1.0	3.0
    0.5	0.5	0.3333333333333330	0.3333333333333330	1.0
]
epsilon = 1e-8
N = 5

# 最適化モデルを作成
model = Model(Ipopt.Optimizer)

# 変数の宣言
@variable(model, w_U[i=1:N] >= epsilon)
@variable(model, w_L[i=1:N] >= epsilon)

# 目的関数の設定
@NLobjective(model, Min, sum(w_U[i] - w_L[i] for i in 1:N) / sum((w_U[i] + w_L[i])/2 for i in 1:N))

# 制約の追加
for i in 1:N
    for j in 1:N
        if i != j
            @constraint(model, w_L[i] <= pcm[i,j] * w_U[j])
            @constraint(model, pcm[i,j] * w_L[j] <= w_U[i])
        end
    end
    @constraint(model, sum(w_U[j] for j in 1:N if j != i) + w_L[i] >= 1)
    @constraint(model, sum(w_L[j] for j in 1:N if j != i) + w_U[i] <= 1)
end

# モデルの最適化
optimize!(model)

# 結果の表示
println("Objective value: ", objective_value(model))
for i in 1:N
    println("w_U[$i]: ", value(w_U[i]))
    println("w_L[$i]: ", value(w_L[i]))
end
