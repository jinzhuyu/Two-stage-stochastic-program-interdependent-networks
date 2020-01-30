# two-stage stochastic programs for pre-disaster recovery planning of interdependent networks
# ctrl-j + ctrl-k to start a new session
clearconsole()

# xxxxx

# import packages
using JuMP, Gurobi
using Distributions, GLPKMathProgInterface
using CSV, DataFrames, Plots
# ] add LightGraphsFlows
# for finding the max flow
# ref.: https://github.com/JuliaGraphs/LightGraphsFlows.jl
# using LightGraphsFlows
# import LightGraphs

# # load data
nodes_data = CSV.read("C:\\Users\\yuj5\\Dropbox\\1-two-stage-stochastic-program-interdependent-networks\\nodes_data_test.csv")
arcs_data = CSV.read("C:\\Users\\yuj5\\Dropbox\\1-two-stage-stochastic-program-interdependent-networks\\arcs_data_test.csv")

# #  damage state
# β_n = CSV.read("C:\\Users\\yuj5\\Dropbox\\1-two-stage-stochastic-program-interdependent-networks\\is_damage_nodes_test.csv")
# β_a = CSV.read("C:\\Users\\yuj5\\Dropbox\\1-two-stage-stochastic-program-interdependent-networks\\is_damage_arcs_test.csv")

round.(Int64, arcs_data.start_node)
round.(Int64, arcs_data.end_node)

# names!(arcs_data[:, 1:9], Symbol('arc_id', 'start_node', 'end_node', 'flow',
#                                 'start_node_lat', 'start_node_long', 'end_node_lat',
#                                 'end_node_long', 'length'))
num_nodes = nrow(nodes_data)
num_arcs = nrow(arcs_data)
num_samples = 1

# scalar parameters
# proportion of budget for hardenning activities
ρ_h = 0.30 #0.2
# total budget
B = 5e5


# cost of slack
csn_unit = 1e3
# cost of flow on arcs per unit
cf_unit = 1e2

# time-dependent weight of networks
t_max=20
# wt1=collect(0.8:-(0.8 - 0.5)/t_max:0.5)
# wt2=collect((1 - 0.8):(0.8 - 0.5)/t_max:0.5)
# wt = hcat(wt1, wt2)

# probability of each scenario
prob_scen=1/num_samples

# orignal demand in total
# is_demand = nodes_data.is_demand
b_supply = nodes_data.b
orig_demand = abs.(b_supply[b_supply .< 0])
orig_demand_total = sum(orig_demand)
# orig_demand =[ sum(b[(b .<0) .& (nodes_data.net_id  .== 1)]) ;
#                    sum(b[(b .<0) .& (nodes_data.net_id  .== 2)])]
# orig_demand = abs.(orig_demand_total) # or broadcast(abs, orig_demand_total)

# # generate capacity (not real but realistic)
# μ_mean = flow/0.8
# μ_var = 3
# # srand(123) # Setting the seed
# μ_jitter = rand(Normal(0, μ_var), num_arcs) # round it later
# μ = μ_mean + μ_jitter
# insert!(arcs_data, 5, μ, :cap)

# create node and arcs id set
nodes_id = nodes_data.node_id
arcs_id = arcs_data.arc_id
# create nodes array
nodes = Array{Int64, 1}(undef, num_nodes)
for i = 1:num_nodes
  nodes[i] = nodes_id[i]
end
# create arc tunple
arcs = Array{Tuple{Int64, Int64}, 1}(undef, num_arcs)
for i = 1:num_arcs
  arcs[i] = (arcs_data.start_node[i], arcs_data.end_node[i])
end

# inter_arcs_P = arcs_id[(arcs_data.start_node.<100) .& (arcs_data.end_node.>100)]
# inter_arcs_W = arcs_id[(arcs_data.start_node.>100) .& (arcs_data.end_node.<100)]
# inter_arcs_index = [inter_arcs_P; inter_arcs_W]

# demand_nodes_P = nodes_id[(b_supply .< 0) .& (nodes_data.net_id .== 1)]
# demand_nodes_W = nodes_id[(b_supply .< 0) .& (nodes_data.net_id .== 2)]
demand_nodes = nodes_id[b_supply .< 0]
num_demand = length(demand_nodes)
supply_nodes = nodes_id[b_supply .> 0]
tran_nodes = nodes_id[b_supply .== 0]

time_set = collect(1:t_max)
Ω = collect(1:num_samples)

ch_n = nodes_data.chn
ch_a = arcs_data.cha

# # to make sure dr_n-1 is no less than 1
dr_n = nodes_data.rn
dr_n_copy = copy(dr_n)
dr_n_copy[dr_n_copy .== 1].=2
dr_a = arcs_data.ra
dr_a_copy = copy(dr_a)
dr_a_copy[dr_a_copy .== 1].=2

cr_n = nodes_data.crn
cr_a = arcs_data.cra

# Converting arrays to dictionaries/tuples
# ref.: https://www.softcover.io/read/7b8eb7d0/juliabook/network

# num_inter_arcs = length(inter_arcs_index)
# inter_arcs = Array{Tuple{Int64, Int64}, 1}(undef, num_inter_arcs)
# for i = 1:num_inter_arcs
#     inter_arcs[i] = (arcs_data.start_node[inter_arcs_index[i]],
#                      arcs_data.end_node[inter_arcs_index[i]])
# end

cost_flow_dict = Dict()
u_a_dict = Dict()
ch_a_dict = Dict()
cr_a_dict = Dict()
dr_a_dict = Dict()
β_a_index_dict = Dict()

for i = 1:num_arcs
  cost_flow_dict[(arcs_data.start_node[i], arcs_data.end_node[i])] = cf_unit*arcs_data.arc_flow[i]
  u_a_dict[(arcs_data.start_node[i], arcs_data.end_node[i])] = arcs_data.u[i]
  ch_a_dict[(arcs_data.start_node[i], arcs_data.end_node[i])] = ch_a[i]
  cr_a_dict[(arcs_data.start_node[i], arcs_data.end_node[i])] = cr_a[i]
  dr_a_dict[(arcs_data.start_node[i], arcs_data.end_node[i])] = dr_a_copy[i]
  β_a_index_dict[(arcs_data.start_node[i], arcs_data.end_node[i])] = i
end

# β_a[β_a_index_dict[(1,2)],18]

# nodes_dict = Dict(zip(nodes_data.node_id, 1:num_nodes))
# is_demand_dict = Dict(zip(nodes_data.node_id, is_demand))
ch_n_dict = Dict(zip(nodes_data.node_id, ch_n))
dr_n_dict = Dict(zip(nodes_data.node_id, dr_n_copy))

cr_n_dict = Dict(zip(nodes_data.node_id, cr_n))
# csn_unit_dict = Dict(zip(nodes_data.node_id, csn_unit))
b_dict = Dict(zip(nodes_data.node_id, b_supply))
ρ_redun = 1  # equal to normal flow divided by capacity
u_n_dict = Dict(zip(nodes_data.node_id, nodes_data.b/ρ_redun))
β_n_index_dict = Dict(zip(nodes_data.node_id, 1:num_nodes))

# initial slack should be determined by flow redistribution, but skipped for now for simplicity
# arc_flow_0 = repeat(arcs_data.arc_flow, 1, num_samples).* is_not_damaged_arcs
# σ_0 = orig_demand -
# max flow problem data preparation
    # arc_flow_damaged = Array{Tuple{Int, Int, Float64}, 1}(undef, num_arcs)
    # for i = 1:num_arcs
    #   arc_flow_damaged[i] =[(arcs_data.start_node[i], arcs_data.end_node[i], arc_flow_0[:, i])
    # end

# creating the LP equivalent (extensive formulation) model

time_limit = 3600*12    # 12 hours
# extsv_model = Model(solver = CplexSolver(CPX_PARAM_TILIM = UB_time, CPX_PARAM_EPGAP = 1e-3))
extsv_model = Model(solver=GurobiSolver(TimeLimit = time_limit))

# # first-stage
@variable(extsv_model, x_n[i in nodes], Bin)
@variable(extsv_model, x_a[arc in arcs], Bin)

# # limited cost
@constraint(extsv_model, cost1, sum(x_n[i]*ch_n_dict[i] for i in nodes) +
                                sum(x_a[arc]*ch_a_dict[arc] for arc in arcs) <= ρ_h*B)

# sum(ch_n_dict[i] for i in nodes) + sum(ch_a_dict[arc] for arc in arcs) >= ρ_h*B # true

# second stage
# weather or not a repair crew starts to repair a component
@variable(extsv_model, y_n[i in nodes, t in 1:t_max, ω in Ω], Bin)
@variable(extsv_model, y_a[arc in arcs, t in 1:t_max, ω in Ω], Bin)

# weather components is functional
@variable(extsv_model, z_n[i in nodes, t in 0:t_max, ω in Ω], Bin)
@variable(extsv_model, z_a[arc in arcs, t in 0:t_max, ω in Ω], Bin)

# weather a repair crew is working at a component
@variable(extsv_model, v_n[i in nodes, t in 1:t_max, ω in Ω], Bin)
@variable(extsv_model, v_a[arc in arcs, t in 1:t_max, ω in Ω], Bin)

# weather a component has been repaired by time t
@variable(extsv_model, w_n[i in nodes, t in 1:t_max, ω in Ω], Bin)
@variable(extsv_model, w_a[arc in arcs, t in 1:t_max, ω in Ω], Bin)

# flow on arcs
@variable(extsv_model, 0 <= f[arc in arcs, t in 1:t_max, ω in Ω] <= u_a_dict[arc])



# actual supply at nodes
# supply nodes b>0; demand nodes b<0; tran nodes, b=0.
@variable(extsv_model, 0 <= q[node in supply_nodes, t in 1:t_max, ω in Ω]<= u_n_dict[node])
# slack
@variable(extsv_model, 0 <= s[node in demand_nodes, t in 1:t_max, ω in Ω] <= -b_dict[node])

# check if slack is non-decreasing

###########
# !!! s + q = -b
# demand node flow conservation: inflow = q
###########

# objective function
# @objective(extsv_model, Max, sum(w[, t]*(sum(f[(j, ii), t]*is_demand_dict[i]
#   *α[i, t] for (j, ii) in arcs if ii == i)/orig_demand_total)  for t in 1:t_max))
@objective(extsv_model, Min,
    prob_scen*sum(s[i, t, ω] for i in demand_nodes, t in 1:t_max, ω in Ω))

# constraints
# cost of restoration is upbounded
@constraint(extsv_model, cost2[ω in Ω],
    sum(csn_unit*s[i, t, ω] for i in demand_nodes, t in 1:t_max) +
    sum(cf_unit*f[arc, t, ω] for arc in arcs, t in 1:t_max) +
    sum(cr_n_dict[i]*v_n[i, t, ω] for i in nodes, t in 1:t_max) +
    sum(cr_a_dict[arc]*v_a[arc, t, ω] for arc in arcs, t in 1:t_max) <=
    (1 - ρ_h)*B)

# sum(csn_unit*s_sol[i, t, 1] for i in demand_nodes, t in 1:t_max)
# (1 - ρ_h)*B
# sum(cf_unit*f[arc, t, ω] for arc in arcs, t in 1:t_max)

# flow conservation
# transmission nodes
@constraint(extsv_model, flow_balnc_tran[t in 1:t_max, ω in Ω, i in tran_nodes],
    sum(f[(ii, j), t, ω] for (ii, j) in arcs if ii == i) ==
    sum(f[(j, ii), t, ω] for (j, ii) in arcs if ii == i))

# supply nodes
@constraint(extsv_model, flow_balnc_sup[t in 1:t_max, ω in Ω, i in supply_nodes],
    sum(f[(ii, j), t, ω] for (ii, j) in arcs if ii == i) == q[i, t, ω])

# demand nodes
@constraint(extsv_model, flow_balnc_dem[t in 1:t_max, ω in Ω, i in demand_nodes],
    sum(f[(j, ii), t, ω] for (j,ii) in arcs if ii == i) == -b_dict[i] - s[i, t, ω] )
           # @constraint(extsv_model, q[i, t, w] + s[i, t, w] == -b_dict[i])

# println(f_sol[(j, ii), :, 1] for (j,ii) in arcs if ii == 22)

# operational constraints
@constraint(extsv_model, [arc in arcs, t in 1:t_max, ω in Ω],
    f[arc, t, ω] <= u_a_dict[arc]*z_a[arc, t, ω])
@constraint(extsv_model, [(i, j) in arcs, t in 1:t_max, ω in Ω],
    f[(i, j), t, ω] <= u_a_dict[(i, j)]*z_n[i, t, ω])  # u-a instead of u_n
# end node of arcs have to work before the arcs can be functioning
@constraint(extsv_model, [(i, j) in arcs, t in 1:t_max, ω in Ω],
    f[(i, j), t, ω] <= u_a_dict[(i, j)]*z_n[j, t, ω])  # u-a instead of u_n

# # additional: the start node and end node of any functional arc should be functional
# @constraint(extsv_model, [(i, j) in arcs, t in 1:t_max, ω in Ω],
#             z_a[(i, j), t, ω] <= z_n[i, t, ω]*z_n[j, t, ω])

# Note: arcs contains inter_arcs, so the following constraint is satified automatically
# physical dependency: node dependent on link
# @constraint(extsv_model, [(i, j) in inter_arcs, t in 1:t_max], z_a[j, t] <= z_n[i, t, ω])

# # scheduling constraints
# # repair crew work at the same component until the component is restored.
# @constraint(extsv_model, [i in nodes, t in 1:t_max, ω in Ω],
#     v_n[i,t, ω] == sum(y_n[i, tt, ω] for tt = max(1,t-dr_n_dict[i]+1):t))
# @constraint(extsv_model, [arc in arcs, t in 1:t_max, ω in Ω],
#     v_a[arc,t, ω] == sum(y_a[arc, tt, ω] for tt = max(1, t-dr_a_dict[arc]+1):t))
#
# # number of repair crew is upbounded at any time period
# # Note: to do - modify it to differentiate between repair crews for different networks
#
# @constraint(extsv_model, crew_bound[t in 1:t_max, ω in Ω],
#     sum(v_n[i, t, ω] for i in nodes) + sum(v_a[arc, t, ω] for arc in arcs) <= sum(N_R))
#                             # sum(N[m]) m in network_index)
#
#
# # components won't be restored until repair crews start to work at it at t-dr
# @constraint(extsv_model, [i in nodes, t in 1:t_max, ω in Ω],
#     w_n[i, t, ω] == sum(y_n[i, tt, ω] for tt = 1:min(t_max, t-dr_n_dict[i])))
# @constraint(extsv_model, [arc in arcs, t in 1:t_max, ω in Ω],
#     w_a[arc, t, ω] == sum(y_a[arc, tt, ω] for tt = 1:min(t_max, t-dr_a_dict[arc])))
#
# # components will not be operational until restored
# @constraint(extsv_model, [i in nodes, t in 1:t_max, ω in Ω],
#     z_n[i, t, ω] <= z_n[i, t-1, ω] + w_n[i, t, ω])
# @constraint(extsv_model, [arc in arcs, t in 1:t_max, ω in Ω],
#     z_a[arc, t, ω] <= z_a[arc, t-1, ω] + w_a[arc, t, ω])



@constraint(extsv_model, [i in nodes, t in 1:t_max, ω in Ω],
    z_n[i, t-1, ω] <= z_n[i, t, ω])
@constraint(extsv_model, [arc in arcs, t in 1:t_max, ω in Ω],
    z_a[arc, t-1, ω] <= z_a[arc, t, ω])

# impact of first stage decision variables
# @constraint(extsv_model, [node in nodes, ω in Ω], z_n[node, 0, ω] <= β_n[β_n_index_dict[node], ω] + x_n[node])
# @constraint(extsv_model, [arc in arcs, ω in Ω], z_a[arc, 0, ω] <= β_a[β_a_index_dict[arc], ω] + x_a[arc])
# @constraint(extsv_model, [node in nodes, ω in Ω], β_n[β_n_index_dict[node], ω] + x_n[node] <=1)
# @constraint(extsv_model, [arc in arcs, ω in Ω],  β_a[β_a_index_dict[arc], ω] + x_a[arc] <=1)

# @constraint(extsv_model, [node in nodes, ω in Ω], z_n[node, 0, ω] + β_n[node, ω] ==1)
# @constraint(extsv_model, [arc in arcs, ω in Ω], z_a[arc, 0, ω] + β_a[arc, ω] ==1)

# suppose that all components are damaged (not functioning) at t=0
@constraint(extsv_model, [node in nodes, ω in Ω], z_n[node, 0, ω] ==0)
@constraint(extsv_model, [arc in arcs, ω in Ω], z_a[arc, 0, ω] ==0)

# anonymous constraint container by dropping the name
# ref.: https://github.com/JuliaOpt/JuMP.jl/blob/master/docs/src/constraints.md
# @constraint(extsv_model, x_n[i] for i in nodes -
#                 α[i, j] for i in nodes for, j in 1:t_max <= 0)
# for i in nodes, j in 1:t_max
#     @constraint(extsv_model, x_n[i] - α[i, j] <= 0)
# end

# solution
t_0 = time_ns()
status = solve(extsv_model)
t_1 = time_ns()

# # solution time
# t_solve_s = round((t_1 - t_0)*1e-9, digits = 4)
# println("The solution time is: ", t_solve_s, " seconds.")

# Optimal solutions
# println("Optimal Solutions:")
println("Status = $status")
println("Optimal objective Function value: ", getobjectivevalue(extsv_model))

# slack
s_sol = getvalue(s)
sum(csn_unit*s_sol[i, t, 1] for i in demand_nodes, t in 1:t_max)
println("\nSlack:\n", s_sol)

# flow on arcs
f_sol = getvalue(f)
println("Flow on arcs:\n", f_sol)

# # demand node:7,8
# ii = 8
# tt = t_max
# println(f_sol[(5,ii),tt,1]+f_sol[(6,ii),tt,1])
# println(sum(f_sol[(j, ii), tt, 1] if (j, ii) in arcs for j in nodes_id))
# println(f_sol[(i,j), 1, 1] for (i,j) in arcs)

# # supply node:1,2
# node_sup =1
# println(f_sol[(node_sup,4),tt,1]+f_sol[(node_sup,5),tt,1])

# y_n_sol = getvalue(y_n)
# println("Scheduling of crews at nodes:\n", y_n_sol)

z_n_sol = getvalue(z_n)
println(z_n_sol)

# w_n_sol = getvalue(w_n)
# println("Node repaired by time:\n", w_n_sol)
#
# println("Supply:\n", getvalue(q))
