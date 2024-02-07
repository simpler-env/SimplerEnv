import numpy as np

def pearson_correlation(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    x = x - np.mean(x)
    y = y - np.mean(y)
    if np.all(x == y):
        pearson = 1
    else:
        pearson = np.sum(x * y) / (np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-8)
    return pearson
    
def pearson_correlation_std_discrep(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    pearson = pearson_correlation(x, y)
    discrep = (1 - pearson) * max(np.std(x), np.std(y))
    return discrep

def mean_diff(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    return np.abs(np.mean(x) - np.mean(y))

def ranking_violation(x, y):
    # assuming x is sim result and y is real result
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    rank_violation = 0.0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if (x[i] > x[j]) != (y[i] > y[j]):
                rank_violation = max(rank_violation, np.abs(y[i] - y[j]))
                # rank_violation += np.abs(y[i] - y[j])
    return rank_violation
    # return rank_violation / (len(x) * (len(x) - 1) / 2)

rt_ckpts = ['rt-1-new-best-late', 'rt-1-new-early', 'rt-1-x', 'rt-1-early-1k']
coke_can_horizontal_sim_success = [1.0, 0.84, 0.80, 0.08]
coke_can_horizontal_real_success = [0.96, 1.0, 0.88, 0.20]
coke_can_vertical_sim_success = [0.80, 0.72, 0.48, 0.00]
coke_can_vertical_real_success = [0.88, 0.96, 0.56, 0.00]
coke_can_standing_sim_success = [0.80, 0.60, 0.68, 0.00]
coke_can_standing_real_success = [0.72, 0.80, 0.84, 0.20]

coke_can_avg_sim = np.mean([coke_can_horizontal_sim_success, coke_can_vertical_sim_success, coke_can_standing_sim_success], axis=0)
coke_can_avg_real = np.mean([coke_can_horizontal_real_success, coke_can_vertical_real_success, coke_can_standing_real_success], axis=0)
print("coke_can_average_sim", coke_can_avg_sim)
print("coke_can_average_real", coke_can_avg_real)

print("mean_diff(coke_can_horizontal_sim, coke_can_horizontal_real)", mean_diff(coke_can_horizontal_sim_success, coke_can_horizontal_real_success))
print("pearson_correlation(coke_can_horizontal_sim, coke_can_horizontal_real)", pearson_correlation(coke_can_horizontal_sim_success, coke_can_horizontal_real_success))
print("ranking_violation(coke_can_horizontal_sim, coke_can_horizontal_real)", ranking_violation(coke_can_horizontal_sim_success, coke_can_horizontal_real_success))
print("mean_diff(coke_can_vertical_sim, coke_can_vertical_real)", mean_diff(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("pearson_correlation(coke_can_vertical_sim, coke_can_vertical_real)", pearson_correlation(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("ranking_violation(coke_can_vertical_sim, coke_can_vertical_real)", ranking_violation(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("mean_diff(coke_can_standing_sim, coke_can_standing_real)", mean_diff(coke_can_standing_sim_success, coke_can_standing_real_success))
print("pearson_correlation(coke_can_standing_sim, coke_can_standing_real)", pearson_correlation(coke_can_standing_sim_success, coke_can_standing_real_success))
print("ranking_violation(coke_can_standing_sim, coke_can_standing_real)", ranking_violation(coke_can_standing_sim_success, coke_can_standing_real_success))
print("mean_diff(coke_can_avg_sim, coke_can_avg_real)", mean_diff(coke_can_avg_sim, coke_can_avg_real))
print("pearson_correlation(coke_can_avg_sim, coke_can_avg_real)", pearson_correlation(coke_can_avg_sim, coke_can_avg_real))
print("ranking_violation(coke_can_avg_sim, coke_can_avg_real)", ranking_violation(coke_can_avg_sim, coke_can_avg_real))

print("=" * 60)

coke_can_lr_boundary_sim_success = [(9+9+6)/30, (7+4+7)/30, (8+3+5)/30]
coke_can_lr_boundary_real_success = [21/30, 23/30, 19/30]
coke_can_middle_sim_success = [(15+12+13)/45, (14+14+8)/45, (12+10+11)/45]
coke_can_middle_real_success = [43/45, 33/45, 38/45]

print("coke_can_lr_boundary_sim_success", coke_can_lr_boundary_sim_success)
print("coke_can_lr_boundary_real_success", coke_can_lr_boundary_real_success)
print("coke_can_middle_sim_success", coke_can_middle_sim_success)
print("coke_can_middle_real_success", coke_can_middle_real_success)

print("=" * 60)

coke_can_l_sim_success = [(9+7+9)/30, (7+8+3)/30, (10+3+9)/30]
coke_can_l_real_success = [22/30, 19/30, 19/30]
coke_can_r_sim_success = [(15+14+11)/45, (14+10+12)/45, (10+10+7)/45]
coke_can_r_real_success = [42/45, 36/45, 38/45]

print("coke_can_l_sim_success", coke_can_l_sim_success)
print("coke_can_l_real_success", coke_can_l_real_success)
print("coke_can_r_sim_success", coke_can_r_sim_success)
print("coke_can_r_real_success", coke_can_r_real_success)

print("=" * 60)

move_near_sim_success = [0.30, 0.367, 0.35, 0.04]
move_near_real_success = [0.633, 0.583, 0.45, 0.00]

print("mean_diff(move_near_sim_success, move_near_real_success)", mean_diff(move_near_sim_success, move_near_real_success))
print("pearson_correlation(move_near_sim_success, move_near_real_success)", pearson_correlation(move_near_sim_success, move_near_real_success))
print("ranking_violation(move_near_sim_success, move_near_real_success)", ranking_violation(move_near_sim_success, move_near_real_success))

print("=" * 60)

# [base success, background, lighting, distractor, table tex (tab6/tab5, note the order), cam pose]
coke_can_rt1_new_late_sim_table_avg = np.mean(
    [[[0.96, 1.00, 1.00, 0.92, 0.88, 0.04],
     [0.84, 0.84, 0.92, 0.76, 0.44, 0.08],
     [0.96, 0.96, 0.96, 0.96, 0.92, 0.04]],
     [[0.96, 1.00, 1.00, 0.96, 1.00, 0.28],
     [0.84, 0.72, 0.92, 0.80, 0.60, 0.00],
     [0.96, 1.00, 0.96, 0.96, 1.00, 0.56]],
    ],
    axis=1
)
coke_can_gengap1234_sim_table_avg = np.mean(
    [[[0.92, 0.80, 0.96, 0.76, 0.92, 0.16],
     [0.84, 0.56, 0.76, 0.52, 0.52, 0.04],
     [0.96, 0.76, 0.92, 0.88, 0.72, 0.36]],
     [[0.92, 0.84, 0.96, 0.88, 0.88, 0.44],
     [0.84, 0.60, 0.84, 0.60, 0.80, 0.24],
     [0.96, 0.60, 0.88, 0.88, 0.68, 0.68]],
    ],
    axis=1
)
coke_can_gengap1_sim_table_avg = np.mean(
    [[[0.96, 0.84, 1.00, 0.88, 0.64, 0.20],
     [0.68, 0.60, 0.76, 0.68, 0.16, 0.00],
     [0.76, 0.80, 0.52, 0.88, 0.68, 0.12]],
     [[0.96, 0.72, 0.96, 0.92, 0.84, 0.32],
     [0.68, 0.64, 0.72, 0.52, 0.56, 0.00],
     [0.76, 0.28, 0.64, 0.80, 0.60, 0.48]],
    ],
    axis=1
)

print("coke_can_rt1_new_late_sim_table_avg", coke_can_rt1_new_late_sim_table_avg)
print("coke_can_gengap1234_sim_table_avg", coke_can_gengap1234_sim_table_avg)
print("coke_can_gengap1_sim_table_avg", coke_can_gengap1_sim_table_avg)


coke_can_rt1_sim_factor_diff_wo_absolute = coke_can_rt1_new_late_sim_table_avg[:, 1:] - coke_can_rt1_new_late_sim_table_avg[:, [0]]
coke_can_gengap1234_sim_factor_diff_wo_absolute = coke_can_gengap1234_sim_table_avg[:, 1:] - coke_can_gengap1234_sim_table_avg[:, [0]]
coke_can_gengap1_sim_factor_diff_wo_absolute = coke_can_gengap1_sim_table_avg[:, 1:] - coke_can_gengap1_sim_table_avg[:, [0]]
coke_can_rt1_sim_factor_diff = np.abs(coke_can_rt1_new_late_sim_table_avg[:, 1:] - coke_can_rt1_new_late_sim_table_avg[:, [0]])
coke_can_gengap1234_sim_factor_diff = np.abs(coke_can_gengap1234_sim_table_avg[:, 1:] - coke_can_gengap1234_sim_table_avg[:, [0]])
coke_can_gengap1_sim_factor_diff = np.abs(coke_can_gengap1_sim_table_avg[:, 1:] - coke_can_gengap1_sim_table_avg[:, [0]])



move_near_rt1_new_late_sim_table_avg = np.array([[0.467, 0.533, 0.483, 0.600, 0.200, 0.117], 
                                                 [0.467, 0.567, 0.600, 0.600, 0.550, 0.433]])
move_near_gengap1234_sim_table_avg = np.array([[0.267, 0.283, 0.300, 0.367, 0.150, 0.100],
                                               [0.267, 0.317, 0.317, 0.367, 0.433, 0.417]])
move_near_gengap1_sim_table_avg = np.array([[0.383, 0.483, 0.517, 0.467, 0.133, 0.200],
                                            [0.383, 0.467, 0.483, 0.467, 0.450, 0.217]])

move_near_rt1_sim_factor_diff_wo_absolute = move_near_rt1_new_late_sim_table_avg[:, 1:] - move_near_rt1_new_late_sim_table_avg[:, [0]]
move_near_gengap1234_sim_factor_diff_wo_absolute = move_near_gengap1234_sim_table_avg[:, 1:] - move_near_gengap1234_sim_table_avg[:, [0]]
move_near_gengap1_sim_factor_diff_wo_absolute = move_near_gengap1_sim_table_avg[:, 1:] - move_near_gengap1_sim_table_avg[:, [0]]
move_near_rt1_sim_factor_diff = np.abs(move_near_rt1_new_late_sim_table_avg[:, 1:] - move_near_rt1_new_late_sim_table_avg[:, [0]])
move_near_gengap1234_sim_factor_diff = np.abs(move_near_gengap1234_sim_table_avg[:, 1:] - move_near_gengap1234_sim_table_avg[:, [0]])
move_near_gengap1_sim_factor_diff = np.abs(move_near_gengap1_sim_table_avg[:, 1:] - move_near_gengap1_sim_table_avg[:, [0]])


raw_rt1_avg_sim_factor_diff_wo_absolute = np.mean(np.stack([coke_can_rt1_sim_factor_diff_wo_absolute, move_near_rt1_sim_factor_diff_wo_absolute], axis=0), axis=0)
raw_gengap1234_avg_sim_factor_diff_wo_absolute = np.mean(np.stack([coke_can_gengap1234_sim_factor_diff_wo_absolute, move_near_gengap1234_sim_factor_diff_wo_absolute], axis=0), axis=0)
raw_gengap1_avg_sim_factor_diff_wo_absolute = np.mean(np.stack([coke_can_gengap1_sim_factor_diff_wo_absolute, move_near_gengap1_sim_factor_diff_wo_absolute], axis=0), axis=0)
print("raw_rt1_avg_sim_factor_diff w/o absolute, size [num_variants, num_factors]", raw_rt1_avg_sim_factor_diff_wo_absolute)
print("raw_gengap1234_avg_sim_factor_diff w/o absolute, size [num_variants, num_factors]", raw_gengap1234_avg_sim_factor_diff_wo_absolute)
print("raw_gengap1_avg_sim_factor_diff , size [num_variants, num_factors]", raw_gengap1_avg_sim_factor_diff_wo_absolute)



coke_can_rt1_sim_factor_diff = np.mean(coke_can_rt1_sim_factor_diff, axis=0)
coke_can_gengap1234_sim_factor_diff = np.mean(coke_can_gengap1234_sim_factor_diff, axis=0)
coke_can_gengap1_sim_factor_diff = np.mean(coke_can_gengap1_sim_factor_diff, axis=0)
move_near_rt1_sim_factor_diff = np.mean(move_near_rt1_sim_factor_diff, axis=0)
move_near_gengap1234_sim_factor_diff = np.mean(move_near_gengap1234_sim_factor_diff, axis=0)
move_near_gengap1_sim_factor_diff = np.mean(move_near_gengap1_sim_factor_diff, axis=0)





rt1_real_avg = np.array([0.9167, 0.8888, 0.8333, 0.8056, 0.5278, 0.4583])
gengap1234_real_avg = np.array([0.8333, 0.7222, 0.7500, 0.4444, 0.7500, 0.5000])
gengap1_real_avg = np.array([0.8333, 0.6667, 0.7917, 0.7500, 0.6667, 0.4583])
rt1_real_factor_diff = np.abs(rt1_real_avg[1:] - rt1_real_avg[0])
gengap1234_real_factor_diff = np.abs(gengap1234_real_avg[1:] - gengap1234_real_avg[0])
gengap1_real_factor_diff = np.abs(gengap1_real_avg[1:] - gengap1_real_avg[0])

print("coke_can_rt1_sim_factor_diff, move_near_rt1_sim_factor_diff, rt1_real_factor_diff", coke_can_rt1_sim_factor_diff, move_near_rt1_sim_factor_diff, rt1_real_factor_diff)
print("coke_can_gengap1234_sim_factor_diff, move_near_gengap1234_sim_factor_diff, gengap1234_real_factor_diff", coke_can_gengap1234_sim_factor_diff, move_near_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)
print("coke_can_gengap1_sim_factor_diff, move_near_gengap1_sim_factor_diff, gengap1_real_factor_diff", coke_can_gengap1_sim_factor_diff, move_near_gengap1_sim_factor_diff, gengap1_real_factor_diff)
print("pearson_correlation(coke_can_rt1_sim_factor_diff, rt1_real_factor_diff)", pearson_correlation(coke_can_rt1_sim_factor_diff, rt1_real_factor_diff))
print("pearson_correlation(coke_can_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)", pearson_correlation(coke_can_gengap1234_sim_factor_diff, gengap1234_real_factor_diff))
print("pearson_correlation(coke_can_gengap1_sim_factor_diff, gengap1_real_factor_diff)", pearson_correlation(coke_can_gengap1_sim_factor_diff, gengap1_real_factor_diff))
print("pearson_correlation(move_near_rt1_sim_factor_diff, rt1_real_factor_diff)", pearson_correlation(move_near_rt1_sim_factor_diff, rt1_real_factor_diff))
print("pearson_correlation(move_near_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)", pearson_correlation(move_near_gengap1234_sim_factor_diff, gengap1234_real_factor_diff))
print("pearson_correlation(move_near_gengap1_sim_factor_diff, gengap1_real_factor_diff)", pearson_correlation(move_near_gengap1_sim_factor_diff, gengap1_real_factor_diff))

avg_rt1_sim_factor_diff = np.mean([coke_can_rt1_sim_factor_diff, move_near_rt1_sim_factor_diff], axis=0)
avg_gengap1234_sim_factor_diff = np.mean([coke_can_gengap1234_sim_factor_diff, move_near_gengap1234_sim_factor_diff], axis=0)
avg_gengap1_sim_factor_diff = np.mean([coke_can_gengap1_sim_factor_diff, move_near_gengap1_sim_factor_diff], axis=0)
print("avg_rt1_sim_factor_diff, rt1_real_factor_diff", avg_rt1_sim_factor_diff, rt1_real_factor_diff)
print("avg_gengap1234_sim_factor_diff, gengap1234_real_factor_diff", avg_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)
print("avg_gengap1_sim_factor_diff, gengap1_real_factor_diff", avg_gengap1_sim_factor_diff, gengap1_real_factor_diff)
print("pearson_correlation(avg_rt1_sim_factor_diff, rt1_real_factor_diff)", pearson_correlation(avg_rt1_sim_factor_diff, rt1_real_factor_diff))
print("pearson_correlation(avg_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)", pearson_correlation(avg_gengap1234_sim_factor_diff, gengap1234_real_factor_diff))
print("pearson_correlation(avg_gengap1_sim_factor_diff, gengap1_real_factor_diff)", pearson_correlation(avg_gengap1_sim_factor_diff, gengap1_real_factor_diff))



print("=" * 60)


# average performance from many simulation variants to select checkpoint and calculate normalized rank loss
print("Average performance from many simulation variants to select checkpoint and calculate normalized rank loss")


# use variants that do not involve cam pose change to select checkpoint (since original cam pose is identical between sim & real)?
horizontal_coke_can_rt1_new_late_avg_variants = np.mean([0.96, 1.00, 1.00, 0.92, 0.96, 1.00, 1.00, 1.00, 0.88]) # 0.04, 0.28])
vertical_coke_can_rt1_new_late_avg_variants = np.mean([0.84, 0.92, 0.76, 0.76, 0.80, 0.84, 0.72, 0.60, 0.44]) # 0.08, 0.00])
standing_coke_can_rt1_new_late_avg_variants = np.mean([0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 1.00, 1.00, 0.92]) # 0.04, 0.56])
horizontal_coke_can_rt1_new_early_avg_variants = np.mean([1.00, 0.96, 0.96, 0.96, 0.92, 1.00, 0.92, 0.88, 0.68]) # 0.08, 0.32])
vertical_coke_can_rt1_new_early_avg_variants = np.mean([0.88, 0.68, 0.56, 0.68, 0.84, 0.72, 0.84, 0.68, 0.84]) # 0.08, 0.00])
standing_coke_can_rt1_new_early_avg_variants = np.mean([0.92, 1.00, 0.92, 0.88, 0.64, 0.92, 0.76, 0.76, 0.52]) # 0.12, 0.40])
horizontal_coke_can_rt1x_late_avg_variants = np.mean([0.64, 0.56, 0.72, 0.60, 0.60, 0.60, 0.84, 0.52, 0.04]) # 0.00, 0.00])
vertical_coke_can_rt1x_late_avg_variants = np.mean([0.32, 0.16, 0.24, 0.20, 0.24, 0.44, 0.16, 0.04, 0.04]) # 0.00, 0.00])
standing_coke_can_rt1x_late_avg_variants = np.mean([0.76, 0.64, 0.80, 0.88, 0.80, 0.64, 0.72, 0.76, 0.28]) # 0.00, 0.04])
horizontal_coke_can_rt1_1k_avg_variants = np.mean([0.04, 0.00, 0.04, 0.00, 0.00, 0.04, 0.00, 0.08, 0.00]) # 0.00, 0.00])
vertical_coke_can_rt1_1k_avg_variants = np.mean([0.04, 0.04, 0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00]) # 0.00, 0.00])
standing_coke_can_rt1_1k_avg_variants = np.mean([0.00, 0.04, 0.08, 0.00, 0.00, 0.04, 0.04, 0.08, 0.00]) # 0.00, 0.04])
print("horizontal_coke_can_rt1_new_late_avg_variants", horizontal_coke_can_rt1_new_late_avg_variants)
print("vertical_coke_can_rt1_new_late_avg_variants", vertical_coke_can_rt1_new_late_avg_variants)
print("standing_coke_can_rt1_new_late_avg_variants", standing_coke_can_rt1_new_late_avg_variants)
print("horizontal_coke_can_rt1_new_early_avg_variants", horizontal_coke_can_rt1_new_early_avg_variants)
print("vertical_coke_can_rt1_new_early_avg_variants", vertical_coke_can_rt1_new_early_avg_variants)
print("standing_coke_can_rt1_new_early_avg_variants", standing_coke_can_rt1_new_early_avg_variants)
print("horizontal_coke_can_rt1x_late_avg_variants", horizontal_coke_can_rt1x_late_avg_variants)
print("vertical_coke_can_rt1x_late_avg_variants", vertical_coke_can_rt1x_late_avg_variants)
print("standing_coke_can_rt1x_late_avg_variants", standing_coke_can_rt1x_late_avg_variants)
print("horizontal_coke_can_rt1_1k_avg_variants", horizontal_coke_can_rt1_1k_avg_variants)
print("vertical_coke_can_rt1_1k_avg_variants", vertical_coke_can_rt1_1k_avg_variants)
print("standing_coke_can_rt1_1k_avg_variants", standing_coke_can_rt1_1k_avg_variants)
horizontal_coke_can_avg_variants = np.array([horizontal_coke_can_rt1_new_late_avg_variants, horizontal_coke_can_rt1_new_early_avg_variants, horizontal_coke_can_rt1x_late_avg_variants, horizontal_coke_can_rt1_1k_avg_variants])
vertical_coke_can_avg_variants = np.array([vertical_coke_can_rt1_new_late_avg_variants, vertical_coke_can_rt1_new_early_avg_variants, vertical_coke_can_rt1x_late_avg_variants, vertical_coke_can_rt1_1k_avg_variants])
standing_coke_can_avg_variants = np.array([standing_coke_can_rt1_new_late_avg_variants, standing_coke_can_rt1_new_early_avg_variants, standing_coke_can_rt1x_late_avg_variants, standing_coke_can_rt1_1k_avg_variants])


coke_can_rt1_new_late_avg_variants = np.mean([horizontal_coke_can_rt1_new_late_avg_variants, vertical_coke_can_rt1_new_late_avg_variants, standing_coke_can_rt1_new_late_avg_variants])
coke_can_rt1_new_early_avg_variants = np.mean([horizontal_coke_can_rt1_new_early_avg_variants, vertical_coke_can_rt1_new_early_avg_variants, standing_coke_can_rt1_new_early_avg_variants])
coke_can_rt1x_late_avg_variants = np.mean([horizontal_coke_can_rt1x_late_avg_variants, vertical_coke_can_rt1x_late_avg_variants, standing_coke_can_rt1x_late_avg_variants])
coke_can_rt1_1k_avg_variants = np.mean([horizontal_coke_can_rt1_1k_avg_variants, vertical_coke_can_rt1_1k_avg_variants, standing_coke_can_rt1_1k_avg_variants])
coke_can_avg_variants = np.array([coke_can_rt1_new_late_avg_variants, coke_can_rt1_new_early_avg_variants, coke_can_rt1x_late_avg_variants, coke_can_rt1_1k_avg_variants])

print("coke_can_rt1_new_late_avg_variants", coke_can_rt1_new_late_avg_variants)
print("coke_can_rt1_new_early_avg_variants", coke_can_rt1_new_early_avg_variants)
print("coke_can_rt1x_late_avg_variants", coke_can_rt1x_late_avg_variants)
print("coke_can_rt1_1k_avg_variants", coke_can_rt1_1k_avg_variants)

coke_can_horizontal_real_success = [0.96, 1.0, 0.88, 0.20]
coke_can_vertical_real_success = [0.88, 0.96, 0.56, 0.00]
coke_can_standing_real_success = [0.72, 0.80, 0.84, 0.20]
coke_can_avg_real = np.mean([coke_can_horizontal_real_success, coke_can_vertical_real_success, coke_can_standing_real_success], axis=0)

print("ranking_violation(horizontal_coke_can_avg_variants, coke_can_horizontal_real_success)", ranking_violation(horizontal_coke_can_avg_variants, coke_can_horizontal_real_success))
print("ranking_violation(vertical_coke_can_avg_variants, coke_can_vertical_real_success)", ranking_violation(vertical_coke_can_avg_variants, coke_can_vertical_real_success))
print("ranking_violation(standing_coke_can_avg_variants, coke_can_standing_real_success)", ranking_violation(standing_coke_can_avg_variants, coke_can_standing_real_success))
print("ranking_violation(coke_can_avg_variants, coke_can_avg_real)", ranking_violation(coke_can_avg_variants, coke_can_avg_real))
print("pearson_corrlation(horizontal_coke_can_avg_variants, coke_can_horizontal_real_success)", pearson_correlation(horizontal_coke_can_avg_variants, coke_can_horizontal_real_success))
print("pearson_correlation(vertical_coke_can_avg_variants, coke_can_vertical_real_success)", pearson_correlation(vertical_coke_can_avg_variants, coke_can_vertical_real_success))
print("pearson_correlation(standing_coke_can_avg_variants, coke_can_standing_real_success)", pearson_correlation(standing_coke_can_avg_variants, coke_can_standing_real_success))
print("pearson_correlation(coke_can_avg_variants, coke_can_avg_real)", pearson_correlation(coke_can_avg_variants, coke_can_avg_real))

print("=" * 60)

move_near_rt1_new_late_avg_variants = np.mean([0.467, 0.483, 0.600, 0.600, 0.533, 0.567, 0.550, 0.200]) # 0.117, 0.433])
move_near_rt1_new_early_avg_variants = np.mean([0.483, 0.483, 0.500, 0.433, 0.500, 0.467, 0.500, 0.200]) # 0.167, 0.217])
move_near_rt1x_avg_variants = np.mean([0.367, 0.333, 0.350, 0.367, 0.433, 0.400, 0.300, 0.033, ]) # 0.050, 0.100])
move_near_rt1_1k_avg_variants = np.mean([0.033, 0.050, 0.033, 0.067, 0.067, 0.017, 0.017, 0.033])

move_near_sim_avg_variants = np.array([move_near_rt1_new_late_avg_variants, move_near_rt1_new_early_avg_variants, move_near_rt1x_avg_variants, move_near_rt1_1k_avg_variants])
move_near_real_success = [0.633, 0.583, 0.45, 0.00]

print("move_near_rt1_new_late_avg_variants", move_near_rt1_new_late_avg_variants)
print("move_near_rt1_new_early_avg_variants", move_near_rt1_new_early_avg_variants)
print("move_near_rt1x_avg_variants", move_near_rt1x_avg_variants)
print("move_near_rt1_1k_avg_variants", move_near_rt1_1k_avg_variants)
print("ranking_violation(move_near_sim_avg_variants, move_near_real_success)", ranking_violation(move_near_sim_avg_variants, move_near_real_success))
print("pearson_correlation(move_near_sim_avg_variants, move_near_real_success)", pearson_correlation(move_near_sim_avg_variants, move_near_real_success))

print("=" * 60)

print("Drawer open pearson correlation", pearson_correlation([0.815,0.704,0.519,0.0], [0.328,0.265,0.102,0.0]))
print("Drawer close pearson correlation", pearson_correlation([0.926,0.889,0.741,0.0], [0.381,0.317,0.487,0.153]))
print("Drawer all pearson correlation", pearson_correlation([0.870,0.796,0.630,0.0], [0.354,0.291,0.295,0.077]))



print("Drawer open greenscreen pearson correlation", pearson_correlation([0.815,0.704,0.519,0.0], [0.667,0.519,0.481,0.0]))
print("Drawer close greenscreen pearson correlation", pearson_correlation([0.926,0.889,0.741,0.0], [0.889,0.556,0.815,0.333]))
print("Drawer all greenscreen pearson correlation", pearson_correlation([0.870,0.796,0.630,0.0], [0.778,0.537,0.648,0.167]))


print("=" * 60)

print("Bridge carrot on plate partial success pearson correlation", pearson_correlation([0.167, 0.500, 0.208], [0.167, 0.333, 0.125]))
print("Bridge carrot on plate pearson correlation", pearson_correlation([0.000, 0.250, 0.083], [0.000, 0.083, 0.000]))
print("Bridge stack cube partial success pearson correlation", pearson_correlation([0.000, 0.292, 0.583], [0.083, 0.208, 0.042]))
print("Bridge stack cube pearson correlation", pearson_correlation([0.000, 0.000, 0.125], [0.000, 0.000, 0.000]))



































































coke_can_horizontal_ctrl2_sim_success = [0.92, 0.92, 1.00]
coke_can_horizontal_real_success = [0.96, 1.0, 0.88]
coke_can_vertical_ctrl2_sim_success = [0.80, 0.76, 0.52]
coke_can_vertical_real_success = [0.88, 0.96, 0.56]
coke_can_standing_ctrl2_sim_success = [0.84, 0.56, 0.80]
coke_can_standing_real_success = [0.72, 0.80, 0.84]

coke_can_avg_ctrl2_sim = np.mean([coke_can_horizontal_ctrl2_sim_success, coke_can_vertical_ctrl2_sim_success, coke_can_standing_ctrl2_sim_success], axis=0)
coke_can_avg_real = np.mean([coke_can_horizontal_real_success, coke_can_vertical_real_success, coke_can_standing_real_success], axis=0)
print("coke_can_average_ctrl2_sim", coke_can_avg_ctrl2_sim)
print("coke_can_average_real", coke_can_avg_real)

print("mean_diff(coke_can_horizontal_ctrl2_sim, coke_can_horizontal_real)", mean_diff(coke_can_horizontal_ctrl2_sim_success, coke_can_horizontal_real_success))
print("pearson_correlation_std_discrep(coke_can_horizontal_ctrl2_sim, coke_can_horizontal_real)", pearson_correlation_std_discrep(coke_can_horizontal_ctrl2_sim_success, coke_can_horizontal_real_success))
print("mean_diff(coke_can_vertical_ctrl2_sim, coke_can_vertical_real)", mean_diff(coke_can_vertical_ctrl2_sim_success, coke_can_vertical_real_success))
print("pearson_correlation_std_discrep(coke_can_vertical_ctrl2_sim, coke_can_vertical_real)", pearson_correlation_std_discrep(coke_can_vertical_ctrl2_sim_success, coke_can_vertical_real_success))
print("mean_diff(coke_can_standing_ctrl2_sim, coke_can_standing_real)", mean_diff(coke_can_standing_ctrl2_sim_success, coke_can_standing_real_success))
print("pearson_correlation_std_discrep(coke_can_standing_ctrl2_sim, coke_can_standing_real)", pearson_correlation_std_discrep(coke_can_standing_ctrl2_sim_success, coke_can_standing_real_success))
print("mean_diff(coke_can_avg_ctrl2_sim, coke_can_avg_real)", mean_diff(coke_can_avg_ctrl2_sim, coke_can_avg_real))
print("pearson_correlation_std_discrep(coke_can_avg_ctrl2_sim, coke_can_avg_real)", pearson_correlation_std_discrep(coke_can_avg_ctrl2_sim, coke_can_avg_real))



move_near_sim_ctrl2_success = [0.30, 0.317, 0.217, 0.027]
move_near_real_success = [0.633, 0.583, 0.45, 0.133]

print("mean_diff(move_near_sim_ctrl2_success, move_near_real_success)", mean_diff(move_near_sim_ctrl2_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_ctrl2_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_ctrl2_success, move_near_real_success))
print("ranking_violation(move_near_sim_ctrl2_success, move_near_real_success)", ranking_violation(move_near_sim_ctrl2_success, move_near_real_success))

print("=" * 60)





coke_can_horizontal_ctrl4_sim_success = [0.92, 0.84, 0.88]
coke_can_horizontal_real_success = [0.96, 1.0, 0.88]
coke_can_vertical_ctrl4_sim_success = [0.68, 0.60, 0.44]
coke_can_vertical_real_success = [0.88, 0.96, 0.56]
coke_can_standing_ctrl4_sim_success = [0.76, 0.44, 0.72]
coke_can_standing_real_success = [0.72, 0.80, 0.84]

coke_can_avg_ctrl4_sim = np.mean([coke_can_horizontal_ctrl4_sim_success, coke_can_vertical_ctrl4_sim_success, coke_can_standing_ctrl4_sim_success], axis=0)
coke_can_avg_real = np.mean([coke_can_horizontal_real_success, coke_can_vertical_real_success, coke_can_standing_real_success], axis=0)
print("coke_can_average_ctrl4_sim", coke_can_avg_ctrl4_sim)
print("coke_can_average_real", coke_can_avg_real)

print("mean_diff(coke_can_horizontal_ctrl4_sim, coke_can_horizontal_real)", mean_diff(coke_can_horizontal_ctrl4_sim_success, coke_can_horizontal_real_success))
print("pearson_correlation_std_discrep(coke_can_horizontal_ctrl4_sim, coke_can_horizontal_real)", pearson_correlation_std_discrep(coke_can_horizontal_ctrl4_sim_success, coke_can_horizontal_real_success))
print("mean_diff(coke_can_vertical_ctrl4_sim, coke_can_vertical_real)", mean_diff(coke_can_vertical_ctrl4_sim_success, coke_can_vertical_real_success))
print("pearson_correlation_std_discrep(coke_can_vertical_ctrl4_sim, coke_can_vertical_real)", pearson_correlation_std_discrep(coke_can_vertical_ctrl4_sim_success, coke_can_vertical_real_success))
print("mean_diff(coke_can_standing_ctrl4_sim, coke_can_standing_real)", mean_diff(coke_can_standing_ctrl4_sim_success, coke_can_standing_real_success))
print("pearson_correlation_std_discrep(coke_can_standing_ctrl4_sim, coke_can_standing_real)", pearson_correlation_std_discrep(coke_can_standing_ctrl4_sim_success, coke_can_standing_real_success))
print("mean_diff(coke_can_avg_ctrl4_sim, coke_can_avg_real)", mean_diff(coke_can_avg_ctrl4_sim, coke_can_avg_real))
print("pearson_correlation_std_discrep(coke_can_avg_ctrl4_sim, coke_can_avg_real)", pearson_correlation_std_discrep(coke_can_avg_ctrl4_sim, coke_can_avg_real))



move_near_sim_ctrl4_success = [0.217, 0.350, 0.183, 0.027]
move_near_real_success = [0.633, 0.583, 0.45, 0.133]

print("mean_diff(move_near_sim_ctrl4_success, move_near_real_success)", mean_diff(move_near_sim_ctrl4_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_ctrl4_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_ctrl4_success, move_near_real_success))
print("ranking_violation(move_near_sim_ctrl4_success, move_near_real_success)", ranking_violation(move_near_sim_ctrl4_success, move_near_real_success))

print("=" * 60)





coke_can_horizontal_ctrl5_sim_success = [1.00, 0.88, 0.36]
coke_can_horizontal_real_success = [0.96, 1.0, 0.88]
coke_can_vertical_ctrl5_sim_success = [0.80, 0.84, 0.04]
coke_can_vertical_real_success = [0.88, 0.96, 0.56]
coke_can_standing_ctrl5_sim_success = [0.76, 0.44, 0.72]
coke_can_standing_real_success = [0.72, 0.80, 0.84]


coke_can_avg_ctrl5_sim = np.mean([coke_can_horizontal_ctrl5_sim_success, coke_can_vertical_ctrl5_sim_success, coke_can_standing_ctrl5_sim_success], axis=0)
coke_can_avg_real = np.mean([coke_can_horizontal_real_success, coke_can_vertical_real_success, coke_can_standing_real_success], axis=0)
print("coke_can_average_ctrl5_sim", coke_can_avg_ctrl5_sim)
print("coke_can_average_real", coke_can_avg_real)

print("mean_diff(coke_can_horizontal_ctrl5_sim, coke_can_horizontal_real)", mean_diff(coke_can_horizontal_ctrl5_sim_success, coke_can_horizontal_real_success))
print("pearson_correlation_std_discrep(coke_can_horizontal_ctrl5_sim, coke_can_horizontal_real)", pearson_correlation_std_discrep(coke_can_horizontal_ctrl5_sim_success, coke_can_horizontal_real_success))
print("mean_diff(coke_can_vertical_ctrl5_sim, coke_can_vertical_real)", mean_diff(coke_can_vertical_ctrl5_sim_success, coke_can_vertical_real_success))
print("pearson_correlation_std_discrep(coke_can_vertical_ctrl5_sim, coke_can_vertical_real)", pearson_correlation_std_discrep(coke_can_vertical_ctrl5_sim_success, coke_can_vertical_real_success))
print("mean_diff(coke_can_standing_ctrl5_sim, coke_can_standing_real)", mean_diff(coke_can_standing_ctrl5_sim_success, coke_can_standing_real_success))
print("pearson_correlation_std_discrep(coke_can_standing_ctrl5_sim, coke_can_standing_real)", pearson_correlation_std_discrep(coke_can_standing_ctrl5_sim_success, coke_can_standing_real_success))
print("mean_diff(coke_can_avg_ctrl5_sim, coke_can_avg_real)", mean_diff(coke_can_avg_ctrl5_sim, coke_can_avg_real))
print("pearson_correlation_std_discrep(coke_can_avg_ctrl5_sim, coke_can_avg_real)", pearson_correlation_std_discrep(coke_can_avg_ctrl5_sim, coke_can_avg_real))



move_near_sim_ctrl5_success = [0.250, 0.367, 0.183, 0.027]
move_near_real_success = [0.633, 0.583, 0.45, 0.133]

print("mean_diff(move_near_sim_ctrl5_success, move_near_real_success)", mean_diff(move_near_sim_ctrl5_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_ctrl5_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_ctrl5_success, move_near_real_success))
print("ranking_violation(move_near_sim_ctrl5_success, move_near_real_success)", ranking_violation(move_near_sim_ctrl5_success, move_near_real_success))

print("=" * 60)
