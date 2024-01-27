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
                rank_violation += np.abs(y[i] - y[j])
    return rank_violation / (len(x) * (len(x) - 1) / 2)

rt_ckpts = ['rt-1-new-best-late', 'rt-1-new-early', 'rt-1-x']
coke_can_horizontal_sim_success = [1.0, 0.84, 0.80]
coke_can_horizontal_real_success = [0.96, 1.0, 0.88]
coke_can_vertical_sim_success = [0.80, 0.72, 0.48]
coke_can_vertical_real_success = [0.88, 0.96, 0.56]
coke_can_standing_sim_success = [0.80, 0.60, 0.68]
coke_can_standing_real_success = [0.72, 0.80, 0.84]

coke_can_avg_sim = np.mean([coke_can_horizontal_sim_success, coke_can_vertical_sim_success, coke_can_standing_sim_success], axis=0)
coke_can_avg_real = np.mean([coke_can_horizontal_real_success, coke_can_vertical_real_success, coke_can_standing_real_success], axis=0)
print("coke_can_average_sim", coke_can_avg_sim)
print("coke_can_average_real", coke_can_avg_real)

print("mean_diff(coke_can_horizontal_sim, coke_can_horizontal_real)", mean_diff(coke_can_horizontal_sim_success, coke_can_horizontal_real_success))
print("pearson_correlation_std_discrep(coke_can_horizontal_sim, coke_can_horizontal_real)", pearson_correlation_std_discrep(coke_can_horizontal_sim_success, coke_can_horizontal_real_success))
print("ranking_violation(coke_can_horizontal_sim, coke_can_horizontal_real)", ranking_violation(coke_can_horizontal_sim_success, coke_can_horizontal_real_success))
print("mean_diff(coke_can_vertical_sim, coke_can_vertical_real)", mean_diff(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("pearson_correlation_std_discrep(coke_can_vertical_sim, coke_can_vertical_real)", pearson_correlation_std_discrep(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("ranking_violation(coke_can_vertical_sim, coke_can_vertical_real)", ranking_violation(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("mean_diff(coke_can_standing_sim, coke_can_standing_real)", mean_diff(coke_can_standing_sim_success, coke_can_standing_real_success))
print("pearson_correlation_std_discrep(coke_can_standing_sim, coke_can_standing_real)", pearson_correlation_std_discrep(coke_can_standing_sim_success, coke_can_standing_real_success))
print("ranking_violation(coke_can_standing_sim, coke_can_standing_real)", ranking_violation(coke_can_standing_sim_success, coke_can_standing_real_success))
print("mean_diff(coke_can_avg_sim, coke_can_avg_real)", mean_diff(coke_can_avg_sim, coke_can_avg_real))
print("pearson_correlation_std_discrep(coke_can_avg_sim, coke_can_avg_real)", pearson_correlation_std_discrep(coke_can_avg_sim, coke_can_avg_real))
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

move_near_sim_success = [0.30, 0.367, 0.35]
move_near_real_success = [0.633, 0.583, 0.45]

print("mean_diff(move_near_sim_success, move_near_real_success)", mean_diff(move_near_sim_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_success, move_near_real_success))
print("ranking_violation(move_near_sim_success, move_near_real_success)", ranking_violation(move_near_sim_success, move_near_real_success))

print("=" * 60)

# [base success, background, lighting, distractor, table tex, cam pose]
coke_can_rt1_new_late_sim_table_avg = np.mean(
    [[0.96, 1.00, 1.00, 0.92, 0.88, 0.04],
     [0.84, 0.84, 0.92, 0.76, 0.44, 0.08],
     [0.96, 0.96, 0.96, 0.96, 0.92, 0.04]],
    axis=0
)
coke_can_gengap1234_sim_table_avg = np.mean(
    [[0.92, 0.80, 0.96, 0.76, 0.92, 0.16],
     [0.84, 0.56, 0.76, 0.52, 0.52, 0.04],
     [0.96, 0.76, 0.92, 0.88, 0.72, 0.36]],
    axis=0
)
coke_can_gengap1_sim_table_avg = np.mean(
    [[0.96, 0.84, 1.00, 0.88, 0.64, 0.20],
     [0.68, 0.60, 0.76, 0.68, 0.16, 0.00],
     [0.76, 0.80, 0.52, 0.88, 0.68, 0.12]],
    axis=0
)

print("coke_can_rt1_new_late_sim_table_avg", coke_can_rt1_new_late_sim_table_avg)
print("coke_can_gengap1234_sim_table_avg", coke_can_gengap1234_sim_table_avg)
print("coke_can_gengap1_sim_table_avg", coke_can_gengap1_sim_table_avg)

coke_can_rt1_sim_factor_diff = np.abs(coke_can_rt1_new_late_sim_table_avg[1:] - coke_can_rt1_new_late_sim_table_avg[0])
coke_can_gengap1234_sim_factor_diff = np.abs(coke_can_gengap1234_sim_table_avg[1:] - coke_can_gengap1234_sim_table_avg[0])
coke_can_gengap1_sim_factor_diff = np.abs(coke_can_gengap1_sim_table_avg[1:] - coke_can_gengap1_sim_table_avg[0])

move_near_rt1_new_late_sim_table_avg = np.array([0.467, 0.533, 0.483, 0.600, 0.200, 0.117])
move_near_gengap1234_sim_table_avg = np.array([0.267, 0.283, 0.300, 0.367, 0.150, 0.100])
move_near_gengap1_sim_table_avg = np.array([0.383, 0.483, 0.517, 0.467, 0.133, 0.200])

move_near_rt1_sim_factor_diff = np.abs(move_near_rt1_new_late_sim_table_avg[1:] - move_near_rt1_new_late_sim_table_avg[0])
move_near_gengap1234_sim_factor_diff = np.abs(move_near_gengap1234_sim_table_avg[1:] - move_near_gengap1234_sim_table_avg[0])
move_near_gengap1_sim_factor_diff = np.abs(move_near_gengap1_sim_table_avg[1:] - move_near_gengap1_sim_table_avg[0])





rt1_real_avg = np.array([0.9167, 0.8888, 0.8333, 0.8056, 0.5278, 0.4583])
gengap1234_real_avg = np.array([0.8333, 0.7222, 0.7500, 0.4444, 0.7500, 0.5000])
gengap1_real_avg = np.array([0.8333, 0.6667, 0.7917, 0.7500, 0.6667, 0.4583])
rt1_real_factor_diff = np.abs(rt1_real_avg[1:] - rt1_real_avg[0])
gengap1234_real_factor_diff = np.abs(gengap1234_real_avg[1:] - gengap1234_real_avg[0])
gengap1_real_factor_diff = np.abs(gengap1_real_avg[1:] - gengap1_real_avg[0])

print("pearson_correlation(coke_can_rt1_sim_factor_diff, rt1_real_factor_diff)", pearson_correlation(coke_can_rt1_sim_factor_diff, rt1_real_factor_diff))
print("pearson_correlation(coke_can_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)", pearson_correlation(coke_can_gengap1234_sim_factor_diff, gengap1234_real_factor_diff))
print("pearson_correlation(coke_can_gengap1_sim_factor_diff, gengap1_real_factor_diff)", pearson_correlation(coke_can_gengap1_sim_factor_diff, gengap1_real_factor_diff))
print("pearson_correlation(move_near_rt1_sim_factor_diff, rt1_real_factor_diff)", pearson_correlation(move_near_rt1_sim_factor_diff, rt1_real_factor_diff))
print("pearson_correlation(move_near_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)", pearson_correlation(move_near_gengap1234_sim_factor_diff, gengap1234_real_factor_diff))
print("pearson_correlation(move_near_gengap1_sim_factor_diff, gengap1_real_factor_diff)", pearson_correlation(move_near_gengap1_sim_factor_diff, gengap1_real_factor_diff))

avg_rt1_sim_factor_diff = np.mean([coke_can_rt1_sim_factor_diff, move_near_rt1_sim_factor_diff], axis=0)
avg_gengap1234_sim_factor_diff = np.mean([coke_can_gengap1234_sim_factor_diff, move_near_gengap1234_sim_factor_diff], axis=0)
avg_gengap1_sim_factor_diff = np.mean([coke_can_gengap1_sim_factor_diff, move_near_gengap1_sim_factor_diff], axis=0)
print("pearson_correlation(avg_rt1_sim_factor_diff, rt1_real_factor_diff)", pearson_correlation(avg_rt1_sim_factor_diff, rt1_real_factor_diff))
print("pearson_correlation(avg_gengap1234_sim_factor_diff, gengap1234_real_factor_diff)", pearson_correlation(avg_gengap1234_sim_factor_diff, gengap1234_real_factor_diff))
print("pearson_correlation(avg_gengap1_sim_factor_diff, gengap1_real_factor_diff)", pearson_correlation(avg_gengap1_sim_factor_diff, gengap1_real_factor_diff))

print("=" * 60)
















































































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



move_near_sim_ctrl2_success = [0.30, 0.317, 0.217]
move_near_real_success = [0.633, 0.583, 0.45]

print("mean_diff(move_near_sim_ctrl2_success, move_near_real_success)", mean_diff(move_near_sim_ctrl2_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_ctrl2_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_ctrl2_success, move_near_real_success))

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



move_near_sim_ctrl4_success = [0.217, 0.350, 0.183]
move_near_real_success = [0.633, 0.583, 0.45]

print("mean_diff(move_near_sim_ctrl4_success, move_near_real_success)", mean_diff(move_near_sim_ctrl4_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_ctrl4_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_ctrl4_success, move_near_real_success))

print("=" * 60)





coke_can_horizontal_ctrl5_sim_success = [1.00, 0.88, 0.36]
coke_can_horizontal_real_success = [0.96, 1.0, 0.88]
coke_can_vertical_ctrl5_sim_success = [0.80, 0.84, 0.04]
coke_can_vertical_real_success = [0.88, 0.96, 0.56]
coke_can_standing_ctrl5_sim_success = [0.76, 0.44, 0.72]
coke_can_standing_real_success = [0.68, 0.52, 0.04]

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



move_near_sim_ctrl5_success = [0.250, 0.367, 0.183]
move_near_real_success = [0.633, 0.583, 0.45]

print("mean_diff(move_near_sim_ctrl5_success, move_near_real_success)", mean_diff(move_near_sim_ctrl5_success, move_near_real_success))
print("pearson_correlation_std_discrep(move_near_sim_ctrl5_success, move_near_real_success)", pearson_correlation_std_discrep(move_near_sim_ctrl5_success, move_near_real_success))

print("=" * 60)
