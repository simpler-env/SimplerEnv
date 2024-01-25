import numpy as np

def pearson_correlation_std_discrep(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    x = x - np.mean(x)
    y = y - np.mean(y)
    if np.all(x == y):
        pearson = 1
    else:
        pearson = np.sum(x * y) / (np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-8)
    discrep = (1 - pearson) * max(np.std(x), np.std(y))
    return discrep

def mean_diff(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    return np.abs(np.mean(x) - np.mean(y))

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
print("mean_diff(coke_can_vertical_sim, coke_can_vertical_real)", mean_diff(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("pearson_correlation_std_discrep(coke_can_vertical_sim, coke_can_vertical_real)", pearson_correlation_std_discrep(coke_can_vertical_sim_success, coke_can_vertical_real_success))
print("mean_diff(coke_can_standing_sim, coke_can_standing_real)", mean_diff(coke_can_standing_sim_success, coke_can_standing_real_success))
print("pearson_correlation_std_discrep(coke_can_standing_sim, coke_can_standing_real)", pearson_correlation_std_discrep(coke_can_standing_sim_success, coke_can_standing_real_success))
print("mean_diff(coke_can_avg_sim, coke_can_avg_real)", mean_diff(coke_can_avg_sim, coke_can_avg_real))
print("pearson_correlation_std_discrep(coke_can_avg_sim, coke_can_avg_real)", pearson_correlation_std_discrep(coke_can_avg_sim, coke_can_avg_real))

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
