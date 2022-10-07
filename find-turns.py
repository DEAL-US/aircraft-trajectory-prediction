from utils import *
import pickle

number_points = [5,6,7,8,9,10]
features_x = [0,1,3]
features_y = [0,1,3]
f_diff = [0,1,3] 
time_intervals = [30,60,90,120]
num_predictions = 10

for nop in number_points:

    seq_size = nop
    if(len(f_diff)>0):
        seq_size -= 1

    all_xs = []
    all_ys = []
    all_ids = []

    for time_interval in time_intervals:
        folder_path = f"./training-data/training-data-{time_interval}/{nop}" 
        xs_path = f"{folder_path}/xs.npy"
        ys_path = f"{folder_path}/ys.npy"

        xs = np.load(xs_path)
        print(len(xs))
        ys = np.load(ys_path)
        ids = [f"{time_interval}-{nop}-{i}" for i in range(len(xs))]

        all_xs.append(xs)
        all_ys.append(ys)
        all_ids.extend(ids)
    
    xs:np.ndarray = np.concatenate(all_xs)
    ys:np.ndarray = np.concatenate(all_ys)
    ids = np.array(all_ids)

    pos = find_positions_without_absent_values(xs, ys, features_x)
    xs = xs[pos]
    ys = ys[pos]
    ids = ids[pos]

    pos = find_positions_without_redoudant_vectors(xs, ys, features_x, compute_diffs=True)
    xs = xs[pos]
    ys = ys[pos]
    ids = ids[pos]

    originals, xs_diff, ys_diff = to_diff(xs, ys, features_x, return_diffs_only=True)

    pos = find_positions_without_anomalous_values(xs_diff, ys_diff)
    xs = xs[pos]
    ys = ys[pos]
    xs_diff = xs_diff[pos]
    ys_diff = ys_diff[pos]
    ids = ids[pos]

    accs = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        acc = compute_flight_accumulated_rotation(y, absolute_input=True, feat_indices=[0,1])
        accs.append(acc)

    stats_df = pd.DataFrame({"accs": accs, "ids": ids})
    stats_df = stats_df.sort_values(by="accs", ascending=False)
    ids = list(stats_df.iloc[0:int(len(stats_df)/10)]["ids"])
    turns_file = f'./training-data/training-data-turns/{nop}'
    with open(turns_file, 'wb') as f:
        pickle.dump(ids, f)
    print("done")