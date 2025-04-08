f1 = open('./perfs/features.txt', 'r').read()

name_by_id = {}
k_by_id = {}
for line in f1.split('\n'):
    if 'key' not in line:
        continue
    kk = line.split('key=')[1]
    if kk not in k_by_id:
        k_by_id[kk] = 0
    k_by_id[kk] += 1
    name_by_id[int(line.split(', ')[0])] = kk.split(',')[0] + '_' + str(k_by_id[kk])


f2 = open('./toad_fork/models/20_02_new_with_ticher_and_5_frozen_cont_3/benchmark_results_maskout.txt', 'r').read()
f2 = reversed(f2.split('\n'))
if True:
    for line in f2:
        if 'maskout' not in line or 'baseline_diffs' not in line:
            continue
        id = int(line.split('maskout: ')[1].split(',')[0])
        if id not in name_by_id:
            continue
        bd = float(line.split('baseline_diffs: ')[1])
        wr = float(line.split(' %    maskout:')[0].split(' ')[-1])

        if bd == 0 and wr == 49.0:
            print("NOT AFFECTING: ", id, name_by_id[id], wr, bd)
            continue
        else:
            print(".   AFFECTING: ", id, name_by_id[id], wr, bd)

        #print(name_by_id[id], wr, bd)