import os
import argparse
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto Running")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='ogbn-arxiv')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    args = parser.parse_args()

    for k in [1, 2, 3, 4, 5]:
        for noise_type in ["symmetric", "asymmetric"]:
            for r in range(args.run):
                print("Start run {}: {}+{}".format(r, k, noise_type))
                gpu = 0
                while True:
                    time.sleep(10)
                    p, m = gpu_info(gpu)
                    if m <= 5000:
                        break
                    else:
                        gpu += 1
                        gpu = gpu % 4
                mislabel_result_file = 'mislabel_results/validl1-laplacian-test=test-MLP-{}-GCN-mislabel={}-{}-sample=0.5-k={}-epochs=200-' \
                                       'lr=0.001-wd=0.0005-exp={}.csv'.format(args.dataset, args.mislabel_rate, noise_type, k, r)
                if os.path.exists(mislabel_result_file):
                    print(mislabel_result_file, " already existed!")
                    continue
                os.system("CUDA_VISIBLE_DEVICES={} python run_CL_negsamp.py --dataset {} --mislabel_rate {} --noise_type {} --k {} --exp {} &".format(
                    gpu, args.dataset, args.mislabel_rate, noise_type, k, r))
