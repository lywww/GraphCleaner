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
    parser.add_argument("--dataset", type=str, default='Cora')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    args = parser.parse_args()

    f2m = dict(zip(["run_CL_negsamp.py", "run_baseline.py", "run_DY-bootstrap.py", "run_AUM.py", "run_Confident_Learning.py"], ['ours', 'baseline', 'DYB', 'AUM', 'CL']))

    for dataset in ["ogbn-arxiv", "Cora", "CiteSeer", "PubMed", "Computers", "Photo"]:
        for noise_type in ["symmetric", "asymmetric"]:
            for mislabel_rate in [0.1, 0.05, 0.025]:
                for r in range(args.run):
        # gpu0, gpu1, gpu2, gpu3 = False, False, False, False
        # while not (gpu0 and gpu1 and gpu2 and gpu3):
        #     if not gpu0:
        #         p, m = gpu_info(0)
        #         # print("gpu0 power {} memory {}".format(p, m))
        #         gpu0 = m <= 5000
        #     if not gpu1:
        #         p, m = gpu_info(1)
        #         # print("gpu1 power {} memory {}".format(p, m))
        #         gpu1 = m <= 5000
        #     if not gpu2:
        #         p, m = gpu_info(2)
        #         # print("gpu2 power {} memory {}".format(p, m))
        #         gpu2 = m <= 5000
        #     if not gpu3:
        #         p, m = gpu_info(3)
        #         # print("gpu3 power {} memory {}".format(p, m))
        #         gpu3 = m <= 5000
        #     time.sleep(5)
                    print("Start run {}: {}+{}+{}".format(r, dataset, noise_type, mislabel_rate))
                    gpu = 0
                    for method in ["run_CL_negsamp.py", "run_DY-bootstrap.py", "run_AUM.py", "run_baseline.py", "run_Confident_Learning.py"]:
                        while True:
                            time.sleep(5)
                            p, m = gpu_info(gpu)
                            if m <= 5000:
                                break
                            else:
                                gpu += 1
                                gpu = gpu % 4
                        if method == "run_CL_negsamp.py":
                            mislabel_result_file = 'mislabel_results/validl1-laplacian-test=test-MLP-{}-{}-mislabel={}-{}-sample=0.5-k=3-epochs=200-' \
                                                   'lr=0.001-wd=0.0005-exp={}.csv'.format(dataset, args.model, mislabel_rate, noise_type, r)
                            if os.path.exists(mislabel_result_file):
                                print(mislabel_result_file, " already existed!")
                                continue
                            os.system("CUDA_VISIBLE_DEVICES={} python {} --dataset {} --mislabel_rate {} --noise_type {} --model {} --exp {}".format(
                                gpu, method, dataset, mislabel_rate, noise_type, args.model, r))
                        else:
                            mislabel_result_file = 'mislabel_results/{}-test=test-{}-{}-mislabel={}-{}-epochs=200-lr=0.001-wd=0.0005-exp={}.csv'.format(
                                f2m[method], dataset, args.model, mislabel_rate, noise_type, r)
                            if os.path.exists(mislabel_result_file):
                                print(mislabel_result_file, " already existed!")
                                continue
                            os.system(
                                "CUDA_VISIBLE_DEVICES={} python {} --dataset {} --mislabel_rate {} --noise_type {} --model {} --exp {} &".format(
                                    gpu, method, dataset, mislabel_rate, noise_type, args.model, r))
