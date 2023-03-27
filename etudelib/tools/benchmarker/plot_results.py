import pickle
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
plt.rcParams['pdf.fonttype'] = 42
plt.rc('text', usetex=True)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"


def read_results(result_path):
    with open(result_path, 'rb') as file:
        results = pickle.load(file)
    return results


def get_results(modelname, runtime, device, datasource):
    result_path = '../../projects/benchmark/results'
    result = {}
    for C in [1000, 10000, 100000, 1000000, 10000000]:
        filename = f'{modelname}_{runtime}_{device}_{datasource}_C{C}_t50_results.pickle'
        f = os.path.join(result_path, filename)
        if os.path.isfile(f):
            r = read_results(f)
            result[C] = r['latency_df']
    return result


def run():
    model_names = ['core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'repeatnet', 'sasrec', 'sine', 'srgnn',
                   'stamp']
    # model_names = ['narm']
    colors = ['b', 'g', 'r', 'w', 'c', 'm', 'y', 'k', ]
    markers = ['D', '+', 'x', 'D']
    lss = ['--', '-.', ':', '-']
    for model_name in model_names:
        # runtimes = ['eager', 'jit', 'jitopt', 'onnx']
        runtimes = ['eager', 'jitopt', 'onnx']
        devices = ['cpu', 'cuda']
        datasource = 'bolcom'
        fig, ax = plt.subplots(figsize=(16, 12))

        for runtime in runtimes:
            for device in devices:
                results = get_results(model_name, runtime, device, datasource)
                if len(results) > 0:
                    q90s = []
                    cs = []
                    for C, latency_df in results.items():
                        filtered_df = latency_df[
                            latency_df['DateTime'] >= (latency_df['DateTime'].min() + pd.Timedelta(seconds=5))]
                        q90 = np.percentile(filtered_df['LatencyInMs'], q=[0.9])
                        q90s.append(q90)
                        cs.append(C)
                    color = colors[devices.index(device)]
                    marker = markers[runtimes.index(runtime)]
                    ls = lss[runtimes.index(runtime)]
                    plt.plot(cs, q90s, color=color, marker=marker, label=f'{runtime} {device}', linestyle=ls, alpha=0.7)

        plt.title(f'Inference latency for {model_name}')
        # Get the matplotlib axis object
        ax = plt.gca()
        # Set the x-axis limits
        ax.set_ylim([0.07, 1e4])
        plt.axhline(y=50, color='r', label='50ms')
        ax.legend(loc=2, fontsize=20, ncol=2)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=24)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.ylabel('Latency (ms) p90')
        plt.xlabel('C')
        plt.savefig(f'{model_name}_{datasource}.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    run()
