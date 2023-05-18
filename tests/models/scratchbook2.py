#%%

from pathlib import Path
root_path = Path(__file__).parents[2]
result_path = root_path.joinpath('projects/microbenchmark/results')
#%%
notfound = []
for C in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000]:
    for model_name in ['gcsan', 'gru4rec', 'lightsans', 'narm', 'repeatnet', 'sasrec',
                        'sine', 'srgnn', 'stamp', 'random', 'core']:
        for topk in ['mm+topk', 'faiss']:
            for t in [50]:
                for param_source in ['bolcom']:
                    for runtime in ["eager", "jitopt"]:
                        for device in ["cpu", "cuda"]:
                            filename = f"{model_name}_{runtime}_{device}_{topk}_{param_source}_C{C}_t{t}_results.pickle"
                            skip = (topk == 'faiss') & (C > 10_000_000) & (device == 'cuda') \
                                   or (topk == 'faiss') & (runtime == 'jitopt') \
                                   or (model_name == 'repeatnet')
                            if not result_path.joinpath(filename).is_file() and not skip:
                                notfound.append(filename)
