def print_results(results: dict):
    for k, v in results.items():
        print(f'===== {k} =====')
        for metric, val in v.items():
            print(f'  {metric}: {val}')
