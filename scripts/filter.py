import argparse
from calc_sim import load_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("output_dir")
    parser.add_argument("--theta", default=0.5, help="threshold")
    args = parser.parse_args()

    files_dict = load_files(args.path)

    for sys_name, f in files_dict.items():
        result = []
        for line in f:
            ref, prob = line.rstrip().split("\t")
            if float(prob) > float(args.theta):
                result.append(ref)
            else:
                result.append(" ")

        with open(f'{args.output_dir}/{sys_name}', 'w') as f:
            for s in result:
                f.write(str(s) + "\n")


if __name__ == '__main__':
    main()
