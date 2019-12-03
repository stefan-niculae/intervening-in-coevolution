import os
import shutil
import sys
import re


KEEP_CHECKPOINTS = [300, 500, 750]

if __name__ == '__main__':
    try:
        root_dir = sys.argv[1]
        destination_dir = sys.argv[2]
        dry_run = len(sys.argv) > 3 and sys.argv[3] == '-n'
    except:
        print('usage: collect_checkpoints <source_root_dir> <destination_dir> [-n]')

    os.makedirs(destination_dir, exist_ok=True)

    for run_dir_name in os.listdir(root_dir):
        run_dir_path = root_dir + '/' + run_dir_name
        if not os.path.isdir(run_dir_path):
            continue

        models_path = run_dir_path + '/trained_models/'
        for checkpoint_name in os.listdir(models_path):
            checkpoint_path = models_path + '/' + checkpoint_name

            checkpoint_number = int(re.findall('-(\d+)', checkpoint_name)[0])
            if checkpoint_number not in KEEP_CHECKPOINTS:
                continue

            if dry_run:
                print(run_dir_name + ' -- ' + checkpoint_name)

            else:
                shutil.copyfile(
                    checkpoint_path,
                    destination_dir + '/' + run_dir_name + ' -- ' + checkpoint_name
                )
