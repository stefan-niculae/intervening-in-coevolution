import os
import shutil
import sys


if __name__ == '__main__':
    try:
        root_dir = sys.argv[1]
        destination_dir = sys.argv[2]
    except:
        print('usage: collect_checkpoints <source_root_dir> <destination_dir>')

    os.makedirs(destination_dir, exist_ok=True)

    for run_dir_name in os.listdir(root_dir):
        run_dir_path = root_dir + '/' + run_dir_name
        if not os.path.isdir(run_dir_path):
            continue

        models_path = run_dir_path + '/trained_models/'
        for checkpoint_name in os.listdir(models_path):
            checkpoint_path = models_path + '/' + checkpoint_name

            shutil.copyfile(
                checkpoint_path,
                destination_dir + '/' + run_dir_name + ' -- ' + checkpoint_name
            )
