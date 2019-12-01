import os
import shutil

root_dir = '../outputs'
destination_dir = '../collected-outputs'

if __name__ == '__main__':
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
