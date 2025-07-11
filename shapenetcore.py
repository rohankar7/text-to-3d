import numpy as np
import os
import config

# ShapeNetCore_classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243', '04401088', '04460130', '04468005', '04530566', '04554684']
np.random.seed(config.random_seed)

def get_random_models():
    random_models = []
    sample_size = config.sample_size
    for classes in os.listdir(config.directory):
        class_dir = os.path.join(config.directory, classes)
        models = []
        for model_name in os.listdir(class_dir):
            model_dir = os.path.join(class_dir, model_name)
            if os.path.isfile(f'{model_dir}/{config.suffix_dir}'):
                models.append(f'{classes}/{model_name}')
        random_models.append(np.random.choice(models, size=sample_size, replace=False))
    random_models = np.array(random_models).reshape(-1)
    return random_models