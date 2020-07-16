import os

from common.const import default_cache_dir
from diskcache import Cache
from encoder.utils import get_imlist


def feature_extract(database_path, model):
    cache = Cache(default_cache_dir)
    feats = []
    names = []
    img_list = get_imlist(database_path)
    model = model
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name.encode())
        current = i+1
        total = len(img_list)
        cache['current'] = current
        cache['total'] = total
        print("extracting feature from image No. %d , %d images in total" %
              (current, total))
#    feats = np.array(feats)
    return feats, names
