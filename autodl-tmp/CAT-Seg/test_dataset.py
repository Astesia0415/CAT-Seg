from cat_seg.data.datasets.register_pest import register_all_pest
from detectron2.data import DatasetCatalog


register_all_pest()


data = DatasetCatalog.get("pest_train")


print("图片数量:", len(data))

print(data[0])
