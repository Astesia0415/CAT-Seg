from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


_PREDEFINED_SPLITS_PEST = {
    "pest_train": (
        "/root/autodl-tmp/dataset5/CATSeg_dataset/images/train",
        "/root/autodl-tmp/dataset5/CATSeg_dataset/masks/train",
    ),

    "pest_val": (
        "/root/autodl-tmp/dataset5/CATSeg_dataset/images/val",
        "/root/autodl-tmp/dataset5/CATSeg_dataset/masks/val",
    ),

    "pest_test": (
        "/root/autodl-tmp/dataset5/CATSeg_dataset/images/test",
        "/root/autodl-tmp/dataset5/CATSeg_dataset/masks/test",
    ),
}


def register_all_pest(root=None):

    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS_PEST.items():

        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir:
            load_sem_seg(
                y,
                x,
                gt_ext="png",
                image_ext="jpg"
            )
        )


        MetadataCatalog.get(name).set(
            ignore_label=255,
            evaluator_type="sem_seg",
            stuff_classes=[
                "background",
                "Aleurocanthus spiniferus",
                "Ceroplastes rubens",
                "Icerya purchasi Maskell",
                "Limacodidae",
                "Locustoidea",
                "Potosiabre vitarsis",
                "alfalfa plant bug",
                "aphids",
                "apple black rot",
                "banana anthracnose",
                "banana black leaf streak",
                "bean halo blight",
                "bell pepper blossom end rot",
                "citrus canker",
                "corn gray leaf spot",
                "corn rust",
                "corn smut",
                "garlic rust",
                "legume blister beetle",
                "oides decempunctata",
                "rice blast",
                "tarnished plant bug",
                "wheat leaf rust",
                "wheat loose smut",
                "wheat septoria blotch",
                "wheat stripe rust",
            ]
        )

register_all_pest()
