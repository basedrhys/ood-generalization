from pathlib import Path
import clinicaldg.cxr.Constants as Constants

def validate_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    meta_dir = Path(Constants.meta_paths['MIMIC'])
    assert (meta_dir/'mimic-cxr-2.0.0-metadata.csv').is_file()
    assert (meta_dir/'mimic-cxr-2.0.0-negbio.csv').is_file()
    assert (meta_dir/'patients.csv').is_file()
    assert (img_dir/'p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()

def validate_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    meta_dir = Path(Constants.meta_paths['CXP'])
    assert (meta_dir/'map.csv').is_file()
    assert (img_dir/'train.csv').is_file()
    assert (img_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (img_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()

def validate_pad():
    img_dir = Path(Constants.image_paths['PAD'])
    meta_dir = Path(Constants.meta_paths['PAD'])
    assert (meta_dir/'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv').is_file()
    assert (img_dir/'185566798805711692534207714722577525271_qb3lyn.png').is_file()

def validate_nih():
    img_dir = Path(Constants.image_paths['NIH'])
    meta_dir = Path(Constants.meta_paths['NIH'])
    assert (meta_dir/'Data_Entry_2017.csv').is_file()
    assert (img_dir/'images/00002072_003.png').is_file()

def validate_splits():
    for dataset in Constants.df_paths:
        for split in Constants.df_paths[dataset]:
            assert Path(Constants.df_paths[dataset][split]).is_file()


def validate_all():
    validate_mimic()
    validate_cxp()
    validate_nih()
    validate_pad()
