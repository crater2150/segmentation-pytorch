

def get_config():
    return \
    {
        # configuration home pc
        #"calamari_version": "old",
        #"venv":"/home/norbert/.pyenv/versions/venv38/bin/activate",
        #"pythonpath": "/home/norbert/uni/page-segment/calamari-larex",
        #"model": "/home/norbert/share/BLOCR/unused_models/camgen_noadj_cf/best.ckpt.json",
        #"model": "/home/norbert/uni/page-segment/calamari_models_experimental-main/c1_latin-script-hist-3/0.ckpt.json",
        #"model":"/home/norbert/share/datasets/OCR/Troja5RepCV/mod_transf_new/best.ckpt.json",
        #"model": "/home/norbert/share/datasets/OCR/camgen/mod_noadj/best.ckpt.json",
        #"ocr_binarize": True,

        # configuration for work pc
        "calamari_version": "old",
        "venv": "/home/norbert/projects/calamari/venv3/bin/activate",
        "pythonpath": "/home/norbert/projects/calamari",
        # "model": "/home/norbert/share/BLOCR/ocr_models/trcf_noaug/best.ckpt.json",
        "model": "/home/norbert/projects/calamari_models/gt4histocr/1.ckpt.json",
        "ocr_binarize": True,
    }
