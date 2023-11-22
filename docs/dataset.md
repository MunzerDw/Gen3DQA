## Data preparation

Since this code is based on [ScanRefer](https://github.com/daveredrum/ScanRefer), you can use the same 3D features. Please also refer to the ScanRefer data preparation.


1. Download the [ScanQA dataset](https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2?usp=sharing) under `data/scanqa/`. 

    ### Dataset format
    ```shell
    "scene_id": [ScanNet scene id, e.g. "scene0000_00"],
    "object_id": [ScanNet object ids (corresponds to "objectId" in ScanNet aggregation file), e.g. "[8]"],
    "object_names": [ScanNet object names (corresponds to "label" in ScanNet aggregation file), e.g. ["cabinet"]],
    "question_id": [...],
    "question": [...],
    "answers": [...],
    ```
2. Download the preprocessed [GLoVE embeddings file](http://kaldir.vc.in.tum.de/glove.p) and put it under `data/`.

3. Go to `code/minsu3d` and complete the last 2 steps.
    ```shell
    cd code/minsu3d
    ```

4. (From minsu3d, must be in `code/minsu3d`) Download the [ScanNet v2](http://www.scan-net.org/) dataset. To acquire the access to the dataset, please refer to their [instructions](https://github.com/ScanNet/ScanNet#scannet-data). You will get a `download-scannet.py` script after your request is approved:

    ```shell
    # about 10.7GB in total
    python download-scannet.py -o data/scannet --type _vh_clean_2.ply
    python download-scannet.py -o data/scannet --type _vh_clean.aggregation.json
    python download-scannet.py -o data/scannet --type _vh_clean_2.0.010000.segs.json
    ```

5. (From minsu3d, must be in `code/minsu3d`) Preprocess the data, it converts original meshes and annotations to `.pth` data:
    ```shell
    cd data/scannet
    python prepare_all_data.py data=scannet +raw_scan_path={PATH_TO_SCANNET_V2}/scans
    ```
    Then prepare the test scenes:
    ```shell
    python prepare_all_data.py data=scannet +raw_scan_path={PATH_TO_SCANNET_V2}/scans_test
    ```

6. Once you are done pretraining SoftGroup, move the ScanNet data from `code/minsu3d/data/scannet` to `data/scannet`. For training and validation we use the precomputed SoftGroup data in `data/precompute_softgroup_data` but for inference we use the original scene data.
