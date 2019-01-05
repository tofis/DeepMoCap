For testing the FCN model, please visit ["testing/"](/testing/) enabling the 3D optical data extraction from colorized depth and 3D optical flow input. The data should be appropriately formed and the DeepMoCap FCN model should be placed to ["testing/model/keras"](/testing/model/keras).

The proposed FCN is evaluated on the DMC2.5D dataset measuring mean Average Precision (mAP) for the entire set, based on Percentage of Correct Keypoints (PCK) thresholds (a = 0.05). The proposed method outperforms the competitive methods as shown in the table below.

| Method  | Total | Total (without end-reflectors) |
| :---: | :---: | :---: |
| CPM  | 92.16%  | 95.27% |
| CPM+PAFs  | 92.79\%  | 95.61% |
| CPM+PAFs + 3D OF  | 92.84\%  | 95.67% |
| **Proposed**  | **93.73%**  | **96.77%** |

![Logo](http://www.deepmocap.com/img/3D_all.png)

