#!/usr/bin/env python

import numpy as np


Configs = {}

class CanonicalConfig:

    def __init__(self):

        self.width = 368
        self.height = 368

        self.stride = 8

        self.parts = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25" ]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.parts += ["background"]
        self.num_parts_with_background = len(self.parts)

        leftParts, rightParts = CanonicalConfig.ltr_parts(self.parts_dict)
        self.leftParts = leftParts
        self.rightParts = rightParts


        # this numbers probably copied from matlab they are 1.. based not 0.. based
        # self.limb_from =  ['0', '0', '2', '14', '15', '16', '1', '9', '10', '11', '14', '9', '5', '6', '7', '18', '19', '20', '7', '22', '23', '24']
        # self.limb_to =    ['1', '2', '14', '15', '16', '17', '9', '10', '11', '12', '4',  '3', '6', '7', '18', '19', '20', '21', '22', '23', '24', '25']
        
        self.limb_from =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25' ]
        self.limb_to =    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25' ]


        self.limb_from = [ self.parts_dict[n] for n in self.limb_from ]
        self.limb_to = [ self.parts_dict[n] for n in self.limb_to ]

        assert self.limb_from == [x-1 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
        assert self.limb_to == [x-1 for x in   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))

        self.paf_layers = 2*len(self.limbs_conn)
        self.heat_layers = self.num_parts
        self.num_layers = self.paf_layers + self.heat_layers + 1

        self.paf_start = 0
        self.heat_start = self.paf_layers
        self.bkg_start = self.paf_layers + self.heat_layers

        #self.data_shape = (self.height, self.width, 3)     # 368, 368, 3
        self.mask_shape = (self.height//self.stride, self.width//self.stride)  # 46, 46
        self.parts_shape = (self.height//self.stride, self.width//self.stride, self.num_layers)  # 46, 46, 57

        class TransformationParams:

            def __init__(self):
                self.target_dist = 0.6;
                self.scale_prob = 1;  # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
                self.scale_min = 0.5;
                self.scale_max = 1.1;
                self.max_rotate_degree = 40.
                self.center_perterb_max = 40.
                self.flip_prob = 0.5
                self.sigma = 7.
                self.paf_thre = 8.  # it is original 1.0 * stride in this program

        self.transform_params = TransformationParams()

    @staticmethod
    def ltr_parts(parts_dict):
        # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
        leftParts  = [ parts_dict[p] for p in ["1", "3", "8", "9", "10", "11", "12", "18", "19", "20", "21"] ]
        rightParts = [ parts_dict[p] for p in ["2", "4", "13", "14", "15", "16", "17", "22", "23", "24", "25"] ]
        return leftParts,rightParts



class COCOSourceConfig:


    def __init__(self, hdf5_source):

        self.hdf5_source = hdf5_source
        self.parts = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]

        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))

    def convert(self, meta, global_config):

        joints = np.array(meta['joints'])
        p_joints = np.array(meta['p_joints'])

        assert joints.shape[1] == len(self.parts)
        assert p_joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result[:,:,2]=3.  # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible

        for p in self.parts:
            coco_id = self.parts_dict[p]

            if p in global_config.parts_dict:
                global_id = global_config.parts_dict[p]
                # assert global_id!=1, "neck shouldn't be known yet"
                result[:,global_id,:]=joints[:,coco_id,:]


        result_p = np.zeros((p_joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result_p[:,:,2]=3.  # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible

        for p in self.parts:
            coco_id = self.parts_dict[p]

            if p in global_config.parts_dict:
                global_id = global_config.parts_dict[p]
                # assert global_id!=1, "neck shouldn't be known yet"
                result_p[:,global_id,:]=p_joints[:,coco_id,:]
        # there is no NECK in DeepMoCap

        # if 'neck' in global_config.parts_dict:
        #     neckG = global_config.parts_dict['neck']
        #     RshoC = self.parts_dict['Rsho']
        #     LshoC = self.parts_dict['Lsho']

        #     # no neck in coco database, we calculate it as average of shoulders
        #     # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        #     both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2] < 2)

        #     result[~both_shoulders_known, neckG, 2] = 2. # otherwise they will be 3. aka 'never marked in this dataset'

        #     result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
        #                                                 joints[both_shoulders_known, LshoC, 0:2]) / 2
        #     result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
        #                                                              joints[both_shoulders_known, LshoC, 2])

        meta['joints'] = result
        meta['p_joints'] = result_p

        return meta

    def convert_mask(self, mask, global_config, joints = None):

        mask = np.repeat(mask[:,:,np.newaxis], global_config.num_layers, axis=2)
        return mask

    def source(self):

        return self.hdf5_source



# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7


Configs["Canonical"] = CanonicalConfig


def GetConfig(config_name):

    config = Configs[config_name]()

    dct = config.parts[:]
    dct = [None]*(config.num_layers-len(dct)) + dct

    for (i,(fr,to)) in enumerate(config.limbs_conn):
        name = "%s->%s" % (config.parts[fr], config.parts[to])
        print(i, name)
        x = i*2
        y = i*2+1

        assert dct[x] is None
        dct[x] = name + ":x"
        assert dct[y] is None
        dct[y] = name + ":y"

    from pprint import pprint
    pprint(dict(zip(range(len(dct)), dct)))

    return config

if __name__ == "__main__":

    # test it
    foo = GetConfig("Canonical")
    print(foo.paf_layers, foo.heat_layers)


