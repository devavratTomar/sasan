from evaluations import SegAccuracy
import sys
from evaluations.generate_fake_data import generate_fake_data

if __name__=="__main__":
    method = sys.argv[1]
    direction = sys.argv[2]

    if direction == 'ct':
        direction = 'mr2ct'
        test_path = '../../img_modality_datasets/whole_heart/test_final_ct'
        seg_path  = './checkpoints_segmentation_mr'
        results = './resutls/mr2ct/'
    elif direction == 'mr':
        direction = 'ct2mr'
        test_path = '../../img_modality_datasets/whole_heart/test_final_mr'
        seg_path  = './checkpoints_segmentation_ct'
        results = './resutls/ct2mr/'

    if method == 'sasan':
        saved_model_path = './checkpoints_domain_adapt'
    elif method == 'cyclegan':
        saved_model_path = './checkpoints_cyclegan_baseline'
    elif method == 'ugatit':
        saved_model_path = '../../img_modality/UGATIT-pytorch/results'

    else:
        saved_model_path = 'none'
    
    evaluator = SegAccuracy(method, 'UNET', direction, test_path, saved_model_path, seg_path, results)
    evaluator.run_eval()
    # if method == 'generate_mr':
    #     generate_fake_data('mr', '../whole_heart/ct_train_filtered', '../whole_heart/fake_mr', './checkpoints_domain_adapt')
    # elif method == 'generate_ct':
    #     generate_fake_data('ct', '../whole_heart/mr_train_filtered', '../whole_heart/fake_ct', './checkpoints_domain_adapt')
    # else:
    #     print("invalid argument")